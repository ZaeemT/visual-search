"""Image embedders behind one interface.

Every model in the bake-off is reached through `Embedder`, so the indexing
script, the bake-off and the API all call the same two methods and switching
models is a config change rather than a code change.

Contract: `embed_images` takes PIL images and returns an (n, dim) float32
array whose rows are L2-normalised. Normalised rows mean a dot product *is*
cosine similarity, so retrieval is one matrix multiply with no per-query
division.

Models download on first use and are cached under data/hf (see app.core.hf_cache);
later runs are offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np
import torch
from PIL import Image

from app.core import hf_cache  # noqa: F401  sets HF_HOME before any HF import


def pick_device(requested: str | None = None) -> str:
    """MPS on Apple silicon, CUDA where present, else CPU."""
    if requested:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def cached_revision(model_id: str) -> str:
    """The exact commit of the checkpoint on disk.

    Recorded in the index manifest: "marqo-fashionSigLIP" is a moving target,
    a commit hash is not, and an index is only comparable to another one built
    from the same weights.
    """
    from huggingface_hub import constants

    repo = Path(constants.HF_HUB_CACHE) / f"models--{model_id.replace('/', '--')}"
    ref = repo / "refs" / "main"
    if ref.exists():
        return ref.read_text().strip()
    snapshots = sorted((repo / "snapshots").glob("*"))
    return snapshots[-1].name if snapshots else "unknown"


class Embedder(Protocol):
    """The one interface every model is used through."""

    name: str
    dim: int

    def embed_images(self, images: Iterable[Image.Image]) -> np.ndarray:
        """PIL images -> (n, dim) float32, rows L2-normalised."""

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        """Strings -> (n, dim) float32, rows L2-normalised."""

    def describe(self) -> dict:
        """Facts about this embedder, for the index manifest."""


@dataclass
class HFEmbedder:
    """A CLIP-style dual encoder loaded through transformers.

    All three bake-off models expose the same `get_image_features` /
    `get_text_features` pair, so one implementation covers them; they differ
    only in checkpoint, embedding width and whether custom code is needed.
    """

    name: str
    model_id: str
    trust_remote_code: bool = False
    device: str = field(default_factory=pick_device)
    batch_size: int = 64

    def __post_init__(self):
        from transformers import AutoModel, AutoProcessor

        self.model = AutoModel.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code
        )
        self.dim = int(self._forward_images([Image.new("RGB", (224, 224))]).shape[1])

    @staticmethod
    def _as_tensor(output) -> torch.Tensor:
        """Pull the embedding out of whatever the model returned.

        transformers 5 wraps CLIP's features in a BaseModelOutputWithPooling
        (the projected vector is `pooler_output`), while the Marqo checkpoints
        run their own modelling code and hand back a plain tensor.
        """
        if isinstance(output, torch.Tensor):
            return output
        for key in ("image_embeds", "text_embeds", "pooler_output"):
            value = getattr(output, key, None)
            if value is not None:
                return value
        raise TypeError(f"no embedding tensor in {type(output).__name__}")

    @classmethod
    def _l2_normalise(cls, output) -> np.ndarray:
        vectors = cls._as_tensor(output).float()
        vectors = vectors / vectors.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return vectors.cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def _forward_images(self, images: list[Image.Image]) -> np.ndarray:
        inputs = self.processor(
            images=[im.convert("RGB") for im in images], return_tensors="pt"
        ).to(self.device)
        return self._l2_normalise(self.model.get_image_features(**inputs))

    @torch.inference_mode()
    def embed_images(self, images: Iterable[Image.Image]) -> np.ndarray:
        images = list(images)
        chunks = [
            self._forward_images(images[i : i + self.batch_size])
            for i in range(0, len(images), self.batch_size)
        ]
        return np.vstack(chunks) if chunks else np.empty((0, self.dim), dtype=np.float32)

    @torch.inference_mode()
    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        out = []
        for i in range(0, len(texts), self.batch_size):
            inputs = self.processor(
                text=texts[i : i + self.batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            out.append(self._l2_normalise(self.model.get_text_features(**inputs)))
        return np.vstack(out) if out else np.empty((0, self.dim), dtype=np.float32)

    def describe(self) -> dict:
        proc = self.processor.image_processor
        crop = getattr(proc, "crop_size", None)
        size = getattr(proc, "size", None)
        side = getattr(crop, "height", None) or getattr(size, "shortest_edge", None)
        return {
            "model": self.name,
            "model_id": self.model_id,
            "revision": cached_revision(self.model_id),
            "backend": "transformers",
            "dim": self.dim,
            "normalised": True,
            "preprocessing": {
                # Resize-then-centre-crop keeps the aspect ratio but discards the
                # edges of a non-square image.
                "method": "crop" if getattr(proc, "do_center_crop", False) else "squash",
                "size": [side, side],
            },
        }


@dataclass
class OpenClipEmbedder:
    """A checkpoint loaded through open_clip rather than transformers.

    The Marqo models ship custom code that builds an open_clip model inside
    __init__. transformers 5 initialises weights on the `meta` device, so that
    inner build fails with "Cannot copy out of meta tensor". Loading them the
    way Marqo documents — straight through open_clip — avoids the clash, and
    the interface above hides the difference from every caller.
    """

    name: str
    model_id: str
    device: str = field(default_factory=pick_device)
    batch_size: int = 64

    def __post_init__(self):
        import open_clip

        hub = f"hf-hub:{self.model_id}"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(hub)
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(hub)
        self.dim = int(self.embed_images([Image.new("RGB", (224, 224))]).shape[1])

    @staticmethod
    def _l2_normalise(vectors: torch.Tensor) -> np.ndarray:
        vectors = vectors.float()
        vectors = vectors / vectors.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return vectors.cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def embed_images(self, images: Iterable[Image.Image]) -> np.ndarray:
        images = list(images)
        out = []
        for i in range(0, len(images), self.batch_size):
            batch = torch.stack(
                [self.preprocess(im.convert("RGB")) for im in images[i : i + self.batch_size]]
            ).to(self.device)
            out.append(self._l2_normalise(self.model.encode_image(batch)))
        return np.vstack(out) if out else np.empty((0, self.dim), dtype=np.float32)

    @torch.inference_mode()
    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        out = []
        for i in range(0, len(texts), self.batch_size):
            tokens = self.tokenizer(texts[i : i + self.batch_size]).to(self.device)
            out.append(self._l2_normalise(self.model.encode_text(tokens)))
        return np.vstack(out) if out else np.empty((0, self.dim), dtype=np.float32)

    def describe(self) -> dict:
        # The transform pipeline is the source of truth: a Resize to a square
        # squashes the aspect ratio, a CenterCrop discards the edges instead.
        steps = list(getattr(self.preprocess, "transforms", []))
        names = [type(s).__name__ for s in steps]
        resize = next((s for s in steps if type(s).__name__ == "Resize"), None)
        size = getattr(resize, "size", None)
        return {
            "model": self.name,
            "model_id": self.model_id,
            "revision": cached_revision(self.model_id),
            "backend": "open_clip",
            "dim": self.dim,
            "normalised": True,
            "preprocessing": {
                "method": "crop" if "CenterCrop" in names else "squash",
                "size": list(size) if isinstance(size, (tuple, list)) else [size, size],
            },
        }


# The bake-off line-up: a generic baseline plus two fashion-domain models.
# Keys are what --model and the EMBEDDING_MODEL setting accept; `backend`
# picks the loader, which is the only thing that differs between them.
MODELS: dict[str, dict] = {
    "clip-vit-b32": {
        "backend": HFEmbedder,
        "model_id": "openai/clip-vit-base-patch32",
    },
    "marqo-fashionclip": {
        "backend": OpenClipEmbedder,
        "model_id": "Marqo/marqo-fashionCLIP",
    },
    "marqo-fashionsiglip": {
        "backend": OpenClipEmbedder,
        "model_id": "Marqo/marqo-fashionSigLIP",
    },
}


def get_embedder(name: str, device: str | None = None, batch_size: int = 64) -> Embedder:
    """Build the embedder registered under `name`."""
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; options: {', '.join(MODELS)}")
    spec = dict(MODELS[name])
    backend = spec.pop("backend")
    return backend(name=name, device=pick_device(device), batch_size=batch_size, **spec)
