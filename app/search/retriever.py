"""Query -> ranked catalogue items.

Ties the three stored pieces together: the embedding model, the index matrix,
and the catalogue metadata. Built once at startup and reused for every query,
because loading the model is seconds and the search itself is under a
millisecond — doing it per request would be a thousandfold waste.

The one invariant worth stating: a query is only comparable to the index if it
goes through the *same* model, revision and preprocessing that built it. The
manifest records all three, and `Retriever.open` refuses to run on a mismatch
rather than returning quietly meaningless results.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageOps, UnidentifiedImageError

from app.search.embedders import MODELS, get_embedder
from app.search.index import INDEX_ROOT, VectorIndex, index_dir

CATALOGUE = Path("data/catalogue")

# Uploads are decoded before they are trusted, so bound the work: a small file
# can expand into an enormous bitmap (a "decompression bomb"). Pillow warns
# past ~89M pixels by default; this is a firmer, explicit ceiling.
MAX_PIXELS = 50_000_000

# Downscale before the model's own preprocessing. The encoder only ever sees
# 224x224, so carrying a 6000px phone photo through resize costs time and
# memory for no gain in quality.
MAX_SIDE = 1024


class BadImage(ValueError):
    """The upload could not be read as an image."""


def load_query_image(data: bytes) -> Image.Image:
    """Turn raw upload bytes into a clean RGB image.

    Four things are fixed here, all of which silently degrade results otherwise:

    * EXIF orientation — phone photos carry a rotation flag rather than rotated
      pixels. Ignore it and a portrait shot reaches the model on its side.
    * Colour mode — PNG with alpha, palette GIF, CMYK and greyscale all need
      converting to RGB; the alpha channel in particular composites as black.
    * Pixel count — refuse absurd images instead of exhausting memory.
    * Size — downscale to something sane before preprocessing.
    """
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except UnidentifiedImageError as exc:
        raise BadImage("file is not a readable image") from exc
    except OSError as exc:  # truncated or corrupt data
        raise BadImage(f"image could not be decoded: {exc}") from exc

    width, height = image.size
    if width * height > MAX_PIXELS:
        raise BadImage(f"image is too large ({width}x{height} pixels)")

    image = ImageOps.exif_transpose(image)  # honour the rotation flag
    image = image.convert("RGB")            # drop alpha / palette / CMYK
    image.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)  # no-op if already small
    return image


@dataclass
class Result:
    rank: int
    item_id: str
    score: float
    file: str
    metadata: dict


@dataclass
class SearchResponse:
    results: list[Result]
    timings_ms: dict
    catalogue_size: int


class Retriever:
    """Everything a query needs, loaded once."""

    def __init__(self, embedder, index: VectorIndex, catalogue: dict[str, dict]):
        self.embedder = embedder
        self.index = index
        self.catalogue = catalogue

    @classmethod
    def open(
        cls,
        model: str | None = None,
        index_root: Path = INDEX_ROOT,
        catalogue_dir: Path = CATALOGUE,
        device: str | None = None,
    ) -> "Retriever":
        """Load the index, then the model that built it."""
        model = model or _only_index(index_root)
        directory = index_dir(MODELS[model]["model_id"], root=index_root)
        index = VectorIndex.load(directory)

        embedder = get_embedder(model, device=device)
        _assert_compatible(index.manifest, embedder)

        table = pq.read_table(catalogue_dir / "metadata.parquet").to_pylist()
        catalogue = {row["id"]: row for row in table}

        missing = [i for i in index.ids if i not in catalogue]
        if missing:
            raise SystemExit(
                f"{len(missing)} indexed IDs are missing from {catalogue_dir} "
                f"(first: {missing[0]}) — rebuild the index"
            )
        return cls(embedder, index, catalogue)

    def search(self, image: Image.Image, top_k: int = 5) -> SearchResponse:
        """Embed the query, score it against the catalogue, return the top K."""
        started = time.perf_counter()
        vector = self.embedder.embed_images([image])[0]
        embedded = time.perf_counter()

        hits = self.index.search(vector, top_k=top_k)
        searched = time.perf_counter()

        results = [
            Result(
                rank=hit.rank,
                item_id=hit.item_id,
                score=hit.score,
                file=self.catalogue[hit.item_id]["file"],
                metadata=self.catalogue[hit.item_id],
            )
            for hit in hits
        ]

        return SearchResponse(
            results=results,
            timings_ms={
                "embed": round((embedded - started) * 1000, 2),
                "search": round((searched - embedded) * 1000, 2),
                "total": round((searched - started) * 1000, 2),
            },
            catalogue_size=self.index.size,
        )

    def search_bytes(self, data: bytes, top_k: int = 5) -> SearchResponse:
        """Convenience for callers holding raw upload bytes."""
        return self.search(load_query_image(data), top_k=top_k)


def _only_index(root: Path) -> str:
    """Infer the model from whatever index exists, when none was named."""
    available = {
        name: spec for name, spec in MODELS.items() if index_dir(spec["model_id"], root).exists()
    }
    if not available:
        raise SystemExit(
            f"no index under {root} — build one first:\n    python scripts/5_build_index.py"
        )
    if len(available) > 1:
        raise SystemExit(
            f"several indexes under {root} ({', '.join(available)}) — name one with --model"
        )
    return next(iter(available))


def _assert_compatible(manifest: dict, embedder) -> None:
    """Refuse to query an index built by different weights or preprocessing.

    Comparing a vector from one encoder against a matrix from another produces
    numbers, not answers — every score would be meaningless but nothing would
    raise. Cheap to check, so check.
    """
    built = manifest.get("embedder", {})
    now = embedder.describe()
    for field in ("model_id", "revision", "dim", "preprocessing"):
        if built.get(field) != now.get(field):
            raise SystemExit(
                f"index/model mismatch on {field!r}:\n"
                f"  index was built with : {built.get(field)}\n"
                f"  loaded model provides: {now.get(field)}\n"
                f"rebuild the index or load the matching model"
            )
