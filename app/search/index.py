"""The catalogue index: a matrix of embeddings, plus the IDs and the manifest.

One directory per model, so indexes built from different checkpoints sit side
by side and switching models never silently reuses the wrong matrix:

    data/index/marqo-fashionSigLIP/
        embeddings.npy   (n, dim) float32, rows L2-normalised
        ids.npy          row -> item ID
        manifest.json    what was built, from what, with which model, how long it took

The matrix is deliberately a plain .npy rather than a library-specific index
format. Nothing here is coupled to the search method: the same array loads
straight into FAISS (`index.add(vectors)`) whenever brute force stops being
enough, so scaling up is an implementation change rather than a rewrite, and
no re-embedding is needed.

Search is brute force — every query is compared against all n rows. That is
O(n·d) per query, which is honest at this catalogue size and is measured
rather than assumed (see scripts/6_benchmark.py).
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

INDEX_ROOT = Path("data/index")

EMBEDDINGS_FILE = "embeddings.npy"
IDS_FILE = "ids.npy"
MANIFEST_FILE = "manifest.json"


def index_dir(model_id: str, root: Path = INDEX_ROOT) -> Path:
    """Where one model's index lives, e.g. data/index/marqo-fashionSigLIP.

    Named after the checkpoint rather than our registry key, so the directory
    says exactly which weights produced it.
    """
    return root / model_id.split("/")[-1]


@dataclass
class SearchHit:
    rank: int
    item_id: str
    score: float
    row: int


@dataclass
class VectorIndex:
    """An in-memory matrix of unit vectors with the IDs that label its rows."""

    vectors: np.ndarray
    ids: list[str]
    manifest: dict

    def __post_init__(self):
        if len(self.vectors) != len(self.ids):
            raise ValueError(
                f"{len(self.vectors)} vectors but {len(self.ids)} ids — index is inconsistent"
            )

    @property
    def size(self) -> int:
        return len(self.ids)

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def search(self, query: np.ndarray, top_k: int = 5) -> list[SearchHit]:
        """Rank the catalogue against one query vector.

        Both sides are unit vectors, so the dot product is cosine similarity —
        no per-query normalisation, and the whole comparison is one matrix
        multiply. argpartition finds the top k in O(n) instead of sorting all
        n scores in O(n log n); only the k winners are then sorted.
        """
        if query.ndim == 2:
            query = query[0]
        scores = self.vectors @ query

        k = min(top_k, self.size)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]

        return [
            SearchHit(rank=r + 1, item_id=self.ids[int(i)], score=float(scores[int(i)]), row=int(i))
            for r, i in enumerate(top)
        ]

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / EMBEDDINGS_FILE, self.vectors)
        # IDs as an array rather than JSON: same format as the matrix, loads
        # with the same call, and the row count is checked on the way back in.
        np.save(directory / IDS_FILE, np.array(self.ids, dtype=np.str_))

        # Recorded after writing, so the sizes are the real ones on disk. Kept on
        # the object too, so a caller sees the same manifest that was saved.
        self.manifest["files"] = {
            name: (directory / name).stat().st_size
            for name in (EMBEDDINGS_FILE, IDS_FILE)
        }
        (directory / MANIFEST_FILE).write_text(
            json.dumps(self.manifest, indent=2), encoding="utf-8"
        )
        return directory

    @classmethod
    def load(cls, directory: Path) -> "VectorIndex":
        if not (directory / EMBEDDINGS_FILE).exists():
            raise SystemExit(
                f"no index at {directory} — build it first:\n"
                f"    python scripts/5_build_index.py"
            )
        return cls(
            vectors=np.load(directory / EMBEDDINGS_FILE),
            ids=np.load(directory / IDS_FILE).tolist(),
            manifest=json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8")),
        )


def build_index(
    embedder,
    rows: list[dict],
    batch_size: int = 64,
    catalogue_manifest: dict | None = None,
    on_progress=None,
) -> VectorIndex:
    """Embed every catalogue image, preserving the order of `rows`.

    Vectors are collected per batch and stacked once at the end, so row i of
    the matrix is always rows[i] — the IDs and the matrix cannot drift apart.
    """
    started = time.perf_counter()
    chunks = []

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        images = [Image.open(r["file"]) for r in batch]
        chunks.append(embedder.embed_images(images))
        for image in images:
            image.close()
        if on_progress:
            on_progress(min(start + batch_size, len(rows)), len(rows))

    elapsed = time.perf_counter() - started
    vectors = (
        np.vstack(chunks) if chunks else np.empty((0, embedder.dim), dtype=np.float32)
    )

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "items": len(rows),
        "embedder": embedder.describe(),
        "build": {
            "seconds": round(elapsed, 2),
            "images_per_second": round(len(rows) / elapsed, 1) if elapsed else None,
            "batch_size": batch_size,
            "device": embedder.device,
            "machine": f"{platform.machine()} / {platform.system()}",
        },
        "catalogue": catalogue_manifest or {},
        "search": {
            "method": "brute force (exact)",
            "metric": "cosine via dot product on unit vectors",
            "complexity": "O(n·d) per query",
        },
    }

    return VectorIndex(vectors=vectors, ids=[r["id"] for r in rows], manifest=manifest)
