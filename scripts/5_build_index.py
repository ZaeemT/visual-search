"""Build the catalogue index: embed every image once, store the matrix.

Reads the catalogue metadata in its stored order, embeds the images in batches
on the GPU, and writes vectors.npy / ids.json / manifest.json to data/index.
The API then loads that matrix instead of embedding anything at startup.

Usage:
    python scripts/5_build_index.py
    python scripts/5_build_index.py --model clip-vit-b32 --batch-size 32
    python scripts/5_build_index.py --out data/index_small --limit 500
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.search.embedders import MODELS, get_embedder  # noqa: E402
from app.search.index import (  # noqa: E402
    EMBEDDINGS_FILE,
    INDEX_ROOT,
    VectorIndex,
    build_index,
    index_dir,
)

CATALOGUE = Path("data/catalogue")
DEFAULT_MODEL = "marqo-fashionsiglip"


def load_catalogue(directory: Path, limit: int | None) -> tuple[list[dict], dict]:
    meta = directory / "metadata.parquet"
    if not meta.exists():
        raise SystemExit(
            f"{meta} not found — build the catalogue first:\n"
            f"    python scripts/3_build_catalogue.py"
        )

    # Fixed order: whatever order the metadata table is stored in is the order
    # the matrix rows take, so ids[i] always describes vectors[i].
    rows = pq.read_table(meta).to_pylist()
    if limit:
        rows = rows[:limit]

    provenance = directory / "catalogue_manifest.json"
    manifest = json.loads(provenance.read_text(encoding="utf-8")) if provenance.exists() else {}
    if limit:
        manifest = {**manifest, "items": len(rows), "note": f"limited to {limit} items"}
    return rows, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    parser.add_argument("--catalogue", type=Path, default=CATALOGUE)
    parser.add_argument("--out", type=Path, default=INDEX_ROOT,
                        help="index root; the model gets its own directory inside it")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="index only the first N items")
    parser.add_argument("--device", default=None, help="mps / cpu / cuda (auto by default)")
    args = parser.parse_args()

    rows, catalogue_manifest = load_catalogue(args.catalogue, args.limit)
    print(f"catalogue : {len(rows):,} items from {args.catalogue}")

    load_started = time.perf_counter()
    embedder = get_embedder(args.model, device=args.device, batch_size=args.batch_size)
    print(
        f"model     : {args.model} on {embedder.device}, dim={embedder.dim} "
        f"(loaded in {time.perf_counter() - load_started:.1f}s)\n"
    )

    def progress(done, total):
        print(f"  embedding {done}/{total}", end="\r", flush=True)

    index = build_index(
        embedder,
        rows,
        batch_size=args.batch_size,
        catalogue_manifest=catalogue_manifest,
        on_progress=progress,
    )
    directory = index.save(index_dir(embedder.model_id, root=args.out))

    build = index.manifest["build"]
    embeddings_bytes = index.manifest["files"][EMBEDDINGS_FILE]
    print(f"\n\nindexed   : {index.size:,} items, dim {index.dim}")
    print(f"time      : {build['seconds']}s  ({build['images_per_second']} images/sec)")
    print(f"embeddings: {embeddings_bytes / 1e6:.1f} MB "
          f"({embeddings_bytes / index.size:.0f} bytes/item)")
    print(f"written   : {directory}/")

    # A load-and-query round trip, so a broken index fails here and not in the API.
    check = VectorIndex.load(directory)
    hits = check.search(check.vectors[0], top_k=3)
    print(f"\nself-check: item {check.ids[0]} -> "
          + ", ".join(f"{h.item_id}({h.score:.3f})" for h in hits))


if __name__ == "__main__":
    main()
