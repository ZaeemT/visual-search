"""Run one query image through the retrieval pipeline and print the ranking.

The same Retriever the API will use, driven from the terminal — so the pipeline
can be checked before any HTTP is involved.

Usage:
    python scripts/6_search.py data/queries/images/000003.jpg
    python scripts/6_search.py data/queries/images/000003.jpg --top-k 10
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.search.index import INDEX_ROOT  # noqa: E402
from app.search.retriever import CATALOGUE, Retriever  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default=None, help="defaults to the only built index")
    parser.add_argument("--index-root", type=Path, default=INDEX_ROOT)
    parser.add_argument("--catalogue", type=Path, default=CATALOGUE)
    args = parser.parse_args()

    if not args.image.exists():
        raise SystemExit(f"{args.image} not found")

    retriever = Retriever.open(
        model=args.model, index_root=args.index_root, catalogue_dir=args.catalogue
    )
    print(f"model     : {retriever.embedder.name} on {retriever.embedder.device}")
    print(f"catalogue : {retriever.index.size:,} items, dim {retriever.index.dim}")
    print(f"query     : {args.image}\n")

    response = retriever.search_bytes(args.image.read_bytes(), top_k=args.top_k)

    for result in response.results:
        meta = result.metadata
        colours = ", ".join(meta["colors"]) or "-"
        print(f"  {result.rank}. {result.score:.3f}  {result.item_id}  {meta['name']}")
        print(f"       {colours} · {meta['fit']} · {meta['length']}")
        print(f"       {result.file}")

    t = response.timings_ms
    print(f"\ntiming    : embed {t['embed']} ms + search {t['search']} ms = {t['total']} ms")


if __name__ == "__main__":
    main()
