"""Compare embedding models on the same catalogue and the same queries.

For each model: embed the 4k catalogue, embed the held-out test-split queries,
retrieve top-5 by cosine similarity, and write a side-by-side HTML contact
sheet so the results can be judged by eye — which is the only thing that
settles "do the similar items actually look similar".

Catalogue embeddings are cached per model under data/embeddings/, so a rerun
re-scores without re-embedding.

Retrieval here is brute force: every query is compared against every catalogue
vector, which is O(n) in catalogue size. At n=4000 that is one small matrix
multiply. The timings printed per model are the measurement this choice will
later be defended with.

Usage:
    python scripts/4_bakeoff.py
    python scripts/4_bakeoff.py --models clip-vit-b32,marqo-fashionsiglip --top-k 5
"""

import argparse
import html
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.search.embedders import MODELS, get_embedder  # noqa: E402

CATALOGUE = Path("data/catalogue")
QUERIES = Path("data/queries")
EMBED_DIR = Path("data/embeddings")
REPORT = Path("metrics/bakeoff")


def load_set(directory: Path) -> list[dict]:
    meta = directory / "metadata.parquet"
    if not meta.exists():
        raise SystemExit(
            f"{meta} not found — build it first:\n"
            f"    python scripts/3_build_catalogue.py --out {directory}"
        )
    return pq.read_table(meta).to_pylist()


def embed_files(embedder, rows: list[dict], label: str) -> tuple[np.ndarray, float]:
    """Embed every row's image file, reporting progress. Returns (vectors, seconds)."""
    start = time.perf_counter()
    vectors = []
    batch = embedder.batch_size
    for i in range(0, len(rows), batch):
        images = [Image.open(r["file"]) for r in rows[i : i + batch]]
        vectors.append(embedder.embed_images(images))
        done = min(i + batch, len(rows))
        print(f"  {label}: {done}/{len(rows)}", end="\r", flush=True)
    elapsed = time.perf_counter() - start
    print(f"  {label}: {len(rows)}/{len(rows)} in {elapsed:.1f}s")
    return np.vstack(vectors), elapsed


def catalogue_vectors(embedder, rows: list[dict]) -> tuple[np.ndarray, float | None]:
    """Embed the catalogue, or reuse the cached vectors for this model."""
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    cache = EMBED_DIR / f"{embedder.name}.npy"
    if cache.exists():
        vectors = np.load(cache)
        if len(vectors) == len(rows):
            print(f"  catalogue: cached {vectors.shape} from {cache}")
            return vectors, None
    vectors, elapsed = embed_files(embedder, rows, "catalogue")
    np.save(cache, vectors)
    return vectors, elapsed


def search(catalogue: np.ndarray, queries: np.ndarray, top_k: int):
    """Brute-force cosine search. Rows are unit vectors, so a dot product is cosine.

    Returns (indices, scores, mean seconds per query).
    """
    start = time.perf_counter()
    scores = queries @ catalogue.T                      # (q, n) all similarities
    top = np.argpartition(-scores, top_k, axis=1)[:, :top_k]   # O(n) per query
    ordered = np.take_along_axis(
        top, np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1), axis=1
    )
    per_query = (time.perf_counter() - start) / max(len(queries), 1)
    return ordered, np.take_along_axis(scores, ordered, axis=1), per_query


def describe(row: dict) -> str:
    bits = [row["name"] or "?"]
    if row["colors"]:
        bits.append(", ".join(row["colors"]))
    return " · ".join(bits)


def write_report(results: dict, queries: list[dict], catalogue: list[dict], path: Path):
    """One row per query, one block per model, so models are compared side by side."""
    css = """
    body{font:14px system-ui;margin:24px;background:#fafafa}
    h1{font-size:20px} h2{font-size:15px;margin:18px 0 6px}
    .q{background:#fff;border:1px solid #ddd;border-radius:8px;padding:16px;margin-bottom:24px}
    .strip{display:flex;gap:10px;flex-wrap:wrap}
    figure{margin:0;width:120px} img{width:120px;height:160px;object-fit:cover;border-radius:4px}
    figcaption{font-size:11px;color:#444;margin-top:4px;line-height:1.3}
    .query img{outline:3px solid #2563eb}
    table{border-collapse:collapse;margin-bottom:24px} td,th{border:1px solid #ddd;padding:6px 10px;text-align:left}
    """
    out = [f"<!doctype html><meta charset='utf-8'><style>{css}</style>", "<h1>Embedding model bake-off</h1>"]

    out.append("<table><tr><th>model</th><th>dim</th><th>catalogue embed</th><th>query latency (brute force)</th></tr>")
    for name, r in results.items():
        embed_time = f"{r['embed_seconds']:.1f}s" if r["embed_seconds"] else "cached"
        out.append(
            f"<tr><td>{name}</td><td>{r['dim']}</td><td>{embed_time}</td>"
            f"<td>{r['per_query'] * 1000:.2f} ms</td></tr>"
        )
    out.append("</table>")

    for qi, q in enumerate(queries):
        out.append("<div class='q'>")
        out.append(
            f"<div class='strip query'><figure><img src='{html.escape(rel(q['file'], path))}'>"
            f"<figcaption><b>query {qi}</b><br>{html.escape(describe(q))}</figcaption></figure></div>"
        )
        for name, r in results.items():
            out.append(f"<h2>{html.escape(name)}</h2><div class='strip'>")
            for idx, score in zip(r["indices"][qi], r["scores"][qi]):
                item = catalogue[int(idx)]
                out.append(
                    f"<figure><img src='{html.escape(rel(item['file'], path))}'>"
                    f"<figcaption>{score:.3f}<br>{html.escape(describe(item))}</figcaption></figure>"
                )
            out.append("</div>")
        out.append("</div>")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")


def rel(file: str, report: Path) -> str:
    """Image path relative to the report, so the HTML opens straight from disk."""
    import os

    return os.path.relpath(Path(file).resolve(), report.parent.resolve())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None, help="mps / cpu / cuda (auto by default)")
    parser.add_argument("--report", type=Path, default=REPORT / "report.html")
    args = parser.parse_args()

    catalogue = load_set(CATALOGUE)
    queries = load_set(QUERIES)
    print(f"catalogue: {len(catalogue):,} items   queries: {len(queries)}\n")

    results = {}
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"=== {name} ===")
        load_start = time.perf_counter()
        embedder = get_embedder(name, device=args.device, batch_size=args.batch_size)
        print(f"  loaded on {embedder.device}, dim={embedder.dim} ({time.perf_counter() - load_start:.1f}s)")

        cat_vectors, embed_seconds = catalogue_vectors(embedder, catalogue)
        query_vectors, _ = embed_files(embedder, queries, "queries")
        indices, scores, per_query = search(cat_vectors, query_vectors, args.top_k)

        print(f"  brute-force search: {per_query * 1000:.2f} ms/query over {len(catalogue):,} items\n")
        results[name] = {
            "dim": embedder.dim,
            "embed_seconds": embed_seconds,
            "per_query": per_query,
            "indices": indices,
            "scores": scores,
        }
        del embedder  # free the model before loading the next one

    write_report(results, queries, catalogue, args.report)
    print(f"report: {args.report.resolve()}")


if __name__ == "__main__":
    main()
