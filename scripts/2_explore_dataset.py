"""Inspect a parquet shard that has already been downloaded to data/raw.

Reads the shard from disk (no network, no streaming), prints the schema and
the structure of the first few records, and saves their images so they can be
eyeballed before deciding which subset becomes the catalogue.

Download a shard first with scripts/1_download_shard.py.

Usage:
    python scripts/2_explore_dataset.py
    python scripts/2_explore_dataset.py --rows 8
    python scripts/2_explore_dataset.py --parquet data/raw/fashion550k-train-00001-of-00007.parquet
"""

import argparse
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

DEFAULT_PARQUET = Path("data/raw/myntra-train-00001-of-00008.parquet")


def is_image_field(value) -> bool:
    """True for a decoded PIL image or the {'bytes':..., 'path':...} struct."""
    if hasattr(value, "mode") and hasattr(value, "save"):
        return True
    return isinstance(value, dict) and isinstance(value.get("bytes"), (bytes, bytearray))


def to_pil(value) -> Image.Image:
    if hasattr(value, "mode"):
        return value
    return Image.open(io.BytesIO(value["bytes"]))


def describe(value, depth=0):
    """Return a short, human-readable description of a record field."""
    indent = "  " * depth
    if is_image_field(value):
        img = to_pil(value)
        path = value.get("path") if isinstance(value, dict) else None
        return f"{indent}image mode={img.mode} size={img.size} path={path}"
    if isinstance(value, dict):
        lines = [f"{indent}dict({len(value)} keys)"]
        for key, sub in value.items():
            lines.append(f"{indent}  {key}:")
            lines.append(describe(sub, depth + 2))
        return "\n".join(lines)
    if isinstance(value, list):
        head = f"{indent}list(len={len(value)})"
        if not value:
            return head
        return head + "\n" + describe(value[0], depth + 1) + f"\n{indent}  ... (first item shown)"
    text = repr(value)
    if len(text) > 160:
        text = text[:160] + "..."
    return f"{indent}{type(value).__name__} = {text}"


def save_image(record, out_dir: Path, prefix: str, idx: int) -> str | None:
    """Save the first image-like field of a record. Returns the saved path."""
    for key, value in record.items():
        if is_image_field(value):
            path = out_dir / f"{prefix}_{idx:02d}_{key}.jpg"
            to_pil(value).convert("RGB").save(path, quality=90)
            return str(path)
    return None


def explore(parquet: Path, rows: int, out_dir: Path):
    if not parquet.exists():
        raise SystemExit(
            f"{parquet} not found — download a shard first:\n"
            f"    python scripts/2_download_shard.py"
        )

    prefix = parquet.stem.split("-")[0]  # e.g. "myntra"

    print("=" * 70)
    print(f"FILE: {parquet}  ({parquet.stat().st_size / 1e9:.2f} GB)")
    print("=" * 70)

    pf = pq.ParquetFile(parquet)
    print(f"\nrows: {pf.metadata.num_rows:,}   row groups: {pf.metadata.num_row_groups}")
    print("\n--- schema ---")
    print(pf.schema_arrow)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Read only the first batch. The shard is one row group of ~60k rows with
    # image bytes inline, so reading the whole table would exhaust memory.
    batch = next(pf.iter_batches(batch_size=rows))
    for idx, record in enumerate(batch.to_pylist()):
        print(f"\n--- row {idx} ---")
        for key, value in record.items():
            print(f"{key}:")
            print(describe(value, depth=1))

        saved = save_image(record, out_dir, prefix, idx)
        if saved:
            print(f"[saved] {saved}")

        scalars = {k: v for k, v in record.items() if not is_image_field(v)}
        print("[json] " + json.dumps(scalars, ensure_ascii=False, default=str)[:1500])

    print(f"\nsample images written to: {out_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--rows", type=int, default=5, help="rows to inspect")
    parser.add_argument("--out", type=Path, default=Path("data/samples"))
    args = parser.parse_args()

    explore(args.parquet, args.rows, args.out)


if __name__ == "__main__":
    main()
