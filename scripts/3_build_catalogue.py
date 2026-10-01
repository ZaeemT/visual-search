"""Turn one or more downloaded parquet shards into a catalogue on disk.

Two outputs:
  data/catalogue/images/000123.jpg   one file per item, named by a stable ID
  data/catalogue/metadata.parquet    one row per item

The JPEG bytes in the parquet are written straight to disk — decoding and
re-encoding would cost quality (JPEG is lossy, so a re-save is a second
generation loss) and time. Pillow is only used to sniff the format so the
extension is honest, which reads the header rather than the whole image.

The VQA columns are dropped: they are question/answer pairs for a different
task and nothing in the retrieval pipeline reads them.

With several shards, --limit is split evenly across them. A shard that comes
up short (duplicates, unreadable images, or simply fewer rows) passes its
leftover on to the shards after it, so the catalogue still reaches --limit
whenever the shards together hold enough.

Usage:
    python scripts/3_build_catalogue.py --limit 4000
    python scripts/3_build_catalogue.py --limit 4000 --format csv
    python scripts/3_build_catalogue.py --parquet data/raw/myntra-train-*.parquet --limit 20000
"""

import argparse
import csv
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

DEFAULT_PARQUET = Path("data/raw/myntra-train-00001-of-00008.parquet")

# Columns kept from the shard. The VQA lists are deliberately absent.
COLUMNS = ["image", "caption", "objects"]

# One row per image, so the per-garment annotation is taken from the first
# object — in this dataset that is the main item, with any second entry being
# a secondary garment or accessory. Their names are kept in `all_items` so the
# extra detections are not silently lost.
FIELDS = [
    "id",
    "file",
    "name",
    "colors",
    "sex",
    "styles",
    "materials",
    "length",
    "fit",
    "caption",
    "all_items",
    "source_path",
    "source_shard",
    "source_row",
]

LIST_FIELDS = {"colors", "styles", "materials", "all_items"}


def guess_extension(raw: bytes) -> str:
    """Read the image header to pick the right extension. No decode, no re-encode."""
    fmt = Image.open(io.BytesIO(raw)).format  # lazy: parses the header only
    return {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(fmt, ".bin")


def build_row(record: dict, item_id: str, file_path: Path, shard: str, row_idx: int) -> dict:
    objects = record.get("objects") or []
    primary = objects[0] if objects else {}
    return {
        "id": item_id,
        "file": file_path.as_posix(),
        "name": (primary.get("name") or "").strip(),
        "colors": primary.get("colors") or [],
        "sex": primary.get("sex") or "",
        "styles": primary.get("styles") or [],
        "materials": primary.get("materials") or [],
        "length": primary.get("length") or "",
        "fit": primary.get("fit") or "",
        "caption": (record.get("caption") or "").strip(),
        "all_items": [o.get("name", "") for o in objects],
        "source_path": (record.get("image") or {}).get("path") or "",
        "source_shard": shard,
        "source_row": row_idx,
    }


def write_metadata(rows: list[dict], out_dir: Path, fmt: str) -> Path:
    if fmt == "csv":
        path = out_dir / "metadata.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            for row in rows:
                # CSV has no list type; join with | so the values stay splittable.
                writer.writerow(
                    {k: "|".join(v) if k in LIST_FIELDS else v for k, v in row.items()}
                )
        return path

    path = out_dir / "metadata.parquet"
    table = pa.Table.from_pylist(rows)  # lists survive as real list columns
    pq.write_table(table, path)
    return path


def build(shards: list[Path], out_dir: Path, limit: int, fmt: str, batch_size: int):
    missing = [s for s in shards if not s.exists()]
    if missing:
        raise SystemExit(
            f"not found: {', '.join(map(str, missing))} — download a shard first:\n"
            f"    python scripts/1_download_shard.py"
        )

    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"source : {len(shards)} shard(s)")
    print(f"target : {out_dir}")
    print(f"limit  : {limit:,}\n")

    rows: list[dict] = []
    skipped = 0
    duplicates = 0
    total_bytes = 0
    per_shard: list[dict] = []
    # The same product can appear in more than one shard; indexing it twice
    # would return it twice in search results.
    seen: set[str] = set()

    for i, shard in enumerate(shards):
        # Recomputed per shard so any shortfall so far is spread over the
        # shards that remain rather than lost.
        quota = math.ceil((limit - len(rows)) / (len(shards) - i))
        pf = pq.ParquetFile(shard)
        print(f"{shard.name} ({pf.metadata.num_rows:,} rows) — quota {quota:,}")

        taken = 0
        row_idx = -1
        # Each shard is a single row group with image bytes inline, so it is
        # read in batches — reading the whole table at once would not fit in memory.
        for batch in pf.iter_batches(batch_size=batch_size, columns=COLUMNS):
            for record in batch.to_pylist():
                row_idx += 1
                if taken >= quota:
                    break

                image = record.get("image") or {}
                raw = image.get("bytes")
                if not raw:
                    skipped += 1
                    continue

                source_path = image.get("path") or ""
                if source_path and source_path in seen:
                    duplicates += 1
                    continue

                # Sequential IDs across all shards in sorted shard order, which
                # is fixed, so a rerun with the same inputs reproduces the same
                # catalogue.
                item_id = f"{len(rows):06d}"
                try:
                    file_path = image_dir / f"{item_id}{guess_extension(raw)}"
                except Exception:  # unreadable header — not worth indexing
                    skipped += 1
                    continue

                file_path.write_bytes(raw)  # original bytes, untouched
                total_bytes += len(raw)
                rows.append(build_row(record, item_id, file_path, shard.name, row_idx))
                if source_path:
                    seen.add(source_path)
                taken += 1

            if taken >= quota:
                break
            print(f"  {len(rows):,} images extracted...", end="\r")

        per_shard.append({"name": shard.name, "rows": pf.metadata.num_rows, "taken": taken})
        print(f"  took {taken:,} → {len(rows):,} total          ")

    meta_path = write_metadata(rows, out_dir, fmt)

    # Provenance for the index manifest: which shards these items came from
    # and how they were chosen. There is no random seed to record because there
    # is no sampling — each shard contributes its first rows in shard order.
    # The shards are already shuffled (each one's category mix matches the
    # full shard), so a prefix behaves like a sample while staying exactly
    # reproducible.
    (out_dir / "catalogue_manifest.json").write_text(
        json.dumps(
            {
                "source_shards": per_shard,
                "items": len(rows),
                "skipped": skipped,
                "duplicates": duplicates,
                "selection": (
                    "limit split evenly across shards in sorted order, first rows "
                    "of each shard, shortfall carried to later shards "
                    "(no sampling, no seed)"
                ),
                "images_bytes": total_bytes,
                "metadata_file": meta_path.name,
                "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nimages     : {len(rows):,} in {image_dir}")
    print(f"skipped    : {skipped}")
    print(f"duplicates : {duplicates}")
    print(f"size       : {total_bytes / 1e6:.0f} MB")
    print(f"metadata   : {meta_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, nargs="+", default=[DEFAULT_PARQUET])
    parser.add_argument("--out", type=Path, default=Path("data/catalogue"))
    parser.add_argument("--limit", type=int, default=4000, help="how many images to extract")
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    # Sorted and de-duplicated so ID assignment does not depend on the order
    # the shell expanded a glob in.
    shards = sorted(set(args.parquet))
    build(shards, args.out, args.limit, args.format, args.batch_size)


if __name__ == "__main__":
    main()
