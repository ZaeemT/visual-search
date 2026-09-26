"""Download a single parquet shard of one RuFashion-2M subset.

The dataset is split into ~1 GB shards. One shard is plenty for this project
(the myntra shards hold ~60k rows each), so we pull exactly one file instead of
the whole subset, and instead of streaming row-by-row over HTTP.

Row counts are read from the parquet footer, which costs a few KB — so
--check reports sizes and row counts without downloading anything.

Usage:
    python scripts/1_download_shard.py --check
    python scripts/1_download_shard.py
    python scripts/1_download_shard.py --config fashion550k --shard 2
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import hf_cache  # noqa: F401,E402  sets HF_HOME and the download timeout

import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download  # noqa: E402

DATASET_ID = "epishchik/RuFashion-2M"


def list_shards(config: str, split: str = "train") -> list[tuple[str, int]]:
    """Return [(path, size_bytes)] for one subset's shards, in order."""
    info = HfApi().repo_info(DATASET_ID, repo_type="dataset", files_metadata=True)
    shards = [
        (s.rfilename, s.size)
        for s in info.siblings
        if s.rfilename.startswith(f"data/{config}/{split}-")
        and s.rfilename.endswith(".parquet")
    ]
    if not shards:
        raise SystemExit(f"no {split} shards found for config {config!r}")
    return sorted(shards)


def row_count_remote(path: str) -> int:
    """Read the row count from the remote parquet footer (a few KB, no download)."""
    with HfFileSystem().open(f"datasets/{DATASET_ID}/{path}") as f:
        return pq.ParquetFile(f).metadata.num_rows


def check(config: str, split: str):
    shards = list_shards(config, split)
    print(f"{config}/{split}: {len(shards)} shards\n")
    total = 0
    for i, (path, size) in enumerate(shards):
        rows = row_count_remote(path)
        total += rows
        print(f"[{i}] {Path(path).name:30s} {size / 1e9:5.2f} GB  {rows:>8,} rows")
    print(f"\ntotal: {total:,} rows across {len(shards)} shards")


def download(config: str, split: str, index: int, out_dir: Path):
    shards = list_shards(config, split)
    if not 0 <= index < len(shards):
        raise SystemExit(f"shard index {index} out of range (0..{len(shards) - 1})")

    path, size = shards[index]
    target = out_dir / f"{config}-{Path(path).name}"

    print(f"shard    : {path}")
    print(f"size     : {size / 1e9:.2f} GB")
    print(f"target   : {target}")

    if target.exists():
        md = pq.ParquetFile(target).metadata
        print(f"\nalready present ({md.num_rows:,} rows) — nothing to do")
        return

    print(f"rows     : {row_count_remote(path):,}")
    print("\ndownloading (resumable)...\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    # local_dir keeps the file inside the project rather than in ~/.cache/huggingface,
    # mirroring the repo's directory layout; flatten it to a single readable name.
    downloaded = Path(hf_hub_download(DATASET_ID, path, repo_type="dataset", local_dir=out_dir))
    downloaded.replace(target)

    md = pq.ParquetFile(target).metadata
    print(f"\nsaved to : {target}")
    print(f"on disk  : {target.stat().st_size / 1e9:.2f} GB")
    print(f"rows     : {md.num_rows:,}")
    print(f"columns  : {md.num_columns}")
    print(f"row groups: {md.num_row_groups}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="myntra")
    parser.add_argument("--split", default="train")
    parser.add_argument("--shard", type=int, default=0, help="shard index, 0-based")
    parser.add_argument("--out", type=Path, default=Path("data/raw"), help="where to put the shard")
    parser.add_argument("--check", action="store_true", help="report sizes and row counts, download nothing")
    args = parser.parse_args()

    if args.check:
        check(args.config, args.split)
    else:
        download(args.config, args.split, args.shard, args.out)


if __name__ == "__main__":
    main()
