"""Measure endpoint latency stage by stage, over repeated queries.

Drives the running HTTP server exactly as a client would, so the numbers are
what a caller actually experiences — including request overhead, JSON encoding
and file upload, none of which appear in an in-process measurement.

Three passes over the query set:

  cold    explanations on, translation cache empty  -> what an LLM call costs
  warm    explanations on, translation cache full   -> template cost alone
  none    explanations off                          -> the retrieval path alone

The "template" and "LLM" rows come from the cache being warm or cold rather
than from separate timers: assembling the sentence is a few string joins, so
whatever a warm request spends in `explain` is effectively all template, and
the difference against a cold one is the Ollama call.

Medians and p95 are reported rather than means, because a single slow request
(a GPU scheduling hiccup, a cache miss) drags a mean somewhere no real request
ever was.

Usage:
    python scripts/7_benchmark_latency.py
    python scripts/7_benchmark_latency.py --queries 20 --repeats 5 --top-k 10
    python scripts/7_benchmark_latency.py --json data/bench/latency.json
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

QUERY_DIR = Path("data/queries/images")
DEFAULT_URL = "http://127.0.0.1:8000"


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile — no interpolation, so every value reported was measured."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * len(ordered)) - 1))
    return ordered[index]


def summarise(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return statistics.median(values), percentile(values, 0.95)


def run_pass(
    client: httpx.Client, url: str, images: list[Path], repeats: int, top_k: int, explain: bool
) -> dict[str, list[float]]:
    """One full sweep of the query set, collecting server timings and wall time."""
    collected: dict[str, list[float]] = {
        "preprocess": [], "embed": [], "search": [], "explain": [], "total": [], "e2e": []
    }

    for repeat in range(repeats):
        for image in images:
            data = image.read_bytes()
            started = time.perf_counter()
            response = client.post(
                f"{url}/search/",
                files={"image": (image.name, data, "image/jpeg")},
                data={"top_k": str(top_k), "explain": str(explain).lower()},
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            response.raise_for_status()

            timings = response.json()["timings_ms"]
            for stage in ("preprocess", "embed", "search", "explain", "total"):
                collected[stage].append(timings[stage])
            collected["e2e"].append(elapsed_ms)

        print(f"    repeat {repeat + 1}/{repeats} done", end="\r", flush=True)
    print(" " * 40, end="\r")
    return collected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--queries", type=int, default=20, help="distinct query images")
    parser.add_argument("--repeats", type=int, default=5, help="passes over the query set")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--query-dir", type=Path, default=QUERY_DIR)
    parser.add_argument("--json", type=Path, default=None, help="also write raw samples here")
    args = parser.parse_args()

    images = sorted(args.query_dir.glob("*.jpg"))[: args.queries]
    if not images:
        raise SystemExit(
            f"no query images in {args.query_dir} — build a query set first:\n"
            f"    python scripts/3_build_catalogue.py "
            f"--parquet data/raw/myntra-test-00001-of-00001.parquet --out data/queries --limit 20"
        )

    with httpx.Client(timeout=120.0) as client:
        try:
            info = client.get(f"{args.url}/search/info").json()
        except Exception as exc:
            raise SystemExit(f"cannot reach {args.url} — is the server running? ({exc})")

        print(f"server    : {args.url}")
        print(f"model     : {info['model']} on {info['device']}")
        print(f"catalogue : {info['catalogue_size']:,} items, dim {info['dim']}")
        print(f"workload  : {len(images)} queries x {args.repeats} repeats, top_k={args.top_k}\n")

        # One sweep before measuring, so no measured request pays for a cold
        # GPU pipeline — the concern is steady-state latency, not first-run cost.
        print("  warming up...")
        run_pass(client, args.url, images[:3], 1, args.top_k, explain=True)

        # Cold explanations: the server has been up a while, but most category
        # names in this set have now been translated, so this pass mostly
        # measures the steady state with occasional misses.
        print("  pass 1/3: explanations on (first sight of these categories)")
        cold = run_pass(client, args.url, images, 1, args.top_k, explain=True)

        print("  pass 2/3: explanations on (translations cached)")
        warm = run_pass(client, args.url, images, args.repeats, args.top_k, explain=True)

        print("  pass 3/3: explanations off")
        plain = run_pass(client, args.url, images, args.repeats, args.top_k, explain=False)

    rows = [
        ("Preprocess", plain["preprocess"]),
        ("Embed query", plain["embed"]),
        ("Search (score + top-K)", plain["search"]),
        ("Explanations: template", warm["explain"]),
        ("Explanations: LLM", cold["explain"]),
        ("Total server-side", warm["total"]),
        ("End-to-end (as seen by client)", warm["e2e"]),
    ]

    print(f"\n{'Stage':<34}{'Median (ms)':>14}{'p95 (ms)':>12}")
    print("-" * 60)
    for label, values in rows:
        median, p95 = summarise(values)
        print(f"{label:<34}{median:>14.2f}{p95:>12.2f}")

    samples = sum(len(v) for v in (plain["total"], warm["total"], cold["total"]))
    print(f"\n{samples} requests. Retrieval rows (preprocess/embed/search) are from the")
    print("explanations-off pass; total and end-to-end are from the cached-explanation pass.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "server": info,
                    "workload": {
                        "queries": len(images),
                        "repeats": args.repeats,
                        "top_k": args.top_k,
                    },
                    "summary": {
                        label: dict(zip(("median_ms", "p95_ms"), summarise(values)))
                        for label, values in rows
                    },
                    "raw": {"cold": cold, "warm": warm, "no_explain": plain},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"raw samples: {args.json}")


if __name__ == "__main__":
    main()
