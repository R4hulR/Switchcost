"""Focused recheck of W2T4 vs. W4T2 (the primary p99 crossover found in the
round-1 sweep, see results/REVIEW.md) after the round-2 harness fixes.

Per the 2026-09-17 review: at least 3 FRESH arrival seeds (not reused from
the round-1 sweep's seeds 5001-5003, which are now treated as exploratory/
already-inspected), matched across configs (same seed -> same schedule
replayed against both configs), randomized execution order, >=5000
requests per steady condition (duration chosen per rate to guarantee this
by construction, not just probabilistically).

These new seeds (20001-20003) are themselves now "inspected" once this
recheck's results are analyzed -- a further, still-unused seed set must be
reserved for eventual policy evaluation (docs/pilot_config_matrix.md §5a),
not reused here.

Usage: python scripts/run_recheck.py
"""

from __future__ import annotations

import csv
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "results" / "recheck"

CONFIGS = ["W2T4", "W4T2"]
RATES = {"low": 40, "medium": 90, "stress": 130}
SEEDS = [20001, 20002, 20003]  # fresh, not reused from the round-1 sweep (5001-5003)
MIN_REQUESTS = 5000
RUN_ORDER_SEED = 42424242


def duration_for(rate_hz: float) -> float:
    # Duration-bounded schedule generation (docs/pilot_config_matrix.md §5)
    # means request count is realized, not exact. A 5% padding above the
    # strict minimum makes MIN_REQUESTS likely in expectation -- it does
    # NOT guarantee it by construction (Poisson arrival count has real
    # variance; a short enough or unlucky draw could still land under the
    # padded target). Corrected 2026-09-17 (review): all 18 recheck runs
    # in fact exceeded 5,000 (5,128-5,455), which is a fact about that
    # specific run, not a property this function guarantees -- run_one()
    # below now validates the actual count post-hoc rather than assuming it.
    return math.ceil(MIN_REQUESTS / rate_hz * 1.05)


def run_one(config: str, rate_label: str, rate_hz: float, seed: int) -> dict:
    duration = duration_for(rate_hz)
    run_id = f"recheck_{config}_{rate_label}_seed{seed}"
    out_dir = f"results/recheck/{run_id}"
    cmd = [
        sys.executable,
        "scripts/smoke_bench.py",
        "--config", config,
        "--rate", str(rate_hz),
        "--duration", str(duration),
        "--arrival", "poisson",
        "--seed", str(seed),
        "--split", "eval",
        "--queue-capacity", "5000",
        "--max-queue-wait", "5.0",
        "--drain-timeout", "15.0",
        "--out", out_dir,
        "--run-id", run_id,
    ]
    print(f"RUN {run_id} (duration={duration:.0f}s): {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        raise RuntimeError(f"recheck run {run_id} failed (exit {proc.returncode})")

    summary = json.loads((REPO_ROOT / out_dir / "summary.json").read_text())
    if summary["n_requests"] < MIN_REQUESTS:
        print(
            f"WARNING: {run_id} realized only {summary['n_requests']} requests, "
            f"under the MIN_REQUESTS={MIN_REQUESTS} target (padding is not a guarantee -- "
            f"see duration_for() docstring). Treat this condition's percentiles with that in mind."
        )

    row = {
        "run_id": run_id,
        "config": config,
        "rate_label": rate_label,
        "rate_hz_nominal": rate_hz,
        "duration_s_nominal": duration,
        "seed": seed,
        "n_requests": summary["n_requests"],
        "run_valid": summary["run_valid"],
        "orderly_shutdown": summary["orderly_shutdown"],
        "collection_confirmed": summary["collection_confirmed"],
        "throughput_within_nominal_window_per_s": summary["throughput"]["completed_within_nominal_window"]["rate_per_s"],
        "e2e_p50_ms": summary["latency_ms"]["e2e"]["p50"],
        "e2e_p95_ms": summary["latency_ms"]["e2e"]["p95"],
        "e2e_p99_ms": summary["latency_ms"]["e2e"]["p99"],
        "wall_elapsed_s": elapsed,
    }
    for outcome in ("COMPLETED", "REJECTED", "TIMED_OUT", "UNFINISHED", "INTERRUPTED",
                     "COMPLETED_UNCONFIRMED", "TIMED_OUT_UNCONFIRMED", "UNKNOWN"):
        row[f"n_{outcome}"] = summary["outcome_counts"].get(outcome, 0)
    return row


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plan = []
    for config in CONFIGS:
        for rate_label, rate_hz in RATES.items():
            for seed in SEEDS:
                plan.append((config, rate_label, rate_hz, seed))

    rng = random.Random(RUN_ORDER_SEED)
    rng.shuffle(plan)

    print(f"Recheck plan: {len(plan)} runs, order randomized with seed={RUN_ORDER_SEED}")
    rows = []
    for i, (config, rate_label, rate_hz, seed) in enumerate(plan, 1):
        print(f"[{i}/{len(plan)}]", end=" ")
        row = run_one(config, rate_label, rate_hz, seed)
        rows.append(row)
        print(
            f"    -> n={row['n_requests']} completed={row['n_COMPLETED']} "
            f"e2e_p50={row['e2e_p50_ms']:.1f}ms e2e_p99={row['e2e_p99_ms']:.1f}ms "
            f"run_valid={row['run_valid']}"
        )

    csv_path = OUT_DIR / "recheck_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
