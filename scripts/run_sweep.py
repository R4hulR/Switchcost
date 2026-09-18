"""Matched fixed-config sweep (docs/pilot_config_matrix.md §5, steps 3-4):
replays IDENTICAL frozen schedules (same rate -> same seed -> bit-identical
arrival offsets and input sampling) across all 4 configs, against the EVAL
corpus split, with run order randomized and each (config, rate) cell
repeated for run-to-run variability (§12). Still not an adaptive
controller -- each run uses one fixed config for its whole duration.

Usage: python scripts/run_sweep.py
"""

from __future__ import annotations

import csv
import json
import random
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "results" / "sweep"

CONFIGS = ["W1T8", "W2T4", "W4T2", "W8T1"]
# Frozen 2026-09-17 from results/calibration/calibration_decision.json.
RATES = {"low": 40, "medium": 90, "stress": 130}
N_REPEATS = 3
DURATION_S = 10.0
RUN_ORDER_SEED = 777  # shuffles execution order only; schedule seeds below are separate and per-rate


def schedule_seed_for_rate(rate_label: str) -> int:
    # Same seed for a given rate regardless of config or repeat index -- this
    # is what makes the schedule "identical across configs" (§5 step 3).
    # Repeats reuse it too: repeats measure run-to-run *execution* noise
    # under a bit-identical input schedule, not schedule variation.
    return {"low": 5001, "medium": 5002, "stress": 5003}[rate_label]


def run_one(config: str, rate_label: str, rate_hz: float, repeat_idx: int) -> dict:
    run_id = f"sweep_{config}_{rate_label}_r{repeat_idx}"
    out_dir = f"results/sweep/{run_id}"
    seed = schedule_seed_for_rate(rate_label)
    cmd = [
        sys.executable,
        "scripts/smoke_bench.py",
        "--config", config,
        "--rate", str(rate_hz),
        "--duration", str(DURATION_S),
        "--arrival", "poisson",
        "--seed", str(seed),
        "--split", "eval",
        "--queue-capacity", "3000",
        "--max-queue-wait", "5.0",
        "--drain-timeout", "10.0",
        "--out", out_dir,
        "--run-id", run_id,
    ]
    print(f"RUN {run_id}: {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        raise RuntimeError(
            f"sweep run {run_id} failed (exit {proc.returncode}) -- exit code 2 means the "
            f"harness marked the run invalid (see its summary.json's run_invalid_reasons), "
            f"not a crash"
        )

    summary = json.loads((REPO_ROOT / out_dir / "summary.json").read_text())
    row = {
        "run_id": run_id,
        "config": config,
        "rate_label": rate_label,
        "rate_hz_nominal": rate_hz,
        "repeat_idx": repeat_idx,
        "schedule_seed": seed,
        "split": "eval",
        "n_requests": summary["n_requests"],
        "achieved_arrival_rate_hz": summary["achieved_arrival_rate_hz_over_nominal_duration"],
        "last_arrival_offset_s": summary["last_arrival_offset_s"],
        "throughput_within_nominal_window_per_s": summary["throughput"]["completed_within_nominal_window"]["rate_per_s"],
        "throughput_including_drain_per_s": summary["throughput"]["completed_including_drain"]["rate_per_s"],
        "orderly_shutdown": summary["orderly_shutdown"],
        "collection_confirmed": summary["collection_confirmed"],
        "run_valid": summary["run_valid"],
        "e2e_p50_ms": summary["latency_ms"]["e2e"]["p50"],
        "e2e_p95_ms": summary["latency_ms"]["e2e"]["p95"],
        "e2e_p99_ms": summary["latency_ms"]["e2e"]["p99"],
        "queueing_delay_p50_ms": summary["latency_ms"]["queueing_delay"]["p50"],
        "service_time_p50_ms": summary["latency_ms"]["service_time"]["p50"],
        "generator_delay_p50_ms": summary["latency_ms"]["generator_delay"]["p50"],
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
            for repeat_idx in range(N_REPEATS):
                plan.append((config, rate_label, rate_hz, repeat_idx))

    rng = random.Random(RUN_ORDER_SEED)
    rng.shuffle(plan)

    print(f"Sweep plan: {len(plan)} runs, order randomized with seed={RUN_ORDER_SEED}")
    rows = []
    for i, (config, rate_label, rate_hz, repeat_idx) in enumerate(plan, 1):
        print(f"[{i}/{len(plan)}]", end=" ")
        row = run_one(config, rate_label, rate_hz, repeat_idx)
        rows.append(row)
        print(
            f"    -> completed={row['n_COMPLETED']}/{row['n_requests']} "
            f"e2e_p50={row['e2e_p50_ms']:.1f}ms e2e_p99={row['e2e_p99_ms']:.1f}ms "
            f"orderly={row['orderly_shutdown']}"
        )

    csv_path = OUT_DIR / "sweep_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
