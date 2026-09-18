"""Tuning-phase calibration sweep (docs/pilot_config_matrix.md §5, steps 1-2):
short probes across all 4 configs at several candidate rates, against the
TUNING corpus split, to find each config's approximate saturation point so
a common absolute rate set can be frozen for the matched sweep. Numbers
from this phase are for calibration only -- not reported as pilot results.

Usage: python scripts/run_calibration.py
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "results" / "calibration"
RATES = [30, 60, 100, 150, 220]
CONFIGS = ["W1T8", "W2T4", "W4T2", "W8T1"]
PROBE_DURATION_S = 6.0


def run_one(config: str, rate: float, run_idx: int) -> dict:
    run_id = f"calib_{config}_r{rate:g}"
    out_dir = f"results/calibration/{run_id}"
    cmd = [
        sys.executable,
        "scripts/smoke_bench.py",
        "--config", config,
        "--rate", str(rate),
        "--duration", str(PROBE_DURATION_S),
        "--arrival", "poisson",
        "--seed", "100",
        "--split", "tuning",
        "--queue-capacity", "1000",
        "--max-queue-wait", "5.0",
        "--drain-timeout", "8.0",
        "--out", out_dir,
        "--run-id", run_id,
    ]
    print(f"[{run_idx}] {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        raise RuntimeError(
            f"calibration run {run_id} failed (exit {proc.returncode}) -- exit code 2 means "
            f"the harness marked the run invalid (see its summary.json's run_invalid_reasons), "
            f"not a crash"
        )

    summary = json.loads((REPO_ROOT / out_dir / "summary.json").read_text())
    return {
        "run_id": run_id,
        "config": config,
        "rate_hz_nominal": rate,
        "achieved_arrival_rate_hz": summary["achieved_arrival_rate_hz_over_nominal_duration"],
        "last_arrival_offset_s": summary["last_arrival_offset_s"],
        "throughput_within_nominal_window_per_s": summary["throughput"]["completed_within_nominal_window"]["rate_per_s"],
        "throughput_including_drain_per_s": summary["throughput"]["completed_including_drain"]["rate_per_s"],
        "run_valid": summary["run_valid"],
        "n_requests": summary["n_requests"],
        "outcome_counts": summary["outcome_counts"],
        "e2e_p50_ms": summary["latency_ms"]["e2e"]["p50"],
        "e2e_p95_ms": summary["latency_ms"]["e2e"]["p95"],
        "e2e_p99_ms": summary["latency_ms"]["e2e"]["p99"],
        "queueing_delay_p50_ms": summary["latency_ms"]["queueing_delay"]["p50"],
        "wall_elapsed_s": elapsed,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    run_idx = 0
    for config in CONFIGS:
        for rate in RATES:
            run_idx += 1
            row = run_one(config, rate, run_idx)
            rows.append(row)
            n_completed = row["outcome_counts"].get("COMPLETED", 0)
            n_total = sum(row["outcome_counts"].values())
            print(
                f"    -> completed={n_completed}/{n_total} "
                f"throughput_within_window={row['throughput_within_nominal_window_per_s']:.1f}/s "
                f"e2e_p50={row['e2e_p50_ms']:.1f}ms e2e_p99={row['e2e_p99_ms']:.1f}ms "
                f"queue_p50={row['queueing_delay_p50_ms']:.1f}ms run_valid={row['run_valid']}"
            )

    csv_path = OUT_DIR / "calibration_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            r = dict(r)
            r["outcome_counts"] = json.dumps(r["outcome_counts"])
            writer.writerow(r)
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
