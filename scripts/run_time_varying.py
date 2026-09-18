"""Time-varying FIXED-CONFIG experiments: sustained low-high-low load change
and short bursts, replayed as saved traces (scripts/common/trace.py) against
W2T4 and W4T2 only -- the pair with the confirmed p99 crossover
(results/REVIEW.md). No switching/controller: each run uses one fixed
config for its entire (multi-phase) duration, per the 2026-09-17 review.

Usage: python scripts/run_time_varying.py
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.trace import build_phased_schedule, save_trace  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACE_DIR = REPO_ROOT / "models" / "traces"
OUT_DIR = REPO_ROOT / "results" / "time_varying"

CONFIGS = ["W2T4", "W4T2"]
BASELINE_RATE = 40.0
BURST_RATE = 130.0

# Fresh seeds -- not reused from the sweep (5001-5003) or the recheck
# (20001-20003), both now treated as exploratory/already-inspected
# (docs/pilot_config_matrix.md §5a).
SUSTAINED_SEEDS = [40001, 40002]
BURST_DURATIONS_S = [0.1, 0.5, 2.0, 10.0]
BURST_TRACE_SEED = 41001  # one fresh seed for the whole burst-trace family


def build_sustained_trace_spec() -> list[dict]:
    return [
        {"phase_id": "low1", "rate_hz": BASELINE_RATE, "duration_s": 50.0},
        {"phase_id": "high", "rate_hz": BURST_RATE, "duration_s": 40.0},
        {"phase_id": "low2", "rate_hz": BASELINE_RATE, "duration_s": 50.0},
    ]


def build_burst_trace_spec(burst_duration_s: float) -> list[dict]:
    # Repeated bursts so pooled burst-affected requests give a percentile
    # with a meaningful sample count (2026-09-17 review: "use repeated-
    # burst distributions ... rather than presenting a p99 from a handful
    # of requests") -- a single burst_duration_s=0.1s burst only contains
    # ~13 requests at BURST_RATE, nowhere near enough alone.
    n_bursts = 10 if burst_duration_s <= 2.0 else 5  # longer bursts already carry more samples per instance
    gap_s = max(5.0, 2.0 * burst_duration_s)  # let the queue fully drain back to baseline between bursts
    phases = []
    for i in range(n_bursts):
        phases.append({"phase_id": f"baseline_{i}", "rate_hz": BASELINE_RATE, "duration_s": gap_s})
        phases.append({"phase_id": f"burst_{i}", "rate_hz": BURST_RATE, "duration_s": burst_duration_s})
    phases.append({"phase_id": f"baseline_{n_bursts}", "rate_hz": BASELINE_RATE, "duration_s": gap_s})
    return phases


def run_one(config: str, trace_path: Path, run_id: str, out_dir: Path) -> dict:
    cmd = [
        sys.executable,
        "scripts/smoke_bench.py",
        "--config", config,
        "--trace-file", str(trace_path),
        "--split", "eval",
        "--queue-capacity", "3000",
        "--max-queue-wait", "5.0",
        "--drain-timeout", "15.0",
        "--out", str(out_dir.relative_to(REPO_ROOT)),
        "--run-id", run_id,
    ]
    print(f"RUN {run_id}: {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        raise RuntimeError(f"run {run_id} failed (exit {proc.returncode})")
    summary = json.loads((out_dir / "summary.json").read_text())
    return {"run_id": run_id, "config": config, "elapsed_s": elapsed, "summary": summary}


def main() -> None:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index_rows = []

    # --- Sustained low-high-low ---
    sustained_phases = build_sustained_trace_spec()
    sustained_traces = {}
    for seed in SUSTAINED_SEEDS:
        trace = build_phased_schedule(sustained_phases, seed=seed, arrival="poisson")
        path = TRACE_DIR / f"sustained_seed{seed}.json"
        save_trace(trace, path)
        sustained_traces[seed] = path
        print(f"Built sustained trace seed={seed} -> {path} ({len(trace['requests'])} requests)")

    plan = []
    for seed, path in sustained_traces.items():
        for config in CONFIGS:
            run_id = f"sustained_{config}_seed{seed}"
            plan.append(("sustained", config, path, run_id))

    # --- Bursts ---
    burst_traces = {}
    for d in BURST_DURATIONS_S:
        phases = build_burst_trace_spec(d)
        trace = build_phased_schedule(phases, seed=BURST_TRACE_SEED, arrival="poisson")
        path = TRACE_DIR / f"burst_{d}s.json"
        save_trace(trace, path)
        burst_traces[d] = path
        n_burst_reqs = sum(p["n_requests"] for p in trace["phases"] if p["phase_id"].startswith("burst_"))
        print(f"Built burst trace d={d}s -> {path} ({len(trace['requests'])} requests, {n_burst_reqs} in bursts)")

    for d, path in burst_traces.items():
        for config in CONFIGS:
            run_id = f"burst_{d}s_{config}"
            plan.append(("burst", config, path, run_id))

    # Randomized execution order (matches the sweep/recheck convention).
    import random

    rng = random.Random(43434343)
    rng.shuffle(plan)

    print(f"\nTime-varying plan: {len(plan)} runs, order randomized")
    for i, (kind, config, path, run_id) in enumerate(plan, 1):
        out_dir = OUT_DIR / run_id
        print(f"[{i}/{len(plan)}]", end=" ")
        result = run_one(config, path, run_id, out_dir)
        summary = result["summary"]
        e2e = summary["latency_ms"]["e2e"]
        print(
            f"    -> n={summary['n_requests']} completed={summary['outcome_counts'].get('COMPLETED', 0)} "
            f"e2e_p50={e2e['p50']:.1f}ms e2e_p99={e2e['p99']:.1f}ms run_valid={summary['run_valid']}"
        )
        index_rows.append(
            {
                "kind": kind,
                "run_id": run_id,
                "config": config,
                "trace_file": str(path.relative_to(REPO_ROOT)),
                "out_dir": str(out_dir.relative_to(REPO_ROOT)),
                "n_requests": summary["n_requests"],
                "n_completed": summary["outcome_counts"].get("COMPLETED", 0),
                "run_valid": summary["run_valid"],
                "e2e_p50_ms": e2e["p50"],
                "e2e_p95_ms": e2e["p95"],
                "e2e_p99_ms": e2e["p99"],
                "elapsed_s": result["elapsed_s"],
            }
        )

    index_path = OUT_DIR / "time_varying_index.csv"
    with open(index_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
        writer.writeheader()
        writer.writerows(index_rows)
    print(f"\nWrote {index_path} ({len(index_rows)} rows)")


if __name__ == "__main__":
    main()
