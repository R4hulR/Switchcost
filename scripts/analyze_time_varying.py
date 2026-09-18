"""Analysis for the time-varying (sustained low-high-low, burst) experiments:
pooled burst/baseline percentiles (not per-individual-burst-instance, and
never averaged from per-phase percentiles), and missed-deadline fractions
against the frozen deadline set (results/time_varying/deadline_set.json),
computed over ALL offered requests including rejected/timed-out/etc.

Usage: python scripts/analyze_time_varying.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.stats import summarize  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
TV_DIR = REPO_ROOT / "results" / "time_varying"

FAILURE_OUTCOMES = {
    "REJECTED", "TIMED_OUT", "UNFINISHED", "INTERRUPTED", "UNKNOWN",
    "COMPLETED_UNCONFIRMED", "TIMED_OUT_UNCONFIRMED",
}


def load_raw(run_dir: Path) -> list[dict]:
    with open(run_dir / "raw_requests.csv") as f:
        return list(csv.DictReader(f))


def missed_deadline_fraction(rows: list[dict], deadline_ms: float) -> dict:
    n_offered = len(rows)
    n_missed = 0
    for r in rows:
        if r["outcome"] != "COMPLETED":
            n_missed += 1
            continue
        e2e = r.get("e2e_latency_ms")
        if e2e == "" or e2e is None or float(e2e) > deadline_ms:
            n_missed += 1
    return {"n_offered": n_offered, "n_missed": n_missed, "missed_fraction": n_missed / n_offered if n_offered else None}


def analyze_burst_run(run_dir: Path, deadlines: list[float]) -> dict:
    rows = load_raw(run_dir)
    groups = {"burst": [], "baseline": []}
    for r in rows:
        prefix = r["phase_id"].rsplit("_", 1)[0]
        if prefix in groups:
            groups[prefix].append(r)

    pooled = {}
    for group_name, group_rows in groups.items():
        completed = [r for r in group_rows if r["outcome"] == "COMPLETED"]
        e2e_vals = [float(r["e2e_latency_ms"]) for r in completed]
        pooled[group_name] = {
            "n_requests_offered_pooled": len(group_rows),
            "n_completed_pooled": len(completed),
            "e2e_latency_ms_pooled": summarize(e2e_vals),
            "deadline_misses": {str(d): missed_deadline_fraction(group_rows, d) for d in deadlines},
        }
    whole_trace = {
        "n_requests": len(rows),
        "deadline_misses": {str(d): missed_deadline_fraction(rows, d) for d in deadlines},
    }
    return {"pooled_by_group": pooled, "whole_trace": whole_trace}


def analyze_sustained_run(run_dir: Path, deadlines: list[float]) -> dict:
    rows = load_raw(run_dir)
    by_phase = {}
    for phase_id in sorted({r["phase_id"] for r in rows}):
        phase_rows = [r for r in rows if r["phase_id"] == phase_id]
        by_phase[phase_id] = {
            "n_requests": len(phase_rows),
            "deadline_misses": {str(d): missed_deadline_fraction(phase_rows, d) for d in deadlines},
        }
    whole_trace = {
        "n_requests": len(rows),
        "deadline_misses": {str(d): missed_deadline_fraction(rows, d) for d in deadlines},
    }
    return {"by_phase": by_phase, "whole_trace": whole_trace}


def main() -> None:
    deadline_set = json.loads((TV_DIR / "deadline_set.json").read_text())
    deadlines = deadline_set["deadline_ms_set"]

    with open(TV_DIR / "time_varying_index.csv") as f:
        index_rows = list(csv.DictReader(f))

    report = {"deadline_ms_set": deadlines, "sustained": {}, "burst": {}}
    for row in index_rows:
        run_dir = REPO_ROOT / row["out_dir"]
        if row["kind"] == "sustained":
            report["sustained"][row["run_id"]] = {
                "config": row["config"],
                **analyze_sustained_run(run_dir, deadlines),
            }
        else:
            report["burst"][row["run_id"]] = {
                "config": row["config"],
                **analyze_burst_run(run_dir, deadlines),
            }

    out_path = TV_DIR / "analysis_summary.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"Wrote {out_path}")

    # Print a compact human-readable digest.
    print("\n=== Sustained low-high-low: whole-trace missed-deadline fraction ===")
    for run_id, d in sorted(report["sustained"].items()):
        misses = d["whole_trace"]["deadline_misses"]
        print(f"{run_id:30s} " + " ".join(f"d={dl}ms:{misses[str(dl)]['missed_fraction']:.3f}" for dl in deadlines))

    print("\n=== Burst: pooled burst-window e2e p99 (ms) and sample count ===")
    for run_id, d in sorted(report["burst"].items()):
        b = d["pooled_by_group"]["burst"]["e2e_latency_ms_pooled"]
        n = d["pooled_by_group"]["burst"]["n_completed_pooled"]
        print(f"{run_id:30s} n={n:5d} p50={b['p50'] and round(b['p50'],1)} p99={b['p99'] and round(b['p99'],1)}")


if __name__ == "__main__":
    main()
