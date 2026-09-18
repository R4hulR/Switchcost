"""Assembles review_bundle_<phase>_<date>.zip for external review (e.g.
uploading to another model). Includes: REVIEW.md, a combined per-run
summary.csv, exact configs/commands/dependency versions/code commit,
relevant scripts, correctness-check output, plots, and raw per-request
logs -- excludes venv, model weights, caches.

Usage: python scripts/make_review_bundle.py --phase calibration-and-sweep
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
BUNDLE_ROOT = RESULTS_DIR / "bundles"

SCRIPTS_TO_INCLUDE = [
    "scripts/smoke_bench.py",
    "scripts/worker.py",
    "scripts/export_model.py",
    "scripts/verify_correctness.py",
    "scripts/measure_ipc_floor.py",
    "scripts/build_corpus.py",
    "scripts/pretokenize.py",
    "scripts/run_calibration.py",
    "scripts/run_sweep.py",
    "scripts/run_recheck.py",
    "scripts/run_time_varying.py",
    "scripts/analyze_time_varying.py",
    "scripts/plot_results.py",
    "scripts/make_review_bundle.py",
    "scripts/common/__init__.py",
    "scripts/common/affinity.py",
    "scripts/common/env_setup.py",
    "scripts/common/pooling.py",
    "scripts/common/protocol.py",
    "scripts/common/stats.py",
    "scripts/common/trace.py",
]
DOCS_TO_INCLUDE = [
    "docs/environment_report.md",
    "docs/pilot_config_matrix.md",
    "docs/research_log.md",
]
RESULT_SUBDIRS_TO_INCLUDE = [
    "verification", "verification2", "calibration", "sweep", "recheck", "time_varying", "plots",
]
EXTRA_FILES_TO_INCLUDE = list((REPO_ROOT / "models" / "traces").glob("*.json")) if (REPO_ROOT / "models" / "traces").exists() else []


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def git_diffstat() -> str:
    try:
        return subprocess.check_output(
            ["git", "status", "--short"], cwd=REPO_ROOT, text=True
        ).strip() or "(clean -- no uncommitted changes)"
    except Exception:
        return "(unknown)"


def build_master_summary_csv(out_path: Path) -> None:
    """One row per run across all phases, tagged with a phase column.
    Different phases have different columns (e.g. recheck adds n_requests/
    duration fields calibration and sweep don't have) -- fieldnames is the
    UNION across all phases (not just the first phase's columns), so later
    phases with extra columns don't get silently truncated/rejected."""
    rows = []
    for phase, csv_name, subdir in [
        ("calibration", "calibration_summary.csv", "calibration"),
        ("sweep", "sweep_summary.csv", "sweep"),
        ("recheck", "recheck_summary.csv", "recheck"),
        ("time_varying", "time_varying_index.csv", "time_varying"),
    ]:
        p = RESULTS_DIR / subdir / csv_name
        if not p.exists():
            continue
        with open(p) as f:
            for row in csv.DictReader(f):
                row["phase"] = phase
                rows.append(row)

    if not rows:
        return
    all_fields = ["phase"]
    for row in rows:
        for k in row.keys():
            if k not in all_fields:
                all_fields.append(k)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in all_fields})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, help="descriptive phase label, e.g. calibration-and-sweep")
    args = ap.parse_args()

    BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    bundle_name = f"review_bundle_{args.phase}_{today}"
    stage_dir = BUNDLE_ROOT / f"_staging_{bundle_name}"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    # REVIEW.md
    review_src = RESULTS_DIR / "REVIEW.md"
    if review_src.exists():
        shutil.copy(review_src, stage_dir / "REVIEW.md")

    # Combined summary.csv
    build_master_summary_csv(stage_dir / "summary.csv")

    # Configs / commit / dependency versions
    meta_dir = stage_dir / "run_metadata"
    meta_dir.mkdir()
    (meta_dir / "commit.txt").write_text(git_commit() + "\n")
    (meta_dir / "git_status_short.txt").write_text(git_diffstat() + "\n")
    shutil.copy(REPO_ROOT / "requirements.txt", meta_dir / "requirements.txt")
    for calib_file in ["calibration_decision.json"]:
        p = RESULTS_DIR / "calibration" / calib_file
        if p.exists():
            shutil.copy(p, meta_dir / calib_file)

    # Code
    code_dir = stage_dir / "scripts"
    for rel in SCRIPTS_TO_INCLUDE:
        src = REPO_ROOT / rel
        if not src.exists():
            continue
        dst = code_dir / Path(rel).relative_to("scripts")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)

    # Docs (design + research log, for context on the raw logs/configs)
    docs_dir = stage_dir / "docs"
    docs_dir.mkdir()
    for rel in DOCS_TO_INCLUDE:
        src = REPO_ROOT / rel
        if src.exists():
            shutil.copy(src, docs_dir / Path(rel).name)

    # Correctness-check output
    cc = RESULTS_DIR / "correctness_check.json"
    if cc.exists():
        shutil.copy(cc, stage_dir / "correctness_check.json")

    # Results: verification / calibration / sweep / time_varying raw logs + summaries, plots
    results_dst = stage_dir / "results"
    for sub in RESULT_SUBDIRS_TO_INCLUDE:
        src = RESULTS_DIR / sub
        if src.exists():
            shutil.copytree(src, results_dst / sub, ignore=shutil.ignore_patterns("_staging_*"))

    # Saved trace files (exact input to the time-varying runs)
    if EXTRA_FILES_TO_INCLUDE:
        traces_dst = stage_dir / "models" / "traces"
        traces_dst.mkdir(parents=True, exist_ok=True)
        for f in EXTRA_FILES_TO_INCLUDE:
            shutil.copy(f, traces_dst / f.name)

    # Zip it up
    zip_path = BUNDLE_ROOT / bundle_name  # shutil adds .zip
    archive_path = shutil.make_archive(str(zip_path), "zip", root_dir=stage_dir)
    shutil.rmtree(stage_dir)

    size_mb = Path(archive_path).stat().st_size / (1024 * 1024)
    print(f"Wrote {archive_path} ({size_mb:.2f} MB)")
    if size_mb > 20:
        print("WARNING: bundle exceeds 20MB -- consider splitting raw logs into numbered archives.")


if __name__ == "__main__":
    main()
