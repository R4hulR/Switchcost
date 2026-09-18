# Results directory

Each subdirectory holds output from one experiment phase; each individual run gets its own folder with a `summary.json` (all computed statistics for that run) and, for a representative subset of runs, a `raw_requests.csv` (one row per request, full timestamps and outcome).

## What's here vs. archived locally

Every run's `summary.json` is included (small, and it's what all the analysis/plotting scripts read). Raw per-request CSVs are included for a **representative sample only** (16 of 92 runs) to keep this repository a reasonable size — one or two runs per phase, generally covering the W2T4/W4T2 comparison at the most informative condition (e.g., the stress-load and burst conditions where the two configs differ most).

**Full raw per-request logs for all 92 runs are preserved locally** on the machine these experiments were run on (not included in this repository, and not hosted anywhere — there is no download link for them). If you need a specific run's full raw log, ask, or regenerate it: every run is reproducible from its recorded config (rate/duration/seed, or trace file) using the scripts in `scripts/` — see the top-level `README.md` for exact commands.

## Directories

- `verification/`, `verification2/` — harness self-checks (confirms the outcome-accounting logic behaves correctly under normal load, overload, and forced termination), from two rounds of fixes.
- `calibration/` — short probes across all four worker/thread configurations at several rates, used to choose the fixed rates used in later phases.
- `sweep/` — all four configurations compared at three fixed rates, three repeats each.
- `recheck/` — a higher-rigor, larger-sample re-test of just the two configurations (W2T4, W4T2) that showed a real difference in `sweep/`.
- `time_varying/` — sustained load-change and repeated-burst traces against W2T4 and W4T2.
- `plots/` — the plots referenced in the top-level README and `docs/`.
- `correctness_check.json` — output of `scripts/verify_correctness.py`.
