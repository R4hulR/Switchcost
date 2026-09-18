# SwitchCost — Review Bundle: Correctness, Harness Fixes (3 rounds), Calibration, Sweep, Recheck, Time-Varying Traces

Date: 2026-09-17
Git commit: see `commit.txt` in this bundle
Repo: `switchcost` (project-local venv, no system-wide installs)

## What ran, in order

1. **Correctness check** (`scripts/verify_correctness.py`): ONNX-exported `all-MiniLM-L6-v2` backbone + our own masked-mean-pooling/L2-normalization vs. the `sentence-transformers` reference, introspecting its actual module pipeline rather than assuming it. Extended in round 2 to also check representative benchmark-corpus inputs across every `intra_op_num_threads` setting the pilot configs actually use (1/2/4/8), not just the original hand-built edge cases at threads=1.
2. **Harness fixes, round 1**, from an independent raw-log review before any benchmarking: percentile convention, queue-timeout semantics, a mislabeled "IPC floor" statistic, shutdown/collection accounting under forced termination, several missing measurements, a duration/request-count mismatch in the arrival-schedule generator.
3. **Verification rerun, round 1** (`results/verification/`): re-exercised the outcome taxonomy after the round-1 fixes.
4. **Tuning-phase calibration** (`scripts/run_calibration.py`, `results/calibration/`): 5 rates × 4 configs, 6s probes, tuning corpus split, to find each config's approximate saturation point.
5. **Matched fixed-config sweep** (`scripts/run_sweep.py`, `results/sweep/`): 3 frozen absolute rates × 4 configs × 3 repeats = 36 runs, run order randomized, identical arrival schedule replayed per rate across all 4 configs, eval corpus split.
6. **Harness fixes, round 2**, from a second independent review of round 1's own raw logs: NPZ lazy-loading overhead inside the timed loop, generator-process background threads (collector/monitor/queue-feeder) never pinned or verified, forced-shutdown status writes in the wrong order relative to their data, rate/duration reporting still conflating two different quantities, throughput conflating in-window and drain-time completions. Also a **correction, not a bug**: round 1's summary framed W2T4 as dominating overall based on p50 -- checking p99 specifically shows a real crossover between W2T4 and W4T2 that the p50-centric framing missed. See §2 below for both.
7. **Verification reruns, round 2** (`results/verification2/`): re-exercised normal load, overload, and forced-termination scenarios against the round-2 fixed code; also re-measured the IPC floor with materialized arrays (dropped from ~2.4ms to ~1.0ms p50, plus a new pickle-only decomposition).
8. **Focused recheck** (`scripts/run_recheck.py`, `results/recheck/`): W2T4 vs. W4T2 only (the pair with the real p99 crossover) at the same 3 frozen rates, 3 **fresh** seeds (20001-20003, not reused from the sweep's 5001-5003, which are now treated as exploratory/already-inspected), matched schedules across configs, randomized execution order, duration sized to make ≥5,000 requests per condition likely (18/18 runs achieved it in fact, 5,128-5,455 realized -- padding does not *guarantee* the threshold, corrected in round 3, see §7), run against the round-2 fixed harness.
9. **Independent verification of all 95,226 recheck requests** (counts, percentiles, timestamp ordering, in-window throughput, matched schedules) confirmed to agree -- **go/no-go criterion 1 passes provisionally for this workload and this WSL environment.**
10. **Time-varying fixed-config traces** (`scripts/common/trace.py`, `scripts/run_time_varying.py`, `scripts/analyze_time_varying.py`, `results/time_varying/`): harness extended to replay a saved, multi-phase arrival trace as ONE continuous run (workers/queues alive across phase boundaries, no drain/restart/re-warm at a boundary). Two trace types, W2T4 vs. W4T2 only, matched schedules: sustained low-high-low (40→130→40 req/s, 2 fresh seeds) and repeated short bursts (baseline 40, burst 130 req/s, durations {0.1, 0.5, 2, 10}s, 5-10 repeats each, pooled for percentiles). A deadline sensitivity set ({25, 50, 100}ms) was frozen from already-collected recheck data *before* these runs and not adjusted afterward.

No adaptive controller has been built. Every run in this bundle uses one fixed worker/thread configuration for its entire duration (including across every phase of a time-varying trace).

## Main findings

### 1. Correctness check passes, including across thread counts
11/11 edge-case sentences at threads=1: cosine similarity 1.000000, max abs diff ≈1.4–2.4e-7 against the `sentence-transformers` reference (tolerance: cosine ≥0.9999, max abs diff ≤1e-3, fp32-vs-fp32). Round 2 extended this to 20 representative benchmark-corpus sentences run at every `intra_op_num_threads` value the pilot configs actually use (1, 2, 4, 8) — floating-point reduction order in a parallel matmul can in principle differ by thread count. **Result: identical to ~1e-7 precision across all four thread counts, all pass.** See `correctness_check.json`.

### 2. Round 1's six findings and round 2's five findings were all real; every fix verified, not just asserted
Round 1 (percentile convention, timeout semantics, mislabeled IPC floor, shutdown/collection accounting, missing measurements, arrival-duration mismatch) — detail in the previous version of this bundle and `docs/pilot_config_matrix.md`.

Round 2, from a second independent review of round 1's own raw logs:
- **NPZ lazy-loading overhead was inside the timed request loop.** `np.load()`'s lazy `NpzFile` re-decodes an archive member on every access. Fixed by materializing into plain ndarrays once, up front; payload slicing now happens before `enqueue_ts` is stamped, not during `Task` construction after it. **Measured effect**: the unloaded IPC-floor measurement dropped from ~2.4ms to ~1.0ms p50 once fixed — about 1.4ms of the old number was NPZ-decode overhead, not IPC. Also stopped calling this "pure IPC": a new in-process pickle-only microbenchmark (~0.07ms p50) decomposes serialization cost from the remaining pipe-transfer/wake-up/unpickle overhead.
- **Generator-process background threads were never pinned or verified.** `os.sched_setaffinity(0, ...)` only affects the calling thread; the collector thread, resource-monitor thread, and `task_queue`'s own feeder thread were unchecked. **Checked and found real violations**: on a fresh verification run, the collector and resource-monitor threads both had full `0-39` affinity, contaminating the generator's dedicated CPU budget. Fixed with `common.affinity.verify_and_repin_all_threads()`, run before and after the generator loop; confirmed 0 unresolved violations after the fix on the same run. Unresolved violations (worker or generator) now set `run_valid: false` and exit the harness with code 2, not just a printed warning.
- **Forced-shutdown status writes were in the wrong order relative to their data.** `STATUS_COMPLETED` could previously be written before `completion_ts`, so a kill between the two lines would leave a status claiming a valid timestamp that was actually still uninitialized. Fixed by writing data before the flag everywhere; timestamp arrays now default to an explicit invalid sentinel (not 0.0); recovered records are validated (present, monotonically ordered) before being trusted, with a new `UNKNOWN` outcome for validation failures. `UNFINISHED` is now reported as "not observed as dequeued," not "confirmed never dequeued" — an irreducible microsecond-scale race remains and is now stated rather than glossed over. Worker exit codes are checked (a crash no longer silently passes as "orderly"). Sentinel insertion is now timeout-bounded (an unbounded `task_queue.put(None)` on a full queue was a second, distinct hang risk from the one fixed in round 1).
- **Rate/duration reporting still conflated two quantities.** `achieved_arrival_rate_hz_over_nominal_duration` (= N / nominal duration) and `last_arrival_offset_s` (the realized schedule span, not guaranteed equal to the nominal duration for a bounded Poisson process) are now separate fields.
- **Throughput conflated in-window and drain-time completions.** Now two explicit fields: `completed_within_nominal_window` and `completed_including_drain`.

All fixes are in the code included in this bundle (`smoke_bench.py`, `worker.py`, `measure_ipc_floor.py`, `common/`), not just described.

### 3. Calibration: configs saturate at very different offered loads (tuning split)
| Config | Approx. saturation point | Notes |
|---|---|---|
| W1T8 | ~100–150 req/s (throughput plateaus ~117/s) | e2e p50 jumps from 27ms (rate=100) to 786ms (rate=150) |
| W2T4 | ~150–220 req/s (plateaus ~181/s) | still tracking offered load at rate=150 |
| W4T2 | not reached by 220 req/s | throughput=214/s, queueing delay p50 still 6.7ms at rate=220 |
| W8T1 | not reached by 220 req/s | throughput=213.5/s, queueing delay p50 still 1.4ms at rate=220 |

See `plots/calibration_throughput_and_latency.png`. Full data: `calibration/calibration_summary.csv`.

### 4. Matched sweep (round 1, all 4 configs, 3 repeats): W1T8 is fragile, not part of the real trade-off
Frozen rates (from calibration, see `calibration/calibration_decision.json`): low=40, medium=90, stress=130 req/s. Same arrival schedule replayed per rate across all 4 configs, 3 repeats each, run order randomized. e2e latency (ms), min–max across repeats:

| Config | low (p50 / p99) | medium (p50 / p99) | stress (p50 / p99) |
|---|---|---|---|
| W1T8 | 12.3–12.6 / 29.4–32.2 | 22.2–23.2 / **81.6–94.3** | **474.8–732.8 / 812.1–1441.2** |
| W2T4 | 13.5–13.9 / 22.0–22.8 | **12.7–13.5** / 31.1–33.6 | 14.3–16.2 / 44.4–47.5 |
| W4T2 | 20.0–20.5 / 24.2–25.0 | 16.2–16.6 / 24.5–28.7 | 16.0–16.4 / 29.2–30.8 |
| W8T1 | 31.7–31.9 / 35.5–41.6 | 26.6–26.9 / 35.4–39.7 | 24.5–25.0 / 35.6–39.0 |

See `plots/sweep_latency_by_config_and_rate.png`. Full data: `sweep/sweep_summary.csv`. All 36 runs had `orderly_shutdown: true` and `collection_confirmed: true`.

W1T8's story is straightforward and not really a trade-off: it wins by 1-8ms at low load and then collapses 30-50x under stress while the other three stay close to their low-load numbers. That's a fragile config with a narrow upside and a catastrophic downside, not a config that "wins under different traffic" in the interesting sense.

**Correction (round 2 review): the original version of this section stopped there and concluded "W2T4 looks like a robust default" — that conclusion was drawn from p50 and did not hold up under p99.** See §5.

### 5. The real crossover is W2T4 vs. W4T2, on p99 specifically — confirmed by a focused, higher-rigor recheck
Checking p99 (not p50) in the same round-1 sweep table above: **W2T4 wins p99 at low load (22.0–22.8 vs. W4T2's 24.2–25.0), but W4T2 wins p99 at medium (24.5–28.7 vs. 31.1–33.6) and stress (29.2–30.8 vs. 44.4–47.5).** This is a real crossover between two configs that both look "robust" on p50 — the round-1 framing missed it by only looking at the median.

To check this wasn't a round-1-sweep artifact, W2T4 and W4T2 were rechecked under the round-2 fixed harness with meaningfully more rigor: **3 fresh seeds (20001-20003, not the sweep's 5001-5003)**, matched schedules across both configs, randomized run order, and duration padded to make ≥5,000 requests likely (not guaranteed by construction -- see §7's round-3 correction; all 18 runs did realize 5,128-5,455 requests in fact, all `run_valid: true`, all `COMPLETED` with zero rejections/timeouts at these rates). e2e latency (ms), min–max across the 3 seeds:

| Config | low=40 (p50 / p99) | medium=90 (p50 / p99) | stress=130 (p50 / p99) |
|---|---|---|---|
| W2T4 | 12.6–13.2 / 21.2–21.7 | 12.3–13.2 / 27.6–36.2 | 13.9–14.7 / **52.6–75.9** |
| W4T2 | 19.2–19.6 / 22.7–23.3 | 15.5–15.9 / **24.1–25.5** | 15.2–15.5 / **27.3–35.6** |

Full data: `recheck/recheck_summary.csv`. Plot: `plots/recheck_w2t4_vs_w4t2.png`.

**The crossover is confirmed and the effect is stronger under more rigorous testing, not weaker**: W2T4 wins p50 at every rate tested (always lower median latency), but W4T2 wins p99 at medium and stress, and by stress the gap has widened to roughly 2-2.7x (W2T4 p99 up to 75.9ms vs. W4T2's 35.6ms). **This means the trade-off this pilot is actually testing is specifically a tail-latency trade-off, not a median-latency one** — W2T4's median advantage persists everywhere, but its tail gets much worse under load while W4T2's tail stays comparatively flat. That's arguably a more interesting and more relevant finding for a deadline-oriented serving system than a median crossover would have been, since SLAs are usually written against p99, not p50.

This is now solid evidence for go/no-go criterion 1. Criterion 3 (room above a strong fixed baseline) is still open: nothing here yet shows that switching between W2T4 and W4T2 beats simply always running whichever of the two is closer to your actual traffic mix — that requires the baseline comparison and transition-cost measurement that come next.

**Independently verified**: all 95,226 recheck requests (counts, percentiles, timestamp ordering, in-window throughput, matched schedules) were checked and confirmed to agree. **Go/no-go criterion 1 passes provisionally for this workload and this WSL environment.**

### 6. Round 3 harness fixes, applied before the time-varying phase
- **Recheck padding correction**: `run_recheck.py`'s duration padding (5% above the strict `MIN_REQUESTS/rate` minimum) makes the 5,000-request target *likely*, not guaranteed by construction -- Poisson arrival count has real variance. All 18 recheck runs did in fact exceed 5,000; that's stated as an observed fact now, not a guarantee, and `run_one()` validates the actual count post-hoc with a warning if a future run falls short.
- **Time-varying harness extension** (`scripts/common/trace.py`, `smoke_bench.py --trace-file`): a trace is a saved JSON file (phases -> a concatenated, globally-ordered arrival schedule, computed once) replayed as one continuous run -- workers and queues spawned once, kept alive across every phase boundary, drained only at the very end. Raw rows carry `phase_id` (attributed by *scheduled* arrival, so a request delayed across a phase boundary by backlog is still attributed to the phase it was scheduled in) and `input_id`. Per-phase latency/outcome/throughput are computed independently from the phase-filtered raw rows -- never derived by averaging a phase value into the whole-trace number or vice versa.

### 7. Time-varying results: the crossover holds under load changes and repeated bursts, and gets no worse on recovery
**Sustained low-high-low (40→130→40 req/s, 50s/40s/50s phases, 2 seeds, `results/time_varying/sustained_*`)**, e2e latency (ms), min-max across seeds:

| Config | low1 (before spike) p50/p99 | high (spike) p50/p99 | low2 (after spike) p50/p99 |
|---|---|---|---|
| W2T4 | 12.3-12.5 / 22.0-23.2 | 13.7-14.7 / **49.4-68.9** | 12.7 / 21.2-21.8 |
| W4T2 | 19.1 / 23.7-25.5 | 14.8-14.9 / **30.3-30.4** | 18.8-19.4 / 22.6-22.7 |

W2T4 wins p99 in both low phases (before *and* after the spike); W4T2 wins p99 dramatically during the spike itself (30.3-30.4ms vs. 49.4-68.9ms). **No visible carryover effect**: `low2`'s p50/p99 are close to `low1`'s for both configs (no formal significance test was run) -- neither config shows an obvious lingering effect from the preceding high-load phase into the recovery phase, within this trace's 50s post-spike window. Missed-deadline fraction (whole trace, all three frozen thresholds, `n_offered` = every scheduled request; see `results/time_varying/deadline_set.json` and `analysis_summary.json`):

| Config | d=25ms | d=50ms | d=100ms |
|---|---|---|---|
| W2T4 | 7.7-10.3% | 0.5-1.3% | 0.0% |
| W4T2 | 1.8-2.3% | 0.0% | 0.0% |

At the moderate 50ms deadline -- the one chosen specifically to separate the two configs -- **W4T2 has essentially zero misses while W2T4 still misses 0.5-1.3% of all offered requests**, driven entirely by the high-load phase.

**Repeated short bursts (baseline 40, burst 130 req/s, `results/time_varying/burst_*`)**, pooled e2e p99 across all repeated burst-window requests of each duration:

| Duration | n pooled | W2T4 p99 | W4T2 p99 |
|---|---|---|---|
| 0.1s | 114 | 46.3ms | 24.2ms |
| 0.5s | 646 | 62.2ms | 27.9ms |
| 2s | 2,626 | 76.2ms | 28.9ms |
| 10s | 6,412 | 46.8ms | 26.8ms |

See `plots/burst_p99_by_duration.png`. **W4T2's pooled burst p99 is flat (~24-29ms) across every duration tested; W2T4's is 2-3x worse at every duration**, peaking at the 2s burst (76.2ms) and easing somewhat by 10s (46.8ms) -- plausibly because a 10s burst is long enough for W2T4 to reach a new (still worse-than-W4T2, but less transient) quasi-steady-state, while 0.5-2s bursts catch the system mid-transient. All 12 time-varying runs had `run_valid: true` and 100% `COMPLETED` (zero rejections/timeouts/unfinished at these particular loads).

**This extends the confirmed crossover from steady-state traffic to time-varying traffic**: it is not an artifact of the fixed-rate sweep design. Still no evidence yet on whether an adaptive policy could beat always-W4T2 (which now looks like the stronger baseline given deadline-oriented framing) -- that's the open question for the baseline comparison and transition-cost measurement that come next, per the pilot's own ordering (transition costs only get measured if a real crossover exists, which is now confirmed).

## Limitations

- **WSL2 environment**: all results are `[WSL2, virtualized topology]` — no NUMA/socket visibility, CPU pinning is guest-level only (see `docs/environment_report.md`). Not resolved in this pass; native-Linux confirmation is still outstanding for anything topology-sensitive.
- **No transition-cost measurement yet.** This bundle establishes that W2T4/W4T2 trade off on p99 under both steady and time-varying traffic (go/no-go criterion 1, now provisionally passed) — it says nothing about criterion 2 (transition costs) yet. That's next, per the pilot's own ordering (only measure transitions once a real crossover is confirmed, which it now is).
- **No strong-fixed-config or simple-adaptive-policy baseline run yet.** Nothing here shows switching beats always running whichever of W2T4/W4T2 is closer to the traffic mix — that's the next open question (criterion 3), pending transition-cost measurement and a real comparison against both fixed baselines.
- **Correctness test set is still modest** (11 hand-built edge cases + 20 corpus samples) even after the round-2 extension across thread counts; not drawn from a standard benchmark.
- **Recheck and time-varying repeat counts are small** (3 seeds for the recheck, 2 for sustained, 5-10 repeated bursts pooled per duration) — enough to confirm the crossover holds and even strengthens/generalizes, but not the full ≥3-5-repeats-with-varied-conditions the complete pilot design calls for before final reported numbers.
- **Sustained-trace phase durations (50s/40s/50s) and burst gap sizing (`max(5, 2×duration)`) were chosen by this session, not independently reviewed before running** — worth a second look given how much the findings depend on the high-phase and burst-window boundaries specifically.
- Calibration, sweep, recheck, and time-varying traces all ran on the same synthetic corpus (240 template-generated sentences, not a standard dataset) — see `docs/pilot_config_matrix.md` for why, and note this limits generalization claims. The sweep's, recheck's, and now the time-varying traces' eval-split seeds (5001-5003, 20001-20003, 40001-40002, 41001) are all now "inspected" — a further, still-unused seed set must be reserved for eventual policy evaluation, not reused in upcoming phases.
- **Deadline set was applied only to the sustained trace, not the bursts**, in this pass — `analyze_time_varying.py` computes it for both, but the burst write-up above reports pooled p99 rather than deadline-miss fractions; worth adding to the next bundle if deadline framing matters for bursts specifically.

## Questions for review

1. Go/no-go criterion 1 is now provisionally confirmed under steady, sustained-change, AND repeated-burst traffic, all pointing the same direction (W2T4 wins median/low-load tail, W4T2 wins tail under load). Is this sufficient evidence to move to criterion 2 (transition-cost measurement), or is there a traffic shape not yet tested that should be checked first?
2. The sustained trace showed no obvious carryover effect (low2 ≈ low1 for both configs, not statistically tested) within a 50s post-spike window — is 50s long enough to be confident there's no slower-decaying effect, or should recovery-phase duration itself be varied in a future run?
3. The burst p99 pattern is non-monotonic in duration (worst at 2s, better at both 0.1s and 10s) for W2T4 specifically — is this queueing-dynamics explanation (mid-transient vs. settled-into-new-steady-state) worth investigating further before moving on, or is it a secondary detail relative to the primary W2T4-vs-W4T2 comparison?
4. Any concerns with the shared-memory "data before flag" pattern (`multiprocessing.Array` written directly by the worker, timestamps before status) for forced-termination reconciliation? Is there a simpler or more standard pattern for this that was missed?
5. The correctness check's tolerance (cosine ≥0.9999, max abs diff ≤1e-3) was chosen for fp32-vs-fp32 comparison — does that bar look right, or too loose/tight given the observed ~1e-7 actual differences (now confirmed stable across thread counts too)?
