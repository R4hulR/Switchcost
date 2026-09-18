"""Open-loop load-test harness: spawns pinned worker processes behind one
shared FIFO queue, replays a scheduled arrival sequence against them, and
writes raw per-request logs + a run summary. This is the smoke-test /
general-purpose harness described in docs/pilot_config_matrix.md, and the
same harness used for calibration and the matched fixed-config sweep.

Deliberately NOT an adaptive controller: --config selects one fixed
worker/thread configuration for the whole run.

Usage:
  python scripts/smoke_bench.py --config W2T4 --rate 20 --duration 20 \
      --out results/smoke_W2T4
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import multiprocessing as mp
import os
import queue as queue_module
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import env_setup  # noqa: E402

env_setup.pin_single_threaded_blas()  # generator/main process also does numpy work (schedule prep)

import numpy as np  # noqa: E402

from common import affinity  # noqa: E402
from common.protocol import (  # noqa: E402
    CONFIGS,
    GENERATOR_CPUS,
    STATUS_COMPLETED,
    STATUS_DEQUEUED,
    STATUS_TIMED_OUT,
    Task,
    WorkerReady,
)
from common.stats import PERCENTILE_METHOD, summarize  # noqa: E402
from common.trace import build_schedule, load_trace  # noqa: E402
from worker import worker_main  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PRETOK_DIR = REPO_ROOT / "models" / "pretokenized"
TS_SENTINEL = -1.0  # shared-array "not written yet" marker; time.monotonic() is always positive


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def read_proc_status_field(pid: int, field: str) -> int | None:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith(field + ":"):
                    return int(line.split()[1])  # kB
    except (FileNotFoundError, ProcessLookupError):
        return None
    return None


def host_snapshot() -> dict:
    loadavg = os.getloadavg()
    meminfo = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, val = line.split(":", 1)
            if key in ("MemTotal", "MemFree", "MemAvailable", "SwapTotal", "SwapFree"):
                meminfo[key] = int(val.strip().split()[0])  # kB
    return {"loadavg_1_5_15": loadavg, "meminfo_kb": meminfo}


def collector_loop(result_queue, results: list, stop_event: threading.Event) -> None:
    while not (stop_event.is_set() and result_queue.empty()):
        try:
            results.append(result_queue.get(timeout=0.2))
        except queue_module.Empty:
            continue


def wait_for_collection(results: list, expected_count: int, timeout_s: float) -> bool:
    """Actively wait (not just assume) until `results` has grown to
    `expected_count` items, or `timeout_s` elapses. Returns whether
    collection was confirmed complete. This is what makes an orderly
    shutdown's outcome accounting trustworthy rather than merely likely
    (2026-09-17 review: "does not establish that completion records were
    safely collected")."""
    deadline = time.monotonic() + timeout_s
    while len(results) < expected_count and time.monotonic() < deadline:
        time.sleep(0.02)
    return len(results) >= expected_count


def resource_monitor_loop(worker_pids: dict[int, int], peak_rss_kb: dict[int, int], stop_event) -> None:
    """Polls /proc/<pid>/status VmRSS for each worker while it's still
    alive and tracks the max seen. VmHWM (kernel high-water-mark) can only
    be read from a live process, and by the time we'd read it after
    join()/terminate() the /proc entry is already gone (2026-09-17 review:
    "every peak_VmHWM entry is null" -- this replaces that approach)."""
    while not stop_event.is_set():
        for wid, pid in worker_pids.items():
            rss = read_proc_status_field(pid, "VmRSS")
            if rss is not None:
                peak_rss_kb[wid] = max(peak_rss_kb.get(wid, 0), rss)
        time.sleep(0.1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS), default="W2T4")
    ap.add_argument("--rate", type=float, default=20.0, help="requests/sec (ignored if --trace-file given)")
    ap.add_argument("--duration", type=float, default=20.0, help="seconds of scheduled arrivals (ignored if --trace-file given)")
    ap.add_argument("--arrival", choices=["poisson", "fixed"], default="poisson")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--trace-file",
        type=str,
        default=None,
        help=(
            "path to a saved trace JSON (scripts/common/trace.py build_phased_schedule + "
            "save_trace) -- a time-varying multi-phase arrival trace, replayed as ONE "
            "continuous run (workers/queues stay alive across phase boundaries; no "
            "drain/restart/re-warm between phases). Overrides --rate/--duration."
        ),
    )
    ap.add_argument("--queue-capacity", type=int, default=256)
    ap.add_argument("--max-queue-wait", type=float, default=2.0, help="seconds; see §11")
    ap.add_argument("--drain-timeout", type=float, default=10.0, help="seconds; see §11")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--split", choices=["tuning", "eval"], default="eval")
    ap.add_argument("--run-id", type=str, default=None)
    args = ap.parse_args()

    if args.run_id:
        run_id = args.run_id
    elif args.trace_file:
        run_id = f"{args.config}_trace_{Path(args.trace_file).stem}_{int(time.time())}"
    else:
        run_id = f"{args.config}_r{args.rate:g}_{args.arrival}_seed{args.seed}_{int(time.time())}"

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Materialize into plain in-memory ndarrays (2026-09-17 review, round 2):
    # np.load() on a .npz returns a lazy NpzFile -- every data["key"] access
    # re-reads/re-decodes that archive member from the zip, which is real,
    # avoidable per-request overhead if done inside the timed generator
    # loop. np.array(...) forces a one-time copy into ordinary ndarrays up
    # front, outside any timed section, so slicing during the loop is plain
    # numpy view creation (no I/O, no decode).
    npz = np.load(PRETOK_DIR / f"{args.split}.npz")
    input_ids_all = np.array(npz["input_ids"])
    token_type_ids_all = np.array(npz["token_type_ids"])
    attention_mask_all = np.array(npz["attention_mask"])
    n_pool = len(npz["ids"])
    npz.close()

    cfg = CONFIGS[args.config]
    n_workers = cfg["workers"]
    threads = cfg["threads"]
    cpu_groups = cfg["cpu_groups"]

    if args.trace_file:
        trace = load_trace(Path(args.trace_file))
        requests_meta = trace["requests"]  # already offset-sorted by construction (see common/trace.py)
        schedule_offsets = [r["offset"] for r in requests_meta]
        phase_by_request: dict[int, str] = {r["request_id"]: r["phase_id"] for r in requests_meta}
        phase_windows = {p["phase_id"]: p for p in trace["phases"]}
        nominal_duration_s = trace["phases"][-1]["end_offset"] if trace["phases"] else 0.0
        trace_arrival = trace["arrival"]
        print(f"[{run_id}] Loaded trace {args.trace_file}: {len(trace['phases'])} phase(s), seed={trace['seed']}")
    else:
        schedule_offsets = build_schedule(args.rate, args.duration, args.seed, args.arrival)
        phase_by_request = {i: "main" for i in range(len(schedule_offsets))}
        phase_windows = {
            "main": {
                "phase_id": "main", "rate_hz": args.rate, "duration_s": args.duration,
                "start_offset": 0.0, "end_offset": args.duration, "n_requests": len(schedule_offsets),
            }
        }
        nominal_duration_s = args.duration
        trace_arrival = args.arrival

    n_requests = len(schedule_offsets)
    if n_requests == 0:
        raise SystemExit("Schedule produced zero requests -- check --rate/--duration or --trace-file")

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue(maxsize=args.queue_capacity)
    result_queue = ctx.Queue()
    status_queue = ctx.Queue()
    # lock=False: each request_id slot is written by exactly one worker,
    # so there's no cross-worker write contention to protect against.
    shared_status = ctx.Array(ctypes.c_ubyte, n_requests, lock=False)
    shared_exec_start = ctx.Array(ctypes.c_double, n_requests, lock=False)
    shared_completion = ctx.Array(ctypes.c_double, n_requests, lock=False)
    for i in range(n_requests):  # explicit sentinel, not the default 0.0 (see TS_SENTINEL)
        shared_exec_start[i] = TS_SENTINEL
        shared_completion[i] = TS_SENTINEL

    run_invalid_reasons: list[str] = []

    print(f"[{run_id}] Spawning {n_workers} worker(s) for config {args.config} (threads={threads})")
    workers = []
    for wid, cpus in enumerate(cpu_groups):
        p = ctx.Process(
            target=worker_main,
            args=(
                wid,
                cpus,
                threads,
                args.max_queue_wait,
                task_queue,
                result_queue,
                status_queue,
                shared_status,
                shared_exec_start,
                shared_completion,
            ),
            daemon=False,
        )
        p.start()
        workers.append(p)

    ready: dict[int, WorkerReady] = {}
    ready_deadline = time.monotonic() + 60.0
    while len(ready) < n_workers:
        if time.monotonic() > ready_deadline:
            raise RuntimeError(f"Only {len(ready)}/{n_workers} workers reported ready within 60s")
        wr: WorkerReady = status_queue.get(timeout=60.0)
        ready[wr.worker_id] = wr
        n_initial = len(wr.initial_affinity_violations)
        n_post_fix = len(wr.post_fix_affinity_violations)
        print(
            f"  worker {wr.worker_id} pid={wr.pid} ready, warmup={wr.warmup_ms:.1f}ms, "
            f"initial_affinity_violations={n_initial}, post_fix_violations={n_post_fix}"
        )
        if n_post_fix:
            for v in wr.post_fix_affinity_violations:
                print(f"    UNFIXABLE AFFINITY VIOLATION: {v}")
            run_invalid_reasons.append(
                f"worker {wr.worker_id}: {n_post_fix} unresolved affinity violation(s) after re-pin attempt"
            )

    before_snapshot = host_snapshot()
    worker_pids = {wid: w.pid for wid, w in ready.items()}
    worker_rss_before = {wid: read_proc_status_field(pid, "VmRSS") for wid, pid in worker_pids.items()}

    results: list = []
    collector_stop = threading.Event()
    collector_thread = threading.Thread(
        target=collector_loop, args=(result_queue, results, collector_stop), daemon=True
    )
    collector_thread.start()

    peak_rss_kb: dict[int, int] = {}
    monitor_stop = threading.Event()
    monitor_thread = threading.Thread(
        target=resource_monitor_loop, args=(worker_pids, peak_rss_kb, monitor_stop), daemon=True
    )
    monitor_thread.start()

    try:
        affinity.set_affinity(GENERATOR_CPUS)
    except OSError as e:
        print(f"WARNING: could not pin generator to {GENERATOR_CPUS}: {e}")
        run_invalid_reasons.append(f"generator: could not set process affinity: {e}")

    # Verify + re-pin EVERY thread of the generator process, not just the
    # main thread (2026-09-17 review, round 2): os.sched_setaffinity(0, ...)
    # only affects the calling thread. The collector and monitor threads
    # just started above need their own affinity checked and fixed, the
    # same way worker.py already does for worker processes. task_queue's
    # own feeder thread (spawned lazily on this process's first put() to
    # it) doesn't exist yet at this point -- checked again after the
    # generator loop below, once it certainly does.
    gen_pid = os.getpid()
    gen_initial_violations, gen_post_fix_violations = affinity.verify_and_repin_all_threads(
        gen_pid, GENERATOR_CPUS
    )
    if gen_post_fix_violations:
        for v in gen_post_fix_violations:
            print(f"    UNFIXABLE GENERATOR AFFINITY VIOLATION (pre-loop): {v}")
        run_invalid_reasons.append(
            f"generator (pre-loop): {len(gen_post_fix_violations)} unresolved affinity violation(s)"
        )

    clock_info = time.get_clock_info("monotonic")

    enqueue_log: dict[int, float] = {}  # request_id -> enqueue_ts (successful)
    rejected: dict[int, float] = {}  # request_id -> attempt_ts (queue was full)
    prep_times_ms: list[float] = []  # input-slice prep time, measured separately from enqueue/IPC delay

    print(
        f"[{run_id}] Replaying {n_requests} scheduled arrivals over {schedule_offsets[-1]:.2f}s "
        f"(nominal window {nominal_duration_s}s, {len(phase_windows)} phase(s), {trace_arrival})"
    )
    t_run_start = time.monotonic()
    for request_id, offset in enumerate(schedule_offsets):
        target = t_run_start + offset
        while True:
            remaining = target - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(remaining, 0.002))

        idx = request_id % n_pool
        # Prepare the payload BEFORE stamping enqueue_attempt_ts (2026-09-17
        # review, round 2): slicing is now plain numpy view creation over
        # already-materialized arrays (see above), not an NPZ decode, but
        # the discipline itself matters regardless of how cheap prep is --
        # any work done between the timestamp and the actual put_nowait()
        # call inflates queueing_delay with generator-side prep time that
        # isn't queueing or IPC. Prep time is measured explicitly rather
        # than assumed negligible.
        prep_start = time.monotonic()
        input_ids = input_ids_all[idx : idx + 1]
        token_type_ids = token_type_ids_all[idx : idx + 1]
        attention_mask = attention_mask_all[idx : idx + 1]
        prep_times_ms.append((time.monotonic() - prep_start) * 1000.0)

        attempt_ts = time.monotonic()  # single point of truth for both outcomes below; nothing but put_nowait follows
        task = Task(
            request_id=request_id,
            scheduled_ts=target,
            enqueue_ts=attempt_ts,
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        try:
            task_queue.put_nowait(task)
            enqueue_log[request_id] = attempt_ts
        except queue_module.Full:
            rejected[request_id] = attempt_ts
    t_gen_finished = time.monotonic()

    print(f"[{run_id}] Generator finished scheduling. rejected_at_enqueue={len(rejected)}")

    # Re-check generator-process affinity now that task_queue's own feeder
    # thread (spawned lazily on this process's first put()) certainly
    # exists (2026-09-17 review, round 2) -- the pre-loop check above
    # couldn't see it.
    gen_late_initial, gen_late_post_fix = affinity.verify_and_repin_all_threads(gen_pid, GENERATOR_CPUS)
    if gen_late_post_fix:
        for v in gen_late_post_fix:
            print(f"    UNFIXABLE GENERATOR AFFINITY VIOLATION (post-loop): {v}")
        run_invalid_reasons.append(
            f"generator (post-loop): {len(gen_late_post_fix)} unresolved affinity violation(s)"
        )

    # Bound sentinel insertion too (2026-09-17 review, round 2): task_queue
    # has a fixed maxsize, so a blocking put(None) with no timeout could
    # itself hang indefinitely if the queue is still full from backlog --
    # defeating the drain_deadline mechanism before it's ever enforced.
    drain_deadline = time.monotonic() + args.drain_timeout
    sentinel_failures = 0
    for _ in workers:
        remaining = max(0.0, drain_deadline - time.monotonic())
        try:
            task_queue.put(None, timeout=remaining)
        except queue_module.Full:
            sentinel_failures += 1
    if sentinel_failures:
        print(
            f"WARNING: could not enqueue {sentinel_failures} sentinel(s) before the drain deadline "
            f"(queue still full) -- affected workers will be force-terminated."
        )

    orderly_shutdown = sentinel_failures == 0
    for p in workers:
        remaining = max(0.0, drain_deadline - time.monotonic())
        p.join(timeout=remaining)
        if p.is_alive():
            orderly_shutdown = False

    terminated = []
    if not orderly_shutdown:
        for p in workers:
            if p.is_alive():
                terminated.append(p.pid)
                p.terminate()
                p.join(timeout=5.0)
        if terminated:
            print(f"WARNING: had to terminate workers still running past drain deadline: {terminated}")

    monitor_stop.set()  # workers are gone (or about to be) either way; stop polling /proc

    # Check worker exit codes before trusting "orderly" (2026-09-17 review,
    # round 2): p.is_alive() being False doesn't distinguish a clean exit
    # from a crash (unhandled exception -> nonzero exitcode). A crashed
    # worker's queue-flush-on-exit guarantee is less certain than a clean
    # one's, so a nonzero exit code downgrades orderly_shutdown even though
    # the process is no longer "alive" in the sense the join loop checked.
    worker_exit_codes = {wid: p.exitcode for wid, p in zip(sorted(ready), workers)}
    # 0 = clean exit. -15 = SIGTERM, expected for a worker we force-terminated
    # past the drain deadline (already reflected in orderly_shutdown/terminated
    # above). Anything else means the worker's own process exit was abnormal.
    crashed = {wid: code for wid, code in worker_exit_codes.items() if code not in (0, -15, None)}
    if crashed:
        print(f"WARNING: worker(s) exited with unexpected code (not 0/SIGTERM): {crashed}")
        run_invalid_reasons.append(f"worker(s) crashed: {crashed}")
        orderly_shutdown = False

    expected_completed = len(enqueue_log)
    if orderly_shutdown:
        # Every worker exited normally, which (per multiprocessing's own
        # documented behavior) only happens after that worker's queue
        # feeder thread has flushed all its put()s into the pipe -- so
        # every enqueued request's Result IS in the pipe. Actively wait for
        # the collector thread to actually read all of them rather than
        # assuming a fixed grace period was enough (2026-09-17 review).
        collection_confirmed = wait_for_collection(results, expected_completed, timeout_s=10.0)
        if not collection_confirmed:
            print(
                f"WARNING: orderly shutdown but only collected {len(results)}/{expected_completed} "
                f"results within the confirmation timeout -- treat as unconfirmed, see summary."
            )
    else:
        # Forced termination (or a crashed worker): don't assume anything.
        # Give a short grace period for whatever was already in the pipe,
        # then stop -- the shared-memory arrays (not the queue) resolve
        # anything still missing below.
        time.sleep(0.5)
        collection_confirmed = len(results) >= expected_completed

    collector_stop.set()
    collector_thread.join(timeout=5.0)

    after_snapshot = host_snapshot()

    # --- reconcile every scheduled request into exactly one outcome ---
    results_by_id = {r.request_id: r for r in results}
    rows = []
    for request_id, offset in enumerate(schedule_offsets):
        scheduled_ts = t_run_start + offset

        if request_id in rejected:
            rows.append(
                {
                    "request_id": request_id,
                    "scheduled_ts": scheduled_ts,
                    "enqueue_ts": None,
                    "enqueue_attempt_ts": rejected[request_id],
                    "execution_start": None,
                    "completion_ts": None,
                    "worker_id": None,
                    "outcome": "REJECTED",
                    "outcome_confirmed": True,
                }
            )
            continue

        r = results_by_id.get(request_id)
        if r is not None:
            rows.append(
                {
                    "request_id": request_id,
                    "scheduled_ts": scheduled_ts,
                    "enqueue_ts": r.enqueue_ts,
                    "enqueue_attempt_ts": r.enqueue_ts,
                    "execution_start": r.execution_start,
                    "completion_ts": r.completion_ts,
                    "worker_id": r.worker_id,
                    "outcome": r.outcome,  # COMPLETED or TIMED_OUT
                    "outcome_confirmed": True,
                }
            )
            continue

        # No Result message arrived for this request. In the orderly case
        # this shouldn't happen if collection_confirmed is True; in the
        # forced-termination/crash case, consult the shared-memory status/
        # timestamp arrays written directly by the worker (not routed
        # through the possibly-corrupted queue pipe) to distinguish real
        # outcomes rather than blanket-labeling everything "UNFINISHED"
        # (2026-09-17 review: "reconciling missing IDs alone does not
        # prove their actual status").
        #
        # Recovered timestamps are validated, not trusted blindly (round 2
        # review: "validate recovered records"): a status flag is only
        # believed if its corresponding timestamp(s) are actually written
        # (!= TS_SENTINEL) and internally ordered (completion >= exec_start
        # >= enqueue_ts where applicable). A flag/data mismatch downgrades
        # the outcome to UNKNOWN instead of reporting a bogus latency.
        status = shared_status[request_id]
        enqueue_ts = enqueue_log.get(request_id)
        exec_start_raw = shared_exec_start[request_id]
        completion_raw = shared_completion[request_id]
        exec_start = exec_start_raw if exec_start_raw != TS_SENTINEL else None
        completion_ts = completion_raw if completion_raw != TS_SENTINEL else None

        if status == STATUS_COMPLETED:
            if completion_ts is not None and exec_start is not None and completion_ts >= exec_start >= (enqueue_ts or 0):
                outcome = "COMPLETED_UNCONFIRMED"  # worker finished the computation; Result message never arrived
            else:
                outcome = "UNKNOWN"  # status claimed COMPLETED but recovered data failed validation
                exec_start = None
                completion_ts = None
        elif status == STATUS_TIMED_OUT:
            if exec_start is not None:
                outcome = "TIMED_OUT_UNCONFIRMED"
            else:
                outcome = "UNKNOWN"
                exec_start = None
            completion_ts = None
        elif status == STATUS_DEQUEUED:
            # Confirmed dequeued (removed from the queue); the worker was
            # killed before deciding TIMED_OUT vs. running inference to
            # completion. True final outcome is genuinely unknown.
            outcome = "INTERRUPTED"
            completion_ts = None
        else:
            # STATUS_NOT_STARTED (default). Best available evidence is that
            # this request was never dequeued, but this is NOT airtight --
            # see the STATUS_NOT_STARTED docstring in common/protocol.py
            # for the narrow, irreducible race this can't fully rule out.
            # Reported conservatively (2026-09-17 review, round 2), not as
            # "confirmed never dequeued".
            outcome = "UNFINISHED"
            exec_start = None
            completion_ts = None

        rows.append(
            {
                "request_id": request_id,
                "scheduled_ts": scheduled_ts,
                "enqueue_ts": enqueue_ts,
                "enqueue_attempt_ts": enqueue_ts,
                "execution_start": exec_start,
                "completion_ts": completion_ts,
                "worker_id": None,
                "outcome": outcome,
                "outcome_confirmed": False,  # every row reaching this branch is, by definition, not from a Result message
            }
        )

    for row in rows:
        # phase_id/input_id, added for time-varying-trace support (2026-09-17
        # review): phase_id is attributed by SCHEDULED arrival (the phase the
        # request's offset fell into at trace-build time), not by when it was
        # actually dequeued/executed -- so a request delayed across a phase
        # boundary is still attributed to the phase it was scheduled in.
        row["phase_id"] = phase_by_request.get(row["request_id"], "main")
        row["input_id"] = row["request_id"] % n_pool
        row["generator_delay_ms"] = (
            (row["enqueue_ts"] - row["scheduled_ts"]) * 1000.0
            if row["enqueue_ts"] is not None
            else None
        )
        row["queueing_delay_ms"] = (
            (row["execution_start"] - row["enqueue_ts"]) * 1000.0
            if row["execution_start"] is not None and row["enqueue_ts"] is not None
            else None
        )
        row["service_time_ms"] = (
            (row["completion_ts"] - row["execution_start"]) * 1000.0
            if row["completion_ts"] is not None and row["execution_start"] is not None
            else None
        )
        row["e2e_latency_ms"] = (
            (row["completion_ts"] - row["scheduled_ts"]) * 1000.0
            if row["completion_ts"] is not None
            else None
        )

    fieldnames = list(rows[0].keys())
    raw_path = out_dir / "raw_requests.csv"
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[{run_id}] Wrote {raw_path} ({len(rows)} rows)")

    outcome_counts = {}
    for row in rows:
        outcome_counts[row["outcome"]] = outcome_counts.get(row["outcome"], 0) + 1

    completed_rows = [r for r in rows if r["outcome"] == "COMPLETED"]
    completed_e2e = [r["e2e_latency_ms"] for r in completed_rows]
    completed_queue = [r["queueing_delay_ms"] for r in completed_rows if r["queueing_delay_ms"] is not None]
    completed_service = [r["service_time_ms"] for r in completed_rows if r["service_time_ms"] is not None]
    generator_delays = [r["generator_delay_ms"] for r in rows if r["generator_delay_ms"] is not None]

    # Per-phase breakdown (2026-09-17 review): computed independently from
    # the whole-trace numbers above, directly from the phase-filtered raw
    # rows -- never derived BY averaging whole-trace/other-phase
    # percentiles, and never used to derive the whole-trace numbers either.
    # Each phase's own completed subset gets its own summarize() call.
    latency_by_phase = {}
    outcome_counts_by_phase = {}
    throughput_by_phase = {}
    for phase_id, window in phase_windows.items():
        phase_rows = [r for r in rows if r["phase_id"] == phase_id]
        phase_completed = [r for r in phase_rows if r["outcome"] == "COMPLETED"]
        phase_outcome_counts = {}
        for r in phase_rows:
            phase_outcome_counts[r["outcome"]] = phase_outcome_counts.get(r["outcome"], 0) + 1
        outcome_counts_by_phase[phase_id] = phase_outcome_counts
        latency_by_phase[phase_id] = {
            "e2e": summarize([r["e2e_latency_ms"] for r in phase_completed]),
            "queueing_delay": summarize([r["queueing_delay_ms"] for r in phase_completed if r["queueing_delay_ms"] is not None]),
            "service_time": summarize([r["service_time_ms"] for r in phase_completed if r["service_time_ms"] is not None]),
        }
        phase_window_s = window["duration_s"]
        phase_start_mono = t_run_start + window["start_offset"]
        phase_end_mono = t_run_start + window["end_offset"]
        n_completed_in_phase_window = sum(
            1 for r in phase_completed if r["completion_ts"] is not None and phase_start_mono <= r["completion_ts"] <= phase_end_mono
        )
        throughput_by_phase[phase_id] = {
            "n_offered": len(phase_rows),
            "n_completed_total": len(phase_completed),
            "n_completed_within_phase_window": n_completed_in_phase_window,
            "rate_per_s_within_phase_window": n_completed_in_phase_window / phase_window_s if phase_window_s > 0 else None,
            "phase_window_s": phase_window_s,
        }

    # FIFO-order check: only meaningful with a single worker/consumer
    # (docs/pilot_config_matrix.md §4 footnote).
    fifo_check = None
    if n_workers == 1:
        dequeued_order = [
            r["request_id"]
            for r in sorted(
                (r for r in rows if r["execution_start"] is not None),
                key=lambda r: r["execution_start"],
            )
        ]
        is_sorted = dequeued_order == sorted(dequeued_order)
        fifo_check = {"single_worker": True, "dequeue_order_matches_enqueue_order": is_sorted}
        if not is_sorted:
            print("WARNING: single-worker dequeue order did NOT match enqueue order")

    # Rate/duration reporting, corrected (2026-09-17 review, round 2): a
    # bounded Poisson schedule need not end exactly at duration_s, so
    # "achieved rate" and "last arrival offset" are reported as two
    # distinct numbers rather than one derived from the other.
    last_arrival_offset_s = schedule_offsets[-1]
    achieved_arrival_rate_hz_over_nominal_duration = n_requests / nominal_duration_s if nominal_duration_s > 0 else None

    completion_ts_values = [r["completion_ts"] for r in rows if r["completion_ts"] is not None]
    measurement_window = {
        "run_wall_start_monotonic": t_run_start,
        "generator_finished_scheduling_at_monotonic": t_gen_finished,
        "first_scheduled_arrival_monotonic": t_run_start + schedule_offsets[0],
        "last_scheduled_arrival_monotonic": t_run_start + schedule_offsets[-1],
        "nominal_window_end_monotonic": t_run_start + nominal_duration_s,
        "first_completion_ts_monotonic": min(completion_ts_values) if completion_ts_values else None,
        "last_completion_ts_monotonic": max(completion_ts_values) if completion_ts_values else None,
    }

    # Throughput, split into two explicitly different windows (2026-09-17
    # review, round 2): completions that landed within the nominal
    # observation window vs. total completions including whatever finished
    # during the post-window drain. Conflating these into one number
    # overstates "throughput" by crediting drain-time work to the nominal
    # window's rate.
    nominal_window_end = measurement_window["nominal_window_end_monotonic"]
    n_completed_within_window = sum(
        1 for ts in completion_ts_values if ts <= nominal_window_end
    )
    throughput_within_nominal_window = (
        n_completed_within_window / nominal_duration_s if nominal_duration_s > 0 else None
    )
    throughput_including_drain = None
    throughput_including_drain_window_s = None
    if completion_ts_values:
        throughput_including_drain_window_s = (
            max(completion_ts_values) - measurement_window["first_scheduled_arrival_monotonic"]
        )
        if throughput_including_drain_window_s > 0:
            throughput_including_drain = len(completed_rows) / throughput_including_drain_window_s

    run_valid = len(run_invalid_reasons) == 0

    summary = {
        "run_id": run_id,
        "run_valid": run_valid,
        "run_invalid_reasons": run_invalid_reasons,
        "git_commit": git_commit(),
        "config": args.config,
        "workers": n_workers,
        "threads_per_worker": threads,
        "cpu_groups": cpu_groups,
        "generator_cpus": GENERATOR_CPUS,
        "resolved_worker_settings": {wid: w.resolved_settings for wid, w in ready.items()},
        # Stored relative to the repo root, not absolute -- an absolute path
        # bakes in the local machine's directory layout (and username, on a
        # typical /home/<user>/... layout) into a result artifact that may
        # get shared or published; relative is portable and sufficient to
        # locate the file within the repo.
        "trace_file": (
            str(Path(args.trace_file).resolve().relative_to(REPO_ROOT))
            if args.trace_file and Path(args.trace_file).resolve().is_relative_to(REPO_ROOT)
            else args.trace_file
        ),
        "phases": list(phase_windows.values()),
        "rate_hz_nominal": args.rate if not args.trace_file else None,  # meaningless for a multi-phase trace
        "duration_s_nominal": nominal_duration_s,
        "last_arrival_offset_s": last_arrival_offset_s,
        "achieved_arrival_rate_hz_over_nominal_duration": achieved_arrival_rate_hz_over_nominal_duration,
        "arrival": trace_arrival,
        "seed": args.seed,
        "split": args.split,
        "n_requests": n_requests,
        "queue_capacity": args.queue_capacity,
        "max_queue_wait_s": args.max_queue_wait,
        "drain_timeout_s": args.drain_timeout,
        "orderly_shutdown": orderly_shutdown,
        "collection_confirmed": collection_confirmed,
        "worker_exit_codes": worker_exit_codes,
        "sentinel_insertion_failures": sentinel_failures,
        "outcome_counts": outcome_counts,
        "outcome_counts_by_phase": outcome_counts_by_phase,
        "percentile_method": PERCENTILE_METHOD,  # see common/stats.py -- documented convention, used everywhere below
        "latency_ms_by_phase": latency_by_phase,  # computed independently per phase -- never averaged from/into the whole-trace numbers below
        "throughput_by_phase": throughput_by_phase,
        "latency_ms": {
            "e2e": summarize(completed_e2e),
            "queueing_delay": {
                **summarize(completed_queue),
                "note": (
                    "enqueue_ts to execution_start; includes generator-to-worker IPC transit "
                    "(pickle/pipe/unpickle/wake), not pure wait -- see docs §9 and "
                    "scripts/measure_ipc_floor.py for a dedicated unloaded decomposition"
                ),
            },
            "service_time": summarize(completed_service),
            "generator_delay": summarize(generator_delays),
            "input_prep_ms": summarize(prep_times_ms),
        },
        "throughput": {
            "completed_within_nominal_window": {
                "count": n_completed_within_window,
                "rate_per_s": throughput_within_nominal_window,
                "window_s": nominal_duration_s,
                "definition": "count of COMPLETED with completion_ts <= run_start + duration_s_nominal, divided by duration_s_nominal",
            },
            "completed_including_drain": {
                "count": len(completed_rows),
                "rate_per_s": throughput_including_drain,
                "window_s": throughput_including_drain_window_s,
                "definition": "len(COMPLETED) / (last completion_ts - first scheduled arrival), includes post-window drain-time completions",
            },
        },
        "measurement_window": measurement_window,
        "fifo_order_check": fifo_check,
        "generator_affinity_checks": {
            "pre_loop": {"initial_violations": gen_initial_violations, "post_fix_violations": gen_post_fix_violations},
            "post_loop": {"initial_violations": gen_late_initial, "post_fix_violations": gen_late_post_fix},
        },
        "worker_readiness": {
            wid: {
                "pid": w.pid,
                "warmup_ms": w.warmup_ms,
                "initial_affinity_violations": w.initial_affinity_violations,
                "post_fix_affinity_violations": w.post_fix_affinity_violations,
            }
            for wid, w in ready.items()
        },
        "worker_rss_kb": {"before": worker_rss_before, "peak_observed": peak_rss_kb},
        "host_snapshot": {"before": before_snapshot, "after": after_snapshot},
        "workers_terminated_past_drain_deadline": terminated,
        "monotonic_clock_resolution_us": clock_info.resolution * 1e6,
        "note_ipc_floor": (
            "This summary intentionally does NOT report an IPC-floor estimate -- "
            "queueing_delay under load is not a valid proxy for it (2026-09-17 review). "
            "Use scripts/measure_ipc_floor.py for a dedicated unloaded measurement, which "
            "itself is now reported as 'IPC + queue wake-up overhead', not 'pure IPC' "
            "(round 2 review correction)."
        ),
        "note_clock": (
            "All timestamps (generator and worker processes) use time.monotonic(), "
            "which on CPython/Linux is backed by the system-wide CLOCK_MONOTONIC and "
            "is therefore comparable across processes on this platform."
        ),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[{run_id}] Wrote {summary_path}")
    print(
        json.dumps(
            {"outcome_counts": outcome_counts, "latency_ms_e2e": summary["latency_ms"]["e2e"], "run_valid": run_valid},
            indent=2,
        )
    )

    if not run_valid:
        print(f"[{run_id}] RUN INVALID: {run_invalid_reasons}")
        sys.exit(2)


if __name__ == "__main__":
    main()
