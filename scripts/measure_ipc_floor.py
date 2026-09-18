"""Dedicated, unloaded measurement of generator<->worker overhead via
multiprocessing.Queue -- NOT derived from a loaded benchmark run.

2026-09-17 review (round 1) finding: smoke_bench.py's summary used to
report `queueing_delay`'s p50 under load as an "IPC floor estimate." That's
wrong whenever the queue actually has depth (e.g. 37ms and 496ms were
reported for overloaded runs) -- at that point queueing_delay IS real
queueing wait, not IPC overhead. This script isolates the unloaded
overhead properly: one worker, requests sent one at a time with a generous
gap so the worker is always idle when the next request arrives (queueing
wait ~0 by construction).

2026-09-17 review (round 2) correction: the earlier version of this script
itself sliced task payloads from a lazy NpzFile inline with timestamp
capture, and called the resulting number "pure IPC" -- overclaiming, since
`queueing_delay_ms` (enqueue_ts to execution_start) bundles pickling,
pipe transfer, OS wake-up scheduling latency for the blocked worker, and
unpickling together; none of that is separable from a single number
without instrumenting the queue internals. Fixed by (a) materializing
payload arrays once up front and slicing before the timestamp, matching
smoke_bench.py's discipline, and (b) adding a direct in-process
pickle/unpickle-only microbenchmark on an identical Task object, to at
least decompose "serialization cost" out from the rest ("pipe transfer +
worker wake-up + queue.get() overhead"), rather than leaving the whole
interval as one undifferentiated "IPC" label.

Usage: python scripts/measure_ipc_floor.py --n-trials 300 --gap-s 0.3
"""

from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import env_setup  # noqa: E402

env_setup.pin_single_threaded_blas()

import numpy as np  # noqa: E402

from common import affinity  # noqa: E402
from common.protocol import CONFIGS, GENERATOR_CPUS, Task, WorkerReady  # noqa: E402
from common.stats import summarize  # noqa: E402
from worker import worker_main  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PRETOK_DIR = REPO_ROOT / "models" / "pretokenized"
TS_SENTINEL = -1.0


def measure_pickle_roundtrip_ms(task: Task, n_reps: int = 500) -> dict:
    """Pure (de)serialization cost of one Task object, in-process, no queue
    or second process involved -- a lower bound on what pickling
    contributes to the measured queueing_delay, isolated from pipe
    transfer and worker wake-up scheduling."""
    times_ms = []
    for _ in range(n_reps):
        t0 = time.monotonic()
        blob = pickle.dumps(task)
        pickle.loads(blob)
        times_ms.append((time.monotonic() - t0) * 1000.0)
    return summarize(times_ms)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=300)
    ap.add_argument("--gap-s", type=float, default=0.3, help="time between requests; must exceed service time")
    ap.add_argument("--config", choices=list(CONFIGS), default="W1T8")
    ap.add_argument("--out", type=str, default="results/ipc_floor")
    args = ap.parse_args()

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Materialize once, outside the timed loop (2026-09-17 review, round 2)
    # -- see smoke_bench.py's identical comment for why this matters.
    npz = np.load(PRETOK_DIR / "eval.npz")
    input_ids_all = np.array(npz["input_ids"])
    token_type_ids_all = np.array(npz["token_type_ids"])
    attention_mask_all = np.array(npz["attention_mask"])
    n_pool = len(npz["ids"])
    npz.close()

    cfg = CONFIGS[args.config]
    cpus = cfg["cpu_groups"][0]
    threads = cfg["threads"]

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue(maxsize=8)
    result_queue = ctx.Queue()
    status_queue = ctx.Queue()
    shared_status = ctx.Array(ctypes.c_ubyte, args.n_trials, lock=False)
    shared_exec_start = ctx.Array(ctypes.c_double, args.n_trials, lock=False)
    shared_completion = ctx.Array(ctypes.c_double, args.n_trials, lock=False)
    for i in range(args.n_trials):
        shared_exec_start[i] = TS_SENTINEL
        shared_completion[i] = TS_SENTINEL

    p = ctx.Process(
        target=worker_main,
        args=(
            0,
            cpus,
            threads,
            30.0,  # generous max_queue_wait; this run should never be at risk of it
            task_queue,
            result_queue,
            status_queue,
            shared_status,
            shared_exec_start,
            shared_completion,
        ),
    )
    p.start()
    wr: WorkerReady = status_queue.get(timeout=60.0)
    print(f"worker ready pid={wr.pid} warmup={wr.warmup_ms:.1f}ms")

    try:
        affinity.set_affinity(GENERATOR_CPUS)
    except OSError:
        pass
    gen_initial, gen_post_fix = affinity.verify_and_repin_all_threads(os.getpid(), GENERATOR_CPUS)
    if gen_post_fix:
        print(f"WARNING: unresolved generator affinity violations: {gen_post_fix}")

    queueing_delay_ms = []
    e2e_ms = []
    prep_times_ms = []
    sample_task_for_pickle_bench = None
    for i in range(args.n_trials):
        idx = i % n_pool
        prep_start = time.monotonic()
        input_ids = input_ids_all[idx : idx + 1]
        token_type_ids = token_type_ids_all[idx : idx + 1]
        attention_mask = attention_mask_all[idx : idx + 1]
        prep_times_ms.append((time.monotonic() - prep_start) * 1000.0)

        enqueue_ts = time.monotonic()  # stamped right before put_nowait, after all prep above
        task = Task(
            request_id=i,
            scheduled_ts=enqueue_ts,
            enqueue_ts=enqueue_ts,
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        if sample_task_for_pickle_bench is None:
            sample_task_for_pickle_bench = task
        task_queue.put_nowait(task)
        result = result_queue.get(timeout=30.0)
        queueing_delay_ms.append((result.execution_start - result.enqueue_ts) * 1000.0)
        e2e_ms.append((result.completion_ts - result.scheduled_ts) * 1000.0)
        time.sleep(args.gap_s)

    task_queue.put(None)
    p.join(timeout=10.0)
    if p.is_alive():
        p.terminate()

    pickle_roundtrip = measure_pickle_roundtrip_ms(sample_task_for_pickle_bench)

    report = {
        "config": args.config,
        "n_trials": args.n_trials,
        "gap_s": args.gap_s,
        "generator_affinity_post_fix_violations": gen_post_fix,
        "input_prep_ms": summarize(prep_times_ms),
        "queueing_delay_ms": {
            **summarize(queueing_delay_ms),
            "label": "IPC + queue wake-up overhead, NOT 'pure IPC' -- see decomposition below",
        },
        "pickle_roundtrip_only_ms": {
            **pickle_roundtrip,
            "label": "in-process pickle.dumps+loads of one Task object, no queue/pipe/second-process involved -- a lower bound on the serialization component alone",
        },
        "e2e_ms_for_reference": summarize(e2e_ms),
        "method": (
            "single worker, requests sent one-at-a-time with a gap exceeding "
            "service time so genuine queueing wait is ~0. queueing_delay_ms "
            "(enqueue_ts to execution_start) bundles pickling + pipe transfer "
            "+ OS wake-up scheduling latency for the blocked worker + "
            "unpickling -- these are not separable from that single number "
            "alone. pickle_roundtrip_only_ms isolates just the "
            "serialization component via an in-process microbenchmark on an "
            "identical Task object, as a partial decomposition."
        ),
    }
    report_path = out_dir / "ipc_floor.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
