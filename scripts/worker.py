"""Worker process entrypoint. Run only via multiprocessing (spawn context),
never imported for its numpy/onnxruntime symbols directly -- see the
import-order comment below.

Order of operations matters and is enforced here (docs/pilot_config_matrix.md
§8, §11):
  1. Pin single-threaded BLAS/tokenizer env vars BEFORE numpy/onnxruntime
     are imported anywhere in this process (those libraries read the env
     vars once, at load time).
  2. Set this process's CPU affinity BEFORE the ONNX Runtime session (and
     its intra-op thread pool) is constructed.
  3. After session construction, verify EVERY thread's actual affinity --
     don't assume setting it on the parent process alone was sufficient.
  4. Warm up (a few dummy inferences) and report readiness BEFORE the timed
     run starts, so warmup cost is excluded from measured results.

Clock convention: ALL timestamps in this process use time.monotonic(), the
same call the generator (scripts/smoke_bench.py) uses -- not
time.perf_counter(). An earlier version mixed the two; on this CPython/
Linux build they happen to share the same underlying clock source
(clock_gettime(CLOCK_MONOTONIC)) so the earlier numbers weren't actually
wrong, but relying on that coincidence instead of stating one convention
explicitly was exactly the kind of thing the 2026-09-17 review flagged
elsewhere (percentile convention) -- fixed here too, not just where flagged.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import env_setup  # noqa: E402

env_setup.pin_single_threaded_blas()  # MUST precede the imports below

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

from common import affinity  # noqa: E402
from common.pooling import l2_normalize, masked_mean_pool  # noqa: E402
from common.protocol import (  # noqa: E402
    STATUS_COMPLETED,
    STATUS_DEQUEUED,
    STATUS_TIMED_OUT,
    Result,
    Task,
    WorkerReady,
)

MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "all-MiniLM-L6-v2" / "model.onnx")


def worker_main(
    worker_id: int,
    cpus: list[int],
    threads: int,
    max_queue_wait_s: float,
    task_queue,
    result_queue,
    status_queue,
    shared_status,  # multiprocessing.Array('B', n_requests), lock=False
    shared_exec_start,  # multiprocessing.Array('d', n_requests), lock=False
    shared_completion,  # multiprocessing.Array('d', n_requests), lock=False
) -> None:
    affinity.set_affinity(cpus)  # before session construction, per module docstring

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = threads
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.add_session_config_entry("session.intra_op.allow_spinning", "1")
    sess_options.add_session_config_entry("session.inter_op.allow_spinning", "1")

    session = ort.InferenceSession(
        MODEL_PATH, sess_options=sess_options, providers=["CPUExecutionProvider"]
    )
    resolved_settings = {
        "intra_op_num_threads": threads,
        "inter_op_num_threads": 1,
        "execution_mode": "ORT_SEQUENTIAL",
        "intra_op_allow_spinning": "1",
        "inter_op_allow_spinning": "1",
        "providers": session.get_providers(),
    }

    # Observed cause (docs/research_log.md, 2026-09-17): under
    # multiprocessing spawn, at least one background thread (likely part of
    # the spawn bootstrap itself) is created before worker_main runs and
    # retains the process's original, unrestricted affinity -- it is not an
    # ONNX Runtime thread pool worker (verified separately: ORT threads
    # correctly inherit restricted affinity outside of multiprocessing).
    initial_violations, post_fix_violations = affinity.verify_and_repin_all_threads(os.getpid(), cpus)

    dummy_ids = np.zeros((1, 32), dtype=np.int64)
    dummy_mask = np.ones((1, 32), dtype=np.int64)
    t_warmup_start = time.monotonic()
    for _ in range(8):
        (lhs,) = session.run(
            ["last_hidden_state"],
            {"input_ids": dummy_ids, "token_type_ids": dummy_ids, "attention_mask": dummy_mask},
        )
        l2_normalize(masked_mean_pool(lhs, dummy_mask))
    warmup_ms = (time.monotonic() - t_warmup_start) * 1000.0

    status_queue.put(
        WorkerReady(
            worker_id, os.getpid(), initial_violations, post_fix_violations, warmup_ms, resolved_settings
        )
    )

    while True:
        item = task_queue.get()
        if item is None:  # sentinel: drain complete, exit
            break
        task: Task = item
        execution_start = time.monotonic()

        # "Data before flag" (2026-09-17 review, round 2): write the
        # timestamp array entry BEFORE the status flag that claims it's
        # valid. An earlier version wrote the flag first -- a kill between
        # the two lines would have left a status claiming a valid timestamp
        # that was actually still the uninitialized sentinel. Every
        # transition below follows this order; reconciliation additionally
        # validates recovered values rather than trusting the flag alone.
        shared_exec_start[task.request_id] = execution_start
        shared_status[task.request_id] = STATUS_DEQUEUED

        # Queue timeout, corrected (2026-09-17 review): this must measure
        # time actually spent WAITING IN THE QUEUE (execution_start minus
        # enqueue_ts), not time since scheduled arrival (which also
        # includes generator scheduling delay -- a different, separately-
        # tracked bucket, see docs/pilot_config_matrix.md §9). The original
        # code compared against task.scheduled_ts, which silently included
        # generator delay in what was meant to be a pure queue-wait bound.
        if execution_start - task.enqueue_ts > max_queue_wait_s:
            shared_status[task.request_id] = STATUS_TIMED_OUT  # exec_start already valid, written above
            result_queue.put(
                Result(
                    task.request_id,
                    task.scheduled_ts,
                    task.enqueue_ts,
                    execution_start,
                    None,
                    worker_id,
                    "TIMED_OUT",
                )
            )
            continue

        (last_hidden_state,) = session.run(
            ["last_hidden_state"],
            {
                "input_ids": task.input_ids,
                "token_type_ids": task.token_type_ids,
                "attention_mask": task.attention_mask,
            },
        )
        pooled = masked_mean_pool(last_hidden_state, task.attention_mask)
        l2_normalize(pooled)  # computed to match real service cost; result itself unused downstream
        completion_ts = time.monotonic()  # stamped BEFORE handing off to result_queue -- see §9

        # Written to shared memory BEFORE the (queue-pipe-dependent) result
        # is enqueued, so a forced SIGTERM right after this line still
        # leaves a recoverable, confirmed completion record -- see
        # docs/pilot_config_matrix.md §11 and research_log.md 2026-09-17.
        # Data (completion_ts) before flag (STATUS_COMPLETED), same
        # discipline as above.
        shared_completion[task.request_id] = completion_ts
        shared_status[task.request_id] = STATUS_COMPLETED

        result_queue.put(
            Result(
                task.request_id,
                task.scheduled_ts,
                task.enqueue_ts,
                execution_start,
                completion_ts,
                worker_id,
                "COMPLETED",
            )
        )
