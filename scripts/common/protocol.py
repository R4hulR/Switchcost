"""Shared message types and config constants for the smoke/load-test harness.

CONFIGS mirrors docs/pilot_config_matrix.md §4 exactly (same CPU budget,
same four worker/thread factorizations). GENERATOR_CPUS mirrors §3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    # Deliberately NOT imported at runtime: worker.py must set BLAS/thread
    # env vars (common/env_setup.py) before numpy is imported anywhere in
    # the process, and this module is imported early. `from __future__
    # import annotations` above makes these annotations strings, so no
    # runtime numpy import is needed just to define these types.
    import numpy as np

SERVER_CPUS = list(range(0, 8))
GENERATOR_CPUS = list(range(8, 12))

CONFIGS: dict[str, dict] = {
    "W1T8": {"workers": 1, "threads": 8, "cpu_groups": [[0, 1, 2, 3, 4, 5, 6, 7]]},
    "W2T4": {"workers": 2, "threads": 4, "cpu_groups": [[0, 1, 2, 3], [4, 5, 6, 7]]},
    "W4T2": {"workers": 4, "threads": 2, "cpu_groups": [[0, 1], [2, 3], [4, 5], [6, 7]]},
    "W8T1": {"workers": 8, "threads": 1, "cpu_groups": [[i] for i in range(8)]},
}


# Shared-memory status codes (docs/pilot_config_matrix.md §11 / research_log
# 2026-09-17 review, round 2): written directly to a multiprocessing.Array
# by the worker BEFORE it puts anything on result_queue, so a request's
# true progress can be recovered even if the worker is killed before its
# Result message makes it through the (possibly corrupted-by-SIGTERM)
# queue pipe.
#
# "Data before flag" discipline (round 2 fix): for every status transition
# below, the corresponding timestamp array entry is written BEFORE the
# status flag itself, never after. An earlier version wrote
# STATUS_COMPLETED before shared_completion[...] -- a kill between those
# two lines would have left a request reading as "confirmed COMPLETED"
# while its recovered completion_ts was still the uninitialized sentinel.
# Reconciliation additionally validates recovered timestamps (positive,
# monotonically ordered) before trusting them; validation failure downgrades
# the outcome to UNKNOWN rather than reporting a bogus latency.
#
# STATUS_NOT_STARTED residual ambiguity (round 2, stated honestly rather
# than overclaimed): this is the array's default value, so "no
# STATUS_DEQUEUED was observed" is the best available evidence a request
# was never dequeued -- but it is not airtight proof. There is an
# irreducible (though narrow -- on the order of one Python bytecode gap)
# race between task_queue.get() returning and the STATUS_DEQUEUED write:
# a SIGTERM landing in that exact window would leave a request that WAS
# actually dequeued still reading as STATUS_NOT_STARTED. Reconciliation
# reports this outcome as "UNFINISHED (not observed as dequeued)", not
# "confirmed never dequeued", per the 2026-09-17 review.
STATUS_NOT_STARTED = 0
STATUS_DEQUEUED = 1  # removed from the queue; final outcome not yet decided
STATUS_TIMED_OUT = 2  # worker decided TIMED_OUT locally (queue-wait check); exec_start valid
STATUS_COMPLETED = 3  # inference + pooling finished locally; completion_ts valid


class Task(NamedTuple):
    request_id: int
    scheduled_ts: float
    enqueue_ts: float  # stamped by generator right before put_nowait; see §9/§11
    input_ids: np.ndarray
    token_type_ids: np.ndarray
    attention_mask: np.ndarray


class Result(NamedTuple):
    request_id: int
    scheduled_ts: float
    enqueue_ts: float
    execution_start: float
    completion_ts: float | None
    worker_id: int
    outcome: str  # "COMPLETED" | "TIMED_OUT"


class WorkerReady(NamedTuple):
    worker_id: int
    pid: int
    initial_affinity_violations: list[dict]
    post_fix_affinity_violations: list[dict]
    warmup_ms: float
    resolved_settings: dict  # actual SessionOptions applied -- see worker.py
