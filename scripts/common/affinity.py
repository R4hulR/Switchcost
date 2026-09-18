"""CPU affinity helpers.

Per docs/pilot_config_matrix.md §8: affinity must be set on a worker process
*before* the ONNX Runtime session (and its thread pool) is constructed, and
must be verified afterwards per-thread rather than assumed to have propagated.
"""

from __future__ import annotations

import os


def set_affinity(cpus: list[int]) -> None:
    os.sched_setaffinity(0, set(cpus))


def get_affinity() -> set[int]:
    return os.sched_getaffinity(0)


def verify_all_threads_affinity(pid: int, expected: set[int]) -> list[dict]:
    """Inspect every OS thread under `pid` and report ones whose affinity
    mask doesn't match `expected`. Returns a list of violation dicts
    (empty list = all threads match). Never assumes the parent process's
    affinity setting propagated to threads spawned by native libraries
    (e.g. ONNX Runtime's intra-op thread pool) -- checks directly.
    """
    task_dir = f"/proc/{pid}/task"
    violations = []
    try:
        tids = os.listdir(task_dir)
    except FileNotFoundError:
        return [{"error": f"{task_dir} not found (process gone?)"}]

    for tid in tids:
        status_path = f"{task_dir}/{tid}/status"
        try:
            with open(status_path) as f:
                content = f.read()
        except FileNotFoundError:
            continue  # thread exited between listdir and read
        allowed = None
        for line in content.splitlines():
            if line.startswith("Cpus_allowed_list:"):
                allowed = line.split(":", 1)[1].strip()
                break
        actual_set = _parse_cpu_list(allowed) if allowed else None
        if actual_set != expected:
            comm_path = f"{task_dir}/{tid}/comm"
            try:
                with open(comm_path) as f:
                    comm = f.read().strip()
            except FileNotFoundError:
                comm = "?"
            violations.append(
                {
                    "tid": tid,
                    "comm": comm,
                    "expected": sorted(expected),
                    "actual": sorted(actual_set) if actual_set else None,
                    "raw": allowed,
                }
            )
    return violations


def verify_and_repin_all_threads(pid: int, cpus: list[int]) -> tuple[list[dict], list[dict]]:
    """Verify every thread under `pid` against `cpus`, re-pin any violator
    directly by tid, then re-verify. Returns (initial_violations,
    post_fix_violations). Factored out (2026-09-17 review, round 2) so the
    generator process can apply the same discipline worker.py already
    does -- not just worker processes. A multiprocessing-spawn-related
    thread with unrestricted affinity has been observed in worker
    processes (docs/research_log.md); nothing rules out the analogous
    background threads a generator process starts (queue feeder threads,
    the results-collector thread, the resource-monitor thread) having the
    same problem, so this must be checked there too, not assumed fine.
    """
    expected = set(cpus)
    initial_violations = verify_all_threads_affinity(pid, expected)
    if initial_violations:
        for v in initial_violations:
            try:
                os.sched_setaffinity(int(v["tid"]), expected)
            except (ProcessLookupError, ValueError):
                pass
    post_fix_violations = verify_all_threads_affinity(pid, expected)
    return initial_violations, post_fix_violations


def _parse_cpu_list(s: str) -> set[int]:
    """Parse a Linux CPU list like '0-3,7' into {0,1,2,3,7}."""
    result = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            result.update(range(int(lo), int(hi) + 1))
        else:
            result.add(int(part))
    return result
