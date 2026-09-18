"""Time-varying arrival traces: build once, save to disk, replay identically.

Per the 2026-09-17 review: "extend the existing harness to replay a saved
arrival-offset trace" -- a trace's concrete (request_id, offset, phase_id)
sequence is computed once and written to a JSON file, so replaying it
against multiple configs means loading the SAME file, not re-invoking the
RNG per run (which would already be deterministic given the same seed, but
a saved file is auditable and removes any dependency on RNG-implementation
stability across a Python version, which "replay identically" shouldn't
have to trust).
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def build_schedule(rate_hz: float, duration_s: float, seed: int, arrival: str) -> list[float]:
    """Returns scheduled arrival offsets (seconds from phase start), monotonically
    increasing, bounded so the LAST offset is <= duration_s by construction.
    See docs/pilot_config_matrix.md §5/§6a for why duration-bounded (not
    count-bounded) generation is used."""
    rng = random.Random(seed)
    offsets = []
    t = 0.0
    mean_gap = 1.0 / rate_hz
    while True:
        gap = rng.expovariate(1.0 / mean_gap) if arrival == "poisson" else mean_gap
        t += gap
        if t > duration_s:
            break
        offsets.append(t)
    return offsets


def build_phased_schedule(phases: list[dict], seed: int, arrival: str) -> dict:
    """phases: list of {"phase_id": str, "rate_hz": float, "duration_s": float},
    in playback order. Returns a trace dict: {"seed", "arrival", "phases"
    (each annotated with start_offset/end_offset), "requests" (list of
    {"request_id", "offset", "phase_id"}, globally ordered)}.

    Each phase gets its own independent Poisson process (a fresh
    random.Random seeded deterministically from (seed, phase_index), not
    shared RNG state across phases) so phases don't have correlated
    arrival patterns purely as an accident of draw order.
    """
    requests = []
    phase_meta = []
    global_start = 0.0
    request_id = 0
    for i, phase in enumerate(phases):
        phase_seed = seed * 1009 + i  # distinct, deterministic per-phase seed
        local_offsets = build_schedule(phase["rate_hz"], phase["duration_s"], phase_seed, arrival)
        for off in local_offsets:
            requests.append(
                {"request_id": request_id, "offset": global_start + off, "phase_id": phase["phase_id"]}
            )
            request_id += 1
        phase_meta.append(
            {
                "phase_id": phase["phase_id"],
                "rate_hz": phase["rate_hz"],
                "duration_s": phase["duration_s"],
                "seed": phase_seed,
                "start_offset": global_start,
                "end_offset": global_start + phase["duration_s"],
                "n_requests": len(local_offsets),
            }
        )
        global_start += phase["duration_s"]

    return {"seed": seed, "arrival": arrival, "phases": phase_meta, "requests": requests}


def save_trace(trace: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace, indent=2))


def load_trace(path: Path) -> dict:
    return json.loads(Path(path).read_text())
