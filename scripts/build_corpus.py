"""Build the pilot sentence corpus and split it into tuning / held-out
evaluation subsets (docs/pilot_config_matrix.md §4, §5a).

This is a small, fixed, reproducible corpus (not downloaded) so the pilot
has no external-dataset dependency at this stage; length is varied on
purpose since sequence length affects compute and we want the pretokenized
fixed-length inputs (scripts/pretokenize.py) to be chosen against a
representative distribution rather than a single short example.

Usage: python scripts/build_corpus.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parent.parent / "models" / "corpus.json"
SEED = 20260917  # fixed for reproducibility; today's date as a mnemonic, not "random"

SUBJECTS = [
    "The system", "A distributed cache", "The scheduler", "Our inference server",
    "The load generator", "A worker process", "The queue", "This benchmark",
    "The embedding model", "A single request", "The research team", "Every worker",
]
VERBS = [
    "processes", "measures", "schedules", "records", "drains", "rejects",
    "completes", "queues", "pins", "verifies", "restarts", "warms up",
]
OBJECTS = [
    "the incoming traffic", "a batch of requests", "the tail latency",
    "its CPU affinity", "the arrival schedule", "memory usage",
    "the transition cost", "a fixed configuration", "the deadline",
    "the raw per-request log", "the correctness check", "its thread pool",
]
CLAUSES = [
    "", " under sustained load", " before the deadline expires",
    ", which is recorded in the raw log", " without dropping any requests",
    ", even during a transition", " across repeated runs",
    " while background load fluctuates", " once the queue is drained",
    " according to the frozen arrival schedule",
]


def generate_sentences(n: int, rng: random.Random) -> list[str]:
    sentences = []
    seen = set()
    while len(sentences) < n:
        s = (
            f"{rng.choice(SUBJECTS)} {rng.choice(VERBS)} {rng.choice(OBJECTS)}"
            f"{rng.choice(CLAUSES)}."
        )
        if s not in seen:
            seen.add(s)
            sentences.append(s)
    return sentences


def main() -> None:
    rng = random.Random(SEED)
    n_total = 240
    sentences = generate_sentences(n_total, rng)

    ids = list(range(len(sentences)))
    rng.shuffle(ids)
    split_point = len(ids) // 2
    tuning_ids = set(ids[:split_point])

    records = [
        {
            "id": i,
            "text": s,
            "split": "tuning" if i in tuning_ids else "eval",
            "n_words": len(s.split()),
        }
        for i, s in enumerate(sentences)
    ]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"seed": SEED, "records": records}, indent=2))

    n_tuning = sum(1 for r in records if r["split"] == "tuning")
    n_eval = len(records) - n_tuning
    word_counts = sorted(r["n_words"] for r in records)
    print(f"Wrote {len(records)} sentences to {OUT_PATH}")
    print(f"tuning={n_tuning} eval={n_eval}")
    print(
        f"word count: min={word_counts[0]} median={word_counts[len(word_counts)//2]} "
        f"max={word_counts[-1]}"
    )


if __name__ == "__main__":
    main()
