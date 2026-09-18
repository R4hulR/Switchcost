"""Pretokenize the corpus to a fixed sequence length, once, up front.

Per docs/pilot_config_matrix.md §9: tokenization is deliberately excluded
from the per-request timing path for this pilot. The load generator replays
these pretokenized arrays; nothing in the benchmarked path calls the
tokenizer. Sequence length is fixed (not left to vary) per §4, to avoid a
second uncontrolled tuning dimension alongside worker/thread count.

Usage: python scripts/pretokenize.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "all-MiniLM-L6-v2"
CORPUS_PATH = Path(__file__).resolve().parent.parent / "models" / "corpus.json"
OUT_DIR = Path(__file__).resolve().parent.parent / "models" / "pretokenized"

FIXED_SEQ_LEN = 32  # see rationale printed below; matches export trace length


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR / "tokenizer"))
    corpus = json.loads(CORPUS_PATH.read_text())
    records = corpus["records"]

    # Check the natural (unpadded) token-length distribution first, so
    # FIXED_SEQ_LEN's coverage is a checked fact, not an assumption.
    unpadded = tokenizer([r["text"] for r in records], truncation=False)
    lengths = sorted(len(ids) for ids in unpadded["input_ids"])
    n_truncated = sum(1 for l in lengths if l > FIXED_SEQ_LEN)
    print(
        f"Natural token length: min={lengths[0]} median={lengths[len(lengths)//2]} "
        f"p95={lengths[int(len(lengths)*0.95)]} max={lengths[-1]}"
    )
    print(
        f"FIXED_SEQ_LEN={FIXED_SEQ_LEN}: {n_truncated}/{len(lengths)} sentences "
        f"would be truncated"
    )
    if n_truncated:
        raise SystemExit(
            f"FIXED_SEQ_LEN={FIXED_SEQ_LEN} truncates {n_truncated} corpus "
            f"sentences -- raise it or shorten the corpus before proceeding, "
            f"rather than silently truncating benchmark inputs."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("tuning", "eval"):
        split_records = [r for r in records if r["split"] == split]
        encoded = tokenizer(
            [r["text"] for r in split_records],
            padding="max_length",
            truncation=True,
            max_length=FIXED_SEQ_LEN,
            return_tensors="np",
        )
        ids_array = np.array([r["id"] for r in split_records], dtype=np.int64)
        out_path = OUT_DIR / f"{split}.npz"
        np.savez(
            out_path,
            ids=ids_array,
            input_ids=encoded["input_ids"].astype(np.int64),
            token_type_ids=encoded["token_type_ids"].astype(np.int64),
            attention_mask=encoded["attention_mask"].astype(np.int64),
        )
        print(f"Wrote {out_path} ({len(split_records)} examples, seq_len={FIXED_SEQ_LEN})")

    meta = {"fixed_seq_len": FIXED_SEQ_LEN, "corpus_seed": corpus["seed"]}
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
