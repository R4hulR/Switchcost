"""Correctness gate (docs/pilot_config_matrix.md §10): verify the full
ONNX Runtime path -- backbone inference + our own masked mean pooling +
L2 normalization -- against the sentence-transformers reference, on a
fixed test set. Fails loudly (non-zero exit) if any sentence is outside
tolerance. Must pass before any benchmarking is trusted.

2026-09-17 review (round 2): added a second pass using representative
benchmark-corpus inputs (not just the hand-built edge-case sentences)
run across every intra_op_num_threads setting actually used by the four
pilot configs (1, 2, 4, 8) -- floating-point reduction order in a
parallel matmul can differ by thread count, so passing at threads=1
alone doesn't guarantee correctness at the thread counts benchmarks
actually run with. Tolerance is unchanged (not the priority right now,
per review); this only widens what's checked.

Usage: python scripts/verify_correctness.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import env_setup  # noqa: E402

env_setup.pin_single_threaded_blas()  # must precede numpy/onnxruntime import

import numpy as np  # noqa: E402
import onnxruntime  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from common.pooling import l2_normalize, masked_mean_pool  # noqa: E402
from common.protocol import CONFIGS  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "all-MiniLM-L6-v2"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CORPUS_PATH = Path(__file__).resolve().parent.parent / "models" / "corpus.json"

# Tolerance justified for fp32-ONNX-export vs. fp32-PyTorch numerical
# differences (op-level implementation / export fidelity noise), NOT a
# quantization-level tolerance -- both paths run at full fp32 precision.
COSINE_SIM_MIN = 0.9999
MAX_ABS_DIFF_MAX = 1e-3

# Correctness test sentences: drawn from the TUNING pool, not the held-out
# evaluation corpus (docs/pilot_config_matrix.md §4/§5a) -- this set exists
# to gate correctness, not to report benchmark numbers, so it's fine (and
# appropriate) for it to be separate from and smaller than the benchmark corpus.
# Deliberately varied lengths (short fragment -> long multi-clause sentence)
# since pooling bugs (e.g. an off-by-one in the attention mask) tend to show
# up at the padding/truncation boundary.
TEST_SENTENCES = [
    "Hello world.",
    "The quick brown fox jumps over the lazy dog.",
    "ONNX Runtime is a cross-platform inference engine.",
    "",
    "A",
    "This is a moderately long sentence meant to exercise the middle of the "
    "sequence length range without hitting the truncation boundary.",
    "In distributed systems, latency percentiles such as p99 often matter more "
    "than the mean, because tail latency directly affects the worst-case "
    "experience of a meaningful fraction of users, especially under load.",
    "Cats sleep a lot.",
    "1234567890 !@#$%^&*()",
    "Multiple    spaces   and\ttabs\nand newlines.",
    "Sentence transformers map sentences to a fixed-size dense vector space "
    "that can be used for clustering or semantic search, and this particular "
    "sentence is deliberately long enough to approach or exceed a short "
    "fixed sequence length so truncation behavior is exercised as well.",
]

THREAD_COUNTS_TO_CHECK = sorted({cfg["threads"] for cfg in CONFIGS.values()})  # [1, 2, 4, 8]
N_CORPUS_SAMPLES = 20


def load_model_card() -> dict:
    return json.loads((MODEL_DIR / "model_card.json").read_text())


def load_corpus_sample(n: int) -> list[str]:
    corpus = json.loads(CORPUS_PATH.read_text())
    # Representative benchmark inputs (2026-09-17 review): drawn from BOTH
    # splits, deterministically, so this check exercises the actual
    # sentence pool the calibration/sweep runs use, not just hand-built
    # edge cases.
    records = corpus["records"]
    step = max(1, len(records) // n)
    return [r["text"] for r in records[::step]][:n]


def check_sentences(
    sess: onnxruntime.InferenceSession,
    tokenizer,
    sentences: list[str],
    ref_embeddings: np.ndarray,
) -> tuple[int, list[dict]]:
    encoded = tokenizer(sentences, padding=True, truncation=True, return_tensors="np")
    ort_inputs = {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "token_type_ids": encoded["token_type_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64),
    }
    (last_hidden_state,) = sess.run(["last_hidden_state"], ort_inputs)
    pooled = masked_mean_pool(last_hidden_state, encoded["attention_mask"])
    ort_embeddings = l2_normalize(pooled)

    max_abs_diff = np.abs(ort_embeddings - ref_embeddings).max(axis=1)
    cosine_sim = np.sum(ort_embeddings * ref_embeddings, axis=1) / (
        np.linalg.norm(ort_embeddings, axis=1) * np.linalg.norm(ref_embeddings, axis=1)
    )

    n_fail = 0
    per_sentence = []
    for i, sentence in enumerate(sentences):
        ok = bool(cosine_sim[i] >= COSINE_SIM_MIN and max_abs_diff[i] <= MAX_ABS_DIFF_MAX)
        n_fail += 0 if ok else 1
        per_sentence.append(
            {
                "sentence": sentence,
                "cosine_similarity": float(cosine_sim[i]),
                "max_abs_diff": float(max_abs_diff[i]),
                "pass": ok,
            }
        )
    return n_fail, per_sentence


def make_session(intra_op_num_threads: int) -> onnxruntime.InferenceSession:
    sess_options = onnxruntime.SessionOptions()
    sess_options.intra_op_num_threads = intra_op_num_threads
    sess_options.inter_op_num_threads = 1
    return onnxruntime.InferenceSession(
        str(MODEL_DIR / "model.onnx"),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )


def main() -> int:
    card = load_model_card()
    revision = card["revision"]

    print(f"Loading reference sentence-transformers model @ {revision}")
    reference = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", revision=revision
    )
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR / "tokenizer"))

    overall_fail = 0
    report_sections = []

    # --- Part 1: original edge-case sentence set, threads=1 (unchanged) ---
    print("\n=== Edge-case sentences, intra_op_num_threads=1 ===")
    ref_embeddings = reference.encode(TEST_SENTENCES, convert_to_numpy=True, show_progress_bar=False)
    sess = make_session(1)
    n_fail, per_sentence = check_sentences(sess, tokenizer, TEST_SENTENCES, ref_embeddings)
    for row in per_sentence:
        status = "OK  " if row["pass"] else "FAIL"
        print(f"[{status}] cos={row['cosine_similarity']:.6f} max|diff|={row['max_abs_diff']:.2e}  {row['sentence'][:60]!r}")
    overall_fail += n_fail
    report_sections.append({"name": "edge_case_sentences_threads1", "n_fail": n_fail, "per_sentence": per_sentence})

    # --- Part 2: representative benchmark-corpus inputs across every
    # thread count the pilot configs actually use (2026-09-17 review) ---
    corpus_sample = load_corpus_sample(N_CORPUS_SAMPLES)
    ref_corpus_embeddings = reference.encode(corpus_sample, convert_to_numpy=True, show_progress_bar=False)
    for threads in THREAD_COUNTS_TO_CHECK:
        print(f"\n=== Representative corpus sample ({len(corpus_sample)} sentences), intra_op_num_threads={threads} ===")
        sess = make_session(threads)
        n_fail, per_sentence = check_sentences(sess, tokenizer, corpus_sample, ref_corpus_embeddings)
        for row in per_sentence:
            status = "OK  " if row["pass"] else "FAIL"
            print(f"[{status}] cos={row['cosine_similarity']:.6f} max|diff|={row['max_abs_diff']:.2e}  {row['sentence'][:60]!r}")
        overall_fail += n_fail
        report_sections.append(
            {"name": f"corpus_sample_threads{threads}", "n_fail": n_fail, "per_sentence": per_sentence}
        )

    report = {
        "model_revision": revision,
        "tolerance": {"cosine_sim_min": COSINE_SIM_MIN, "max_abs_diff_max": MAX_ABS_DIFF_MAX},
        "thread_counts_checked": THREAD_COUNTS_TO_CHECK,
        "overall_pass": overall_fail == 0,
        "overall_n_fail": overall_fail,
        "sections": report_sections,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "correctness_check.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {report_path}")

    if overall_fail:
        print(f"\nFAIL: {overall_fail} sentence(s) outside tolerance across all sections.")
        return 1

    print(f"\nPASS: all sections within tolerance (edge cases @ threads=1, corpus sample @ threads={THREAD_COUNTS_TO_CHECK}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
