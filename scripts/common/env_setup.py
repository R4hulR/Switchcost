"""Thread-pool environment control (docs/pilot_config_matrix.md §8).

MUST be called before numpy (or anything that transitively imports numpy,
e.g. onnxruntime) is imported in a process -- BLAS libraries read these
env vars once at load time. Import this module and call pin_single_threaded_blas()
as the very first thing in any worker/benchmark entrypoint.
"""

import os


def pin_single_threaded_blas() -> None:
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = "1"
    # HuggingFace tokenizers' Rust-side thread pool otherwise defaults to
    # spinning up threads sized to visible CPU count.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
