"""Single definition of "percentile" used everywhere in this project.

Review finding (2026-09-17): the original summary code used
sorted_values[int(n*p)] (nearest-rank, biased low), undocumented, and
different from the standard linear-interpolation convention most readers
would assume -- for W2T4 this was a 2.2ms difference at p99 on a 200-row
sample. Fixed by standardizing on linear interpolation between closest
ranks (numpy's default 'linear' method, equivalent to Excel's
PERCENTILE.INC and R's type=7) and centralizing it here so every script
that reports a percentile uses the same convention.
"""

from __future__ import annotations

import numpy as np

PERCENTILE_METHOD = "linear"  # numpy's default; see module docstring


def percentile(values: list[float], p: float) -> float | None:
    """p in [0, 1]. Returns None for an empty input rather than raising,
    since callers routinely compute this over possibly-empty COMPLETED
    subsets."""
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), p * 100.0, method=PERCENTILE_METHOD))


def summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "p50": None, "p95": None, "p99": None, "max": None, "mean": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }
