"""Masked mean pooling + L2 normalization, mirroring sentence-transformers'
Pooling/Normalize modules, applied to raw ONNX Runtime `last_hidden_state`
output. Lives outside the traced ONNX graph, which is exactly why
docs/pilot_config_matrix.md §10 requires verifying this path explicitly
against the reference implementation rather than trusting raw ORT output
in isolation.
"""

from __future__ import annotations

import numpy as np


def masked_mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """last_hidden_state: (batch, seq, hidden) fp32
    attention_mask: (batch, seq) int/float, 1 for real tokens, 0 for padding
    Returns (batch, hidden).
    """
    mask = attention_mask.astype(np.float32)[..., None]  # (batch, seq, 1)
    summed = (last_hidden_state * mask).sum(axis=1)  # (batch, hidden)
    counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)  # (batch, 1)
    return summed / counts


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, ord=2, axis=-1, keepdims=True)
    norm = np.clip(norm, a_min=1e-12, a_max=None)
    return x / norm
