from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EmbeddingConfig:
    """Delay embedding configuration for 1D time series."""

    m: int = 3
    tau: int = 1


def delay_embed(x: np.ndarray, cfg: EmbeddingConfig) -> np.ndarray:
    """
    Create delay-embedded vectors for a 1D series.

    x: shape (n,)
    returns: shape (n - (m-1)*tau, m)
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    if cfg.m < 1:
        raise ValueError("m must be >= 1")
    if cfg.tau < 1:
        raise ValueError("tau must be >= 1")

    n = x.shape[0]
    span = (cfg.m - 1) * cfg.tau
    if n <= span:
        raise ValueError("Series too short for delay embedding.")

    rows = n - span
    out = np.empty((rows, cfg.m), dtype=float)
    for j in range(cfg.m):
        out[:, j] = x[j * cfg.tau : j * cfg.tau + rows]
    return out

def time_delay_embedding(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    """
    Backward-compatible wrapper around delay_embed.
    """
    return delay_embed(x, EmbeddingConfig(m=m, tau=tau))