# src/scalable_rqa_volatility/recurrence/beta_rqa.py
#
# IMPROVED: adds horizontal line measures (LAM_h, TT_h) and asymmetry
# measures (ΔLAM, ΔTT) from Dreesen/Deckert/Marwan/Boussé 2025.
# For β ≠ 2, β-RPs are asymmetric → horizontal ≠ vertical line structures.
# This asymmetry carries directional information about the dynamics.

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from scalable_rqa_volatility.recurrence.embeddings import EmbeddingConfig, delay_embed
from scalable_rqa_volatility.recurrence.rqa import (
    _pairwise_distances,
    _epsilon_for_rr,
    _recurrence_matrix,
    _diag_line_lengths,
    _vert_line_lengths,
)


@dataclass(frozen=True)
class BetaRQAConfig:
    window: int = 90
    step: int = 5
    embed: EmbeddingConfig = EmbeddingConfig(m=4, tau=2)
    recurrence_rate: float = 0.1
    lmin: int = 2
    vmin: int = 2
    exclude_diagonal: bool = True
    beta: float = 1.0
    eps: float = 1e-10
    transform: str = "minmax"


def _minmax_pos(x: np.ndarray, eps: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mn = float(np.nanmin(x))
    mx = float(np.nanmax(x))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.full_like(x, 1.0, dtype=float)
    y = (x - mn) / (mx - mn)
    return y * (1.0 - eps) + eps


def _pairwise_beta_distances(X: np.ndarray, beta: float, eps: float) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    X = np.maximum(X, eps)
    b = float(beta)
    n = X.shape[0]

    if np.isclose(b, 2.0):
        G = X @ X.T
        sq = np.sum(X * X, axis=1, keepdims=True)
        D2 = np.maximum(sq - 2.0 * G + sq.T, 0.0)
        return 0.5 * D2

    if np.isclose(b, 1.0):
        logX = np.log(X)
        a = np.sum(X * logX, axis=1)
        s = np.sum(X, axis=1)
        B = X @ logX.T
        D = a[:, None] - B - s[:, None] + s[None, :]
        np.fill_diagonal(D, 0.0)
        return D

    if np.isclose(b, 0.0):
        R = X[:, None, :] / X[None, :, :]
        D = np.sum(R - np.log(R) - 1.0, axis=2)
        np.fill_diagonal(D, 0.0)
        return D

    Xb = X ** b
    Yb = Xb[None, :, :]
    Xb2 = Xb[:, None, :]
    Yb_1 = X ** (b - 1.0)
    term = Xb2 + (b - 1.0) * Yb - b * (X[:, None, :] * Yb_1[None, :, :])
    c = 1.0 / (b * (b - 1.0))
    D = c * np.sum(term, axis=2)
    np.fill_diagonal(D, 0.0)
    return D


def _horiz_line_lengths(R: np.ndarray) -> np.ndarray:
    """
    NEW: horizontal line lengths from rows of the recurrence matrix.
    For symmetric RPs (β=2), these equal vertical line lengths.
    For asymmetric β-RPs (β≠2), they capture different structure.
    From Dreesen et al. 2025, Section IV.B.
    """
    lengths: list[int] = []
    for i in range(R.shape[0]):
        row = R[i, :]
        # run-length encoding on this row
        if row.size == 0:
            continue
        diff = np.diff(row.astype(np.int8))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        if row[0]:
            starts = np.r_[0, starts]
        if row[-1]:
            ends = np.r_[ends, row.size]
        lens = (ends - starts).astype(int)
        if lens.size:
            lengths.extend(lens.tolist())
    return np.asarray(lengths, dtype=int)


def beta_rqa_features_from_series(x: np.ndarray, cfg: BetaRQAConfig) -> dict[str, float]:
    x = np.asarray(x, dtype=float).reshape(-1)

    if cfg.transform == "minmax":
        x = _minmax_pos(x, eps=cfg.eps)
    elif cfg.transform != "none":
        raise ValueError("transform must be 'minmax' or 'none'")

    X = delay_embed(x, cfg.embed)

    # β=2 => Euclidean (symmetric), use fast path
    if np.isclose(cfg.beta, 2.0):
        D = _pairwise_distances(X)
    else:
        D = _pairwise_beta_distances(X, beta=cfg.beta, eps=cfg.eps)

    eps = _epsilon_for_rr(D, cfg.recurrence_rate, cfg.exclude_diagonal)
    R = _recurrence_matrix(D, eps, cfg.exclude_diagonal)

    n = R.shape[0]
    total_pairs = n * (n - 1) if cfg.exclude_diagonal else n * n
    Rsum = int(R.sum())
    rr = float(Rsum / max(total_pairs, 1))

    # --- diagonal lines ---
    diag = _diag_line_lengths(R)
    diag_ge = diag[diag >= cfg.lmin]
    det = float(diag_ge.sum() / max(Rsum, 1))
    lmax = float(diag_ge.max()) if diag_ge.size else 0.0

    # --- vertical lines ---
    vert = _vert_line_lengths(R)
    vert_ge = vert[vert >= cfg.vmin]
    lam = float(vert_ge.sum() / max(Rsum, 1))
    tt = float(vert_ge.mean()) if vert_ge.size else 0.0

    # --- NEW: horizontal lines (Dreesen 2025) ---
    horiz = _horiz_line_lengths(R)
    horiz_ge = horiz[horiz >= cfg.vmin]
    lam_h = float(horiz_ge.sum() / max(Rsum, 1))
    tt_h = float(horiz_ge.mean()) if horiz_ge.size else 0.0

    # --- NEW: asymmetry measures (Dreesen 2025, Table I) ---
    delta_lam = abs(lam - lam_h)
    delta_tt = abs(tt - tt_h)

    # --- entropy ---
    entr = 0.0
    if diag_ge.size:
        counts = np.bincount(diag_ge)
        p = counts[counts > 0].astype(float)
        p = p / p.sum()
        entr = float(-(p * np.log(p)).sum())

    return {
        "rr": rr, "det": det, "lam": lam, "lmax": lmax, "tt": tt, "entr": entr,
        "lam_h": lam_h, "tt_h": tt_h, "delta_lam": delta_lam, "delta_tt": delta_tt,
    }


def beta_rqa_features_rolling(
    df: pd.DataFrame,
    cols: Iterable[str],
    cfg: BetaRQAConfig,
    prefix: str = "beta_rqa",
) -> pd.DataFrame:
    cols = list(cols)
    if cfg.window < 10:
        raise ValueError("window too small for beta-RQA.")
    if cfg.step < 1:
        raise ValueError("step must be >= 1")

    n = len(df)
    # IMPROVED: 10 features instead of 6
    feature_names = ["rr", "det", "lam", "lmax", "tt", "entr", "lam_h", "tt_h", "delta_lam", "delta_tt"]
    out_cols = [f"{prefix}_{c}_{f}" for c in cols for f in feature_names]
    out_np = np.full((n, len(out_cols)), np.nan, dtype=float)

    arr = df[cols].to_numpy(dtype=float, copy=False)

    start_t = cfg.window - 1
    for t in range(start_t, n, cfg.step):
        w0 = t - cfg.window + 1
        w1 = t + 1
        base = 0
        for ci, _ in enumerate(cols):
            x = arr[w0:w1, ci]
            feats = beta_rqa_features_from_series(x, cfg)
            for fi, f in enumerate(feature_names):
                out_np[t, base + fi] = float(feats[f])
            base += len(feature_names)

    out = pd.DataFrame(out_np, index=df.index, columns=out_cols)
    return out.ffill()