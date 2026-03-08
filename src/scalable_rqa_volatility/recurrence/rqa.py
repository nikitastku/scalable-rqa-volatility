from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from scalable_rqa_volatility.recurrence.embeddings import EmbeddingConfig


Mode = Literal["per_series", "joint"]


@dataclass(frozen=True)
class RQAConfig:
    window: int = 90
    step: int = 5
    embed: EmbeddingConfig = EmbeddingConfig(m=4, tau=2)
    recurrence_rate: float = 0.1
    lmin: int = 2
    vmin: int = 2
    exclude_diagonal: bool = True
    standardize: bool = True
    eps_max_points: int = 500
    mode: Mode = "per_series"


def _pairwise_distances(X: np.ndarray) -> np.ndarray:
    G = X @ X.T
    sq = np.sum(X * X, axis=1, keepdims=True)
    D2 = np.maximum(sq - 2.0 * G + sq.T, 0.0)
    return np.sqrt(D2, dtype=float)


def _epsilon_for_rr(D: np.ndarray, rr: float, exclude_diagonal: bool) -> float:
    if not (0.0 < rr < 1.0):
        raise ValueError("recurrence_rate must be in (0,1)")
    if exclude_diagonal:
        mask = ~np.eye(D.shape[0], dtype=bool)
        vals = D[mask]
    else:
        vals = D.reshape(-1)
    return float(np.nanquantile(vals, rr))


def _recurrence_matrix(D: np.ndarray, eps: float, exclude_diagonal: bool) -> np.ndarray:
    R = (D <= eps)
    if exclude_diagonal:
        np.fill_diagonal(R, False)
    return R


def _run_lengths_1d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=bool).reshape(-1)
    if a.size == 0:
        return np.array([], dtype=int)
    diff = np.diff(a.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if a[0]:
        starts = np.r_[0, starts]
    if a[-1]:
        ends = np.r_[ends, a.size]
    return (ends - starts).astype(int)


def _diag_line_lengths(R: np.ndarray) -> np.ndarray:
    n = R.shape[0]
    lengths: list[int] = []
    for k in range(-(n - 1), n):
        d = np.diagonal(R, offset=k)
        lens = _run_lengths_1d(d)
        if lens.size:
            lengths.extend(lens.tolist())
    return np.asarray(lengths, dtype=int)


def _vert_line_lengths(R: np.ndarray) -> np.ndarray:
    lengths: list[int] = []
    for j in range(R.shape[1]):
        lens = _run_lengths_1d(R[:, j])
        if lens.size:
            lengths.extend(lens.tolist())
    return np.asarray(lengths, dtype=int)


def _standardize_cols(W: np.ndarray) -> np.ndarray:
    mu = np.nanmean(W, axis=0, keepdims=True)
    sd = np.nanstd(W, axis=0, keepdims=True)
    sd = np.where(~np.isfinite(sd) | (sd == 0.0), 1.0, sd)
    return (W - mu) / sd


def _delay_embed_1d(x: np.ndarray, cfg: EmbeddingConfig) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.shape[0]
    span = (cfg.m - 1) * cfg.tau
    if n <= span:
        raise ValueError("Series too short for delay embedding.")
    rows = n - span
    out = np.empty((rows, cfg.m), dtype=float)
    for j in range(cfg.m):
        out[:, j] = x[j * cfg.tau : j * cfg.tau + rows]
    return out


def _delay_embed_joint(W: np.ndarray, cfg: EmbeddingConfig) -> np.ndarray:
    W = np.asarray(W, dtype=float)
    n, d = W.shape
    span = (cfg.m - 1) * cfg.tau
    if n <= span:
        raise ValueError("Series too short for delay embedding.")
    rows = n - span
    out = np.empty((rows, cfg.m * d), dtype=float)
    col = 0
    for j in range(cfg.m):
        block = W[j * cfg.tau : j * cfg.tau + rows, :]
        out[:, col : col + d] = block
        col += d
    return out


def estimate_epsilon_from_train(train_block: np.ndarray, cfg: RQAConfig) -> float:
    W = np.asarray(train_block, dtype=float)
    if W.ndim == 1:
        W = W.reshape(-1, 1)
    if cfg.standardize:
        W = _standardize_cols(W)
    X = _delay_embed_joint(W, cfg.embed) if cfg.mode == "joint" else _delay_embed_1d(W[:, 0], cfg.embed)
    n = X.shape[0]
    k = int(min(max(cfg.eps_max_points, 10), n))
    if k < n:
        idx = np.linspace(0, n - 1, k).astype(int)
        X = X[idx]
    D = _pairwise_distances(X)
    return _epsilon_for_rr(D, cfg.recurrence_rate, cfg.exclude_diagonal)


def _rqa_from_embedded(X: np.ndarray, cfg: RQAConfig, eps_fixed: float | None) -> dict[str, float]:
    D = _pairwise_distances(X)
    eps = float(eps_fixed) if eps_fixed is not None else _epsilon_for_rr(D, cfg.recurrence_rate, cfg.exclude_diagonal)
    R = _recurrence_matrix(D, eps, cfg.exclude_diagonal)

    n = R.shape[0]
    total_pairs = n * (n - 1) if cfg.exclude_diagonal else n * n
    Rsum = int(R.sum())
    rr = float(Rsum / max(total_pairs, 1))

    diag = _diag_line_lengths(R)
    diag_ge = diag[diag >= cfg.lmin]
    det = float(diag_ge.sum() / max(Rsum, 1))
    lmax = float(diag_ge.max()) if diag_ge.size else 0.0

    vert = _vert_line_lengths(R)
    vert_ge = vert[vert >= cfg.vmin]
    lam = float(vert_ge.sum() / max(Rsum, 1))
    tt = float(vert_ge.mean()) if vert_ge.size else 0.0

    entr = 0.0
    if diag_ge.size:
        counts = np.bincount(diag_ge)
        p = counts[counts > 0].astype(float)
        p /= p.sum()
        entr = float(-(p * np.log(p)).sum())

    return {"rr": rr, "det": det, "lam": lam, "lmax": lmax, "tt": tt, "entr": entr}


def rqa_features_rolling(
    df: pd.DataFrame,
    cols: Iterable[str],
    cfg: RQAConfig,
    prefix: str = "rqa",
    eps_fixed: float | None = None,
) -> pd.DataFrame:
    cols = list(cols)
    if cfg.window < 10:
        raise ValueError("window too small for RQA.")
    if cfg.step < 1:
        raise ValueError("step must be >= 1")

    n = len(df)
    feature_names = ["rr", "det", "lam", "lmax", "tt", "entr"]

    if cfg.mode == "joint":
        out_cols = [f"{prefix}_joint_{f}" for f in feature_names]
        out_np = np.full((n, len(out_cols)), np.nan, dtype=float)
        arr = df[cols].to_numpy(dtype=float, copy=False)

        start_t = cfg.window - 1
        for t in range(start_t, n, cfg.step):
            w0 = t - cfg.window + 1
            w1 = t + 1
            W = arr[w0:w1, :]
            if cfg.standardize:
                W = _standardize_cols(W)
            X = _delay_embed_joint(W, cfg.embed)
            feats = _rqa_from_embedded(X, cfg, eps_fixed)
            for i, f in enumerate(feature_names):
                out_np[t, i] = float(feats[f])

        out = pd.DataFrame(out_np, index=df.index, columns=out_cols)
        return out.ffill()

    out_cols = [f"{prefix}_{c}_{f}" for c in cols for f in feature_names]
    out_np = np.full((n, len(out_cols)), np.nan, dtype=float)
    arr = df[cols].to_numpy(dtype=float, copy=False)

    start_t = cfg.window - 1
    for t in range(start_t, n, cfg.step):
        w0 = t - cfg.window + 1
        w1 = t + 1
        base = 0
        for ci in range(len(cols)):
            x = arr[w0:w1, ci]
            if cfg.standardize:
                x = _standardize_cols(x.reshape(-1, 1)).reshape(-1)
            X = _delay_embed_1d(x, cfg.embed)
            feats = _rqa_from_embedded(X, cfg, eps_fixed)
            for fi, f in enumerate(feature_names):
                out_np[t, base + fi] = float(feats[f])
            base += len(feature_names)

    out = pd.DataFrame(out_np, index=df.index, columns=out_cols)
    return out.ffill()