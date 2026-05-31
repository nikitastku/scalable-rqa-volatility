"""
Implement and benchmark alternative scalable RQA methods.

Implements two algorithms from Marwan (2025):
  1. RQA_woRP: Exact RQA without constructing the recurrence matrix (O(N) memory)
  2. RQA_Samp: Sampled RQA: randomly sample M pairs, trace lines (O(N) memory)

Benchmarks against our current windowed RQA on actual D3 (intraday) data:
  A. Single-stock accuracy comparison (RQA measures vs exact)
  B. Single-stock timing comparison across N values
  C. Windowed pipeline timing: current vs RQA_Samp per window
  D. Full D3 timing comparison (503 stocks)
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scalable_rqa_volatility.logging_utils import get_logger
from scalable_rqa_volatility.recurrence.embeddings import EmbeddingConfig
from scalable_rqa_volatility.recurrence.rqa import (
    RQAConfig,
    _pairwise_distances,
    _epsilon_for_rr,
    _recurrence_matrix,
    _diag_line_lengths,
    _vert_line_lengths,
    _delay_embed_joint,
    _standardize_cols,
    _rqa_from_embedded,
    rqa_features_rolling,
    estimate_epsilon_from_train,
)

COLORS = {
    "primary": "#1E2761", "accent": "#3B82F6", "green": "#10B981",
    "red": "#EF4444", "orange": "#F59E0B", "purple": "#8B5CF6",
    "gray": "#94A3B8",
}
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "legend.fontsize": 9, "figure.figsize": (10, 6),
})


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_split(name: str) -> pd.DataFrame:
    return pd.read_parquet(repo_root() / "data" / "processed" / f"dataset3_{name}.parquet")


def savefig(fig, name: str):
    out_dir = repo_root() / "figures"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def rqa_standard(X: np.ndarray, eps: float, lmin: int = 2, vmin: int = 2) -> dict[str, float]:
    """
    Standard RQA: construct full N×N RP, then compute measures.
    This is our current approach. O(N²) time and memory.
    """
    D = _pairwise_distances(X)
    R = _recurrence_matrix(D, eps, exclude_diagonal=True)

    n = R.shape[0]
    total_pairs = n * (n - 1)
    Rsum = int(R.sum())
    rr = float(Rsum / max(total_pairs, 1))

    diag = _diag_line_lengths(R)
    diag_ge = diag[diag >= lmin]
    det = float(diag_ge.sum() / max(Rsum, 1))
    lmax = float(diag_ge.max()) if diag_ge.size else 0.0

    vert = _vert_line_lengths(R)
    vert_ge = vert[vert >= vmin]
    lam = float(vert_ge.sum() / max(Rsum, 1))
    tt = float(vert_ge.mean()) if vert_ge.size else 0.0

    entr = 0.0
    if diag_ge.size:
        counts = np.bincount(diag_ge)
        p = counts[counts > 0].astype(float)
        p /= p.sum()
        entr = float(-(p * np.log(p)).sum())

    return {"rr": rr, "det": det, "lam": lam, "lmax": lmax, "tt": tt, "entr": entr}


def rqa_worp(X: np.ndarray, eps: float, lmin: int = 2, vmin: int = 2) -> dict[str, float]:
    """
    RQA without Recurrence Plot.

    Computes RQA measures WITHOUT constructing the N×N recurrence matrix.
    Iterates over diagonals and columns, computing distances on-the-fly.

    Memory: O(N), only stores line length histograms.
    Time:   O(N²), still checks all pairs, but avoids matrix allocation.

    This is exact (identical results to standard RQA).
    """
    n = X.shape[0]
    eps2 = eps * eps  
    total_recurrent = 0

    diag_lengths: list[int] = []

    for k in range(1, n):
        for sign in [1, -1]:
            run = 0
            num_pairs = n - k
            for i in range(num_pairs):
                if sign == 1:
                    r, c = i, i + k
                else:
                    r, c = i + k, i

                d2 = 0.0
                for d in range(X.shape[1]):
                    diff = X[r, d] - X[c, d]
                    d2 += diff * diff

                if d2 <= eps2:
                    run += 1
                    total_recurrent += 1
                else:
                    if run > 0:
                        diag_lengths.append(run)
                    run = 0

            if run > 0:
                diag_lengths.append(run)

    diag_arr = np.array(diag_lengths, dtype=int) if diag_lengths else np.array([], dtype=int)

    vert_lengths: list[int] = []

    for j in range(n):
        run = 0
        for i in range(n):
            if i == j:
                continue

            d2 = 0.0
            for d in range(X.shape[1]):
                diff = X[i, d] - X[j, d]
                d2 += diff * diff

            if d2 <= eps2:
                run += 1
            else:
                if run > 0:
                    vert_lengths.append(run)
                run = 0

        if run > 0:
            vert_lengths.append(run)

    vert_arr = np.array(vert_lengths, dtype=int) if vert_lengths else np.array([], dtype=int)

    total_pairs = n * (n - 1)
    rr = float(total_recurrent / max(total_pairs, 1))

    diag_ge = diag_arr[diag_arr >= lmin]
    det = float(diag_ge.sum() / max(total_recurrent, 1))
    lmax = float(diag_ge.max()) if diag_ge.size else 0.0

    vert_ge = vert_arr[vert_arr >= vmin]
    lam = float(vert_ge.sum() / max(total_recurrent, 1))
    tt = float(vert_ge.mean()) if vert_ge.size else 0.0

    entr = 0.0
    if diag_ge.size:
        counts = np.bincount(diag_ge)
        p = counts[counts > 0].astype(float)
        p /= p.sum()
        entr = float(-(p * np.log(p)).sum())

    return {"rr": rr, "det": det, "lam": lam, "lmax": lmax, "tt": tt, "entr": entr}


def rqa_samp(
    X: np.ndarray, eps: float, M: int,
    lmin: int = 2, vmin: int = 2, seed: int = 42,
) -> dict[str, float]:
    """
    Sampled RQA.

    Instead of checking all N² pairs, randomly samples M pairs and traces
    line structures from each recurrent sample.

    For each sampled (i, j) that is recurrent:
      - Trace the diagonal line forward: (i+1,j+1), (i+2,j+2), ...
      - Record the full diagonal line length
      - Trace the vertical line down from (i, j): (i+1,j), (i+2,j), ...
      - Record the vertical line length

    Memory: O(N + M)
    Time:   O(M × L_avg) where L_avg is average line length

    Approximate: converges to exact as M → N².
    """
    n = X.shape[0]
    rng = np.random.RandomState(seed)
    eps2 = eps * eps

    sq_norms = np.sum(X * X, axis=1)

    def is_recurrent(i: int, j: int) -> bool:
        d2 = float(sq_norms[i] + sq_norms[j] - 2.0 * np.dot(X[i], X[j]))
        return d2 <= eps2

    pairs_i = rng.randint(0, n, size=M)
    pairs_j = rng.randint(0, n, size=M)
    same = pairs_i == pairs_j
    while same.any():
        pairs_j[same] = rng.randint(0, n, size=int(same.sum()))
        same = pairs_i == pairs_j

    sampled_recurrent = 0
    diag_lengths: list[int] = []
    vert_lengths: list[int] = []
    sample_diag_lengths: list[int] = []   
    sample_vert_lengths: list[int] = []   

    seen_diag_starts: set[tuple[int, int]] = set()
    seen_vert_starts: set[tuple[int, int]] = set()
    diag_length_cache: dict[tuple[int, int], int] = {}
    vert_length_cache: dict[tuple[int, int], int] = {}

    for idx in range(M):
        i, j = int(pairs_i[idx]), int(pairs_j[idx])

        if not is_recurrent(i, j):
            continue

        sampled_recurrent += 1

        di, dj = i, j
        while di > 0 and dj > 0 and is_recurrent(di - 1, dj - 1):
            di -= 1
            dj -= 1

        diag_key = (min(di, dj), abs(di - dj))

        if diag_key not in seen_diag_starts:
            seen_diag_starts.add(diag_key)

            length = 0
            ci, cj = di, dj
            while ci < n and cj < n and ci != cj and is_recurrent(ci, cj):
                length += 1
                ci += 1
                cj += 1

            diag_length_cache[diag_key] = length
            if length > 0:
                diag_lengths.append(length)

        sample_diag_lengths.append(diag_length_cache.get(diag_key, 0))

        vi = i
        while vi > 0 and vi - 1 != j and is_recurrent(vi - 1, j):
            vi -= 1

        vert_key = (j, vi)

        if vert_key not in seen_vert_starts:
            seen_vert_starts.add(vert_key)

            length = 0
            ci = vi
            while ci < n and ci != j and is_recurrent(ci, j):
                length += 1
                ci += 1

            vert_length_cache[vert_key] = length
            if length > 0:
                vert_lengths.append(length)

        sample_vert_lengths.append(vert_length_cache.get(vert_key, 0))

    rr = float(sampled_recurrent / max(M, 1))

    diag_arr = np.array(diag_lengths, dtype=int) if diag_lengths else np.array([], dtype=int)
    sample_diag = np.array(sample_diag_lengths, dtype=int) if sample_diag_lengths else np.array([], dtype=int)

    if sampled_recurrent > 0 and sample_diag.size > 0:
        det = float(np.sum(sample_diag >= lmin) / sampled_recurrent)
    else:
        det = 0.0

    diag_ge = diag_arr[diag_arr >= lmin]
    lmax = float(diag_ge.max()) if diag_ge.size else 0.0

    vert_arr = np.array(vert_lengths, dtype=int) if vert_lengths else np.array([], dtype=int)
    sample_vert = np.array(sample_vert_lengths, dtype=int) if sample_vert_lengths else np.array([], dtype=int)

    if sampled_recurrent > 0 and sample_vert.size > 0:
        lam = float(np.sum(sample_vert >= vmin) / sampled_recurrent)
    else:
        lam = 0.0

    vert_ge = vert_arr[vert_arr >= vmin]
    if sampled_recurrent > 0 and sample_vert.size > 0:
        valid_vert = sample_vert[sample_vert >= vmin]
        tt = float(valid_vert.mean()) if valid_vert.size > 0 else 0.0
    else:
        tt = float(vert_ge.mean()) if vert_ge.size else 0.0

    entr = 0.0
    if diag_ge.size:
        counts = np.bincount(diag_ge)
        p = counts[counts > 0].astype(float)
        p /= p.sum()
        entr = float(-(p * np.log(p)).sum())

    return {"rr": rr, "det": det, "lam": lam, "lmax": lmax, "tt": tt, "entr": entr}


def rqa_worp_vectorised(X: np.ndarray, eps: float, lmin: int = 2, vmin: int = 2) -> dict[str, float]:
    """
    Vectorised RQA without RP construction.

    Instead of building the full N×N boolean matrix, processes one
    diagonal / column at a time using vectorised distance computation.
    Each diagonal/column requires O(N) memory, not O(N²).

    This is the PRACTICAL Python implementation, the pure loop version
    (rqa_worp) is too slow in Python for N > 500.
    """
    n = X.shape[0]
    total_recurrent = 0
    eps2 = eps * eps

    from scalable_rqa_volatility.recurrence.rqa import _run_lengths_1d

    diag_lengths: list[int] = []

    for k in range(1, n):
        diff = X[:n - k] - X[k:]  
        d2 = np.sum(diff * diff, axis=1)  
        recur = d2 <= eps2

        total_recurrent += int(recur.sum()) * 2  

        runs = _run_lengths_1d(recur)
        if runs.size:
            diag_lengths.extend(runs.tolist())
            diag_lengths.extend(runs.tolist()) 

    diag_arr = np.array(diag_lengths, dtype=int) if diag_lengths else np.array([], dtype=int)

    vert_lengths: list[int] = []

    for j in range(n):
        diff = X - X[j]  
        d2 = np.sum(diff * diff, axis=1)  
        d2[j] = np.inf  
        recur = d2 <= eps2

        runs = _run_lengths_1d(recur)
        if runs.size:
            vert_lengths.extend(runs.tolist())

    vert_arr = np.array(vert_lengths, dtype=int) if vert_lengths else np.array([], dtype=int)

    total_pairs = n * (n - 1)
    rr = float(total_recurrent / max(total_pairs, 1))

    diag_ge = diag_arr[diag_arr >= lmin]
    det = float(diag_ge.sum() / max(total_recurrent, 1))
    lmax = float(diag_ge.max()) if diag_ge.size else 0.0

    vert_ge = vert_arr[vert_arr >= vmin]
    lam = float(vert_ge.sum() / max(total_recurrent, 1))
    tt = float(vert_ge.mean()) if vert_ge.size else 0.0

    entr = 0.0
    if diag_ge.size:
        counts = np.bincount(diag_ge)
        p = counts[counts > 0].astype(float)
        p /= p.sum()
        entr = float(-(p * np.log(p)).sum())

    return {"rr": rr, "det": det, "lam": lam, "lmax": lmax, "tt": tt, "entr": entr}


def rqa_samp_features_rolling(
    df: pd.DataFrame,
    cols: tuple[str, ...],
    cfg: RQAConfig,
    M_factor: float = 4.0,
    prefix: str = "rqa_samp",
    eps_fixed: float | None = None,
) -> pd.DataFrame:
    """
    Windowed RQA using RQA_Samp within each window.
    Drop-in replacement for rqa_features_rolling.

    M_factor: M = M_factor × N_embedded per window.
    """
    cols_list = list(cols)
    n = len(df)
    feature_names = ["rr", "det", "lam", "lmax", "tt", "entr"]
    out_cols = [f"{prefix}_joint_{f}" for f in feature_names]
    out_np = np.full((n, len(out_cols)), np.nan, dtype=float)

    arr = df[cols_list].to_numpy(dtype=float, copy=False)

    start_t = cfg.window - 1
    for t in range(start_t, n, cfg.step):
        w0 = t - cfg.window + 1
        w1 = t + 1
        W = arr[w0:w1, :]
        if cfg.standardize:
            W = _standardize_cols(W)
        X = _delay_embed_joint(W, cfg.embed)
        n_emb = X.shape[0]

        if eps_fixed is not None:
            eps = float(eps_fixed)
        else:
            D_tmp = _pairwise_distances(X)
            eps = _epsilon_for_rr(D_tmp, cfg.recurrence_rate, cfg.exclude_diagonal)

        M = max(int(M_factor * n_emb), 100)
        feats = rqa_samp(X, eps, M=M)

        for i, f in enumerate(feature_names):
            out_np[t, i] = float(feats[f])

    out = pd.DataFrame(out_np, index=df.index, columns=out_cols)
    return out.ffill()


def benchmark_single_stock(X: np.ndarray, eps: float, logger, n_values=None):
    """
    Benchmark A: Compare methods on a single embedded time series.
    Tests accuracy and timing at different N values.
    """
    if n_values is None:
        n_values = [100, 200, 500, 1000, 2000]
        n_values = [nv for nv in n_values if nv <= X.shape[0]]

    results = []

    for N in n_values:
        X_sub = X[:N]
        logger.info(f"  N={N}:")

        t0 = time.time()
        exact = rqa_standard(X_sub, eps)
        t_standard = time.time() - t0

        t0 = time.time()
        worp = rqa_worp_vectorised(X_sub, eps)
        t_worp = time.time() - t0

        samp_results = {}
        for M_factor in [0.2, 1.0, 4.0]:
            M = max(int(M_factor * N), 50)
            t0 = time.time()
            samp = rqa_samp(X_sub, eps, M=M)
            t_samp = time.time() - t0
            samp_results[M_factor] = {"measures": samp, "time": t_samp, "M": M}

        logger.info(f"    Standard:       {t_standard:.4f}s")
        logger.info(f"    woRP (vec):     {t_worp:.4f}s  (speedup: {t_standard/max(t_worp,1e-6):.1f}x)")
        for mf, sr in samp_results.items():
            logger.info(f"    Samp M={mf:.1f}N:   {sr['time']:.4f}s  "
                        f"(speedup: {t_standard/max(sr['time'],1e-6):.1f}x)")

        for measure in ["rr", "det", "lam", "lmax", "tt", "entr"]:
            worp_err = abs(worp[measure] - exact[measure])
            samp_errs = {mf: abs(sr["measures"][measure] - exact[measure])
                         for mf, sr in samp_results.items()}
            if exact[measure] != 0:
                worp_rel = worp_err / abs(exact[measure])
                samp_rels = {mf: e / abs(exact[measure]) for mf, e in samp_errs.items()}
            else:
                worp_rel = worp_err
                samp_rels = samp_errs

        results.append({
            "N": N,
            "t_standard": t_standard,
            "t_worp": t_worp,
            "exact": exact,
            "worp": worp,
            "samp": samp_results,
        })

    return results


def benchmark_windowed_pipeline(df_stock: pd.DataFrame, rqa_cfg: RQAConfig,
                                 eps_fixed: float, logger):
    """
    Benchmark B: Compare windowed approaches on one stock for the ML pipeline.
    """
    rqa_cols = ("log_return", "rv")
    n_bars = len(df_stock)
    logger.info(f"  Stock bars: {n_bars}")

    t0 = time.time()
    X_current = rqa_features_rolling(df_stock, rqa_cols, rqa_cfg,
                                     prefix="rqa", eps_fixed=eps_fixed)
    t_current = time.time() - t0

    t0 = time.time()
    X_samp4 = rqa_samp_features_rolling(df_stock, rqa_cols, rqa_cfg,
                                         M_factor=4.0, prefix="rqa_s4",
                                         eps_fixed=eps_fixed)
    t_samp4 = time.time() - t0

    t0 = time.time()
    X_samp1 = rqa_samp_features_rolling(df_stock, rqa_cols, rqa_cfg,
                                         M_factor=1.0, prefix="rqa_s1",
                                         eps_fixed=eps_fixed)
    t_samp1 = time.time() - t0

    logger.info(f"    Windowed exact:      {t_current:.2f}s")
    logger.info(f"    Windowed Samp M=4N:  {t_samp4:.2f}s  "
                f"(speedup: {t_current/max(t_samp4,1e-6):.1f}x)")
    logger.info(f"    Windowed Samp M=1N:  {t_samp1:.2f}s  "
                f"(speedup: {t_current/max(t_samp1,1e-6):.1f}x)")

    valid = X_current.notna().all(axis=1) & X_samp4.notna().all(axis=1)
    if valid.sum() > 0:
        corrs = {}
        for col_exact, col_samp in zip(X_current.columns, X_samp4.columns):
            c = X_current.loc[valid, col_exact].corr(X_samp4.loc[valid, col_samp])
            corrs[col_exact.split("_")[-1]] = c
        logger.info(f"    Samp M=4N correlation with exact: {corrs}")

    return {
        "t_current": t_current,
        "t_samp4": t_samp4,
        "t_samp1": t_samp1,
    }


def plot_accuracy_vs_M(single_results, fig_name):
    """Fig 7.1: RQA_Samp accuracy vs sample size M."""
    measures = ["rr", "det", "lam", "tt", "entr"]
    res = single_results[-1]
    N = res["N"]
    exact = res["exact"]

    M_factors = sorted(res["samp"].keys())
    M_values = [res["samp"][mf]["M"] for mf in M_factors]

    fig, axes = plt.subplots(1, len(measures), figsize=(3.5 * len(measures), 4))
    fig.suptitle(f"RQA_Samp Accuracy vs Sample Size (N={N})", fontsize=14)

    for ax, measure in zip(axes, measures):
        exact_val = exact[measure]
        samp_vals = [res["samp"][mf]["measures"][measure] for mf in M_factors]

        ax.axhline(exact_val, color=COLORS["red"], linestyle="--", label="Exact", linewidth=2)
        ax.plot(M_values, samp_vals, "o-", color=COLORS["green"], label="RQA_Samp", linewidth=2)

        ax.set_xlabel("M (samples)")
        ax.set_ylabel(measure.upper())
        ax.set_title(measure.upper())
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    savefig(fig, fig_name)


def plot_timing_scaling(single_results, fig_name):
    """Fig 7.2: Wall-clock time vs N for different methods."""
    N_vals = [r["N"] for r in single_results]
    t_standard = [r["t_standard"] for r in single_results]
    t_worp = [r["t_worp"] for r in single_results]
    t_samp4 = [r["samp"][4.0]["time"] for r in single_results if 4.0 in r["samp"]]
    t_samp1 = [r["samp"][1.0]["time"] for r in single_results if 1.0 in r["samp"]]
    t_samp02 = [r["samp"][0.2]["time"] for r in single_results if 0.2 in r["samp"]]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(N_vals, t_standard, "o-", color=COLORS["red"], linewidth=2, label="Standard (full RP)")
    ax.plot(N_vals, t_worp, "s-", color=COLORS["accent"], linewidth=2, label="RQA_woRP (exact, no RP)")
    if t_samp4:
        ax.plot(N_vals[:len(t_samp4)], t_samp4, "^-", color=COLORS["green"],
                linewidth=2, label="RQA_Samp M=4N")
    if t_samp1:
        ax.plot(N_vals[:len(t_samp1)], t_samp1, "v-", color=COLORS["purple"],
                linewidth=2, label="RQA_Samp M=N")
    if t_samp02:
        ax.plot(N_vals[:len(t_samp02)], t_samp02, "d-", color=COLORS["orange"],
                linewidth=2, label="RQA_Samp M=0.2N")

    ax.set_xlabel("N (embedded vectors)")
    ax.set_ylabel("Wall-clock time (seconds)")
    ax.set_title("RQA Computation Time vs Series Length\n(Single stock, m=4, tau=2)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale("log")
    ax.set_xscale("log")

    fig.tight_layout()
    savefig(fig, fig_name)


def plot_pipeline_comparison(windowed_results, pipeline_total, fig_name):
    """Fig 7.3: Pipeline timing comparison bar chart."""
    methods = ["Windowed exact\n(current)", "Windowed Samp\nM=4N", "Windowed Samp\nM=N"]
    times = [windowed_results["t_current"], windowed_results["t_samp4"],
             windowed_results["t_samp1"]]
    colors_list = [COLORS["red"], COLORS["green"], COLORS["purple"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    bars = ax.barh(range(len(methods)), times, color=colors_list, edgecolor="white", height=0.5)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Per-Stock Windowed RQA Timing")
    ax.invert_yaxis()
    for bar, t in zip(bars, times):
        ax.text(t + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{t:.2f}s", va="center", fontsize=10)

    ax = axes[1]
    n_stocks = 503
    proj_times = [t * n_stocks for t in times]
    proj_labels = [f"{t:.0f}s ({t/60:.1f}min)" for t in proj_times]
    bars = ax.barh(range(len(methods)), proj_times, color=colors_list, edgecolor="white", height=0.5)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xlabel("Projected time for 503 stocks (seconds)")
    ax.set_title("Projected Full D3 Pipeline Timing")
    ax.invert_yaxis()
    for bar, t, label in zip(bars, proj_times, proj_labels):
        ax.text(t + 5, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=10)

    if pipeline_total:
        ax.axvline(pipeline_total, color=COLORS["gray"], linestyle="--", alpha=0.7,
                   label=f"Measured total: {pipeline_total:.0f}s")
        ax.legend()

    fig.suptitle("Dataset 3: Scalable RQA Pipeline Comparison", fontsize=14)
    fig.tight_layout()
    savefig(fig, fig_name)


def main():
    logger = get_logger()
    parser = argparse.ArgumentParser(description="Benchmark scalable RQA methods on D3")
    parser.add_argument("--max_tickers", type=int, default=10,
                        help="Number of tickers for pipeline benchmark (default: 10)")
    parser.add_argument("--full", action="store_true",
                        help="Run on all 503 stocks (slow)")
    args = parser.parse_args()

    if args.full:
        args.max_tickers = None

    print("=" * 70)
    print("  BENCHMARK: SCALABLE RQA METHODS ON DATASET 3")
    print("  Implements RQA_woRP + RQA_Samp (Marwan 2025)")
    print("=" * 70)

    logger.info("Loading D3 train split...")
    train = load_split("train")
    tickers = sorted(train["ticker"].unique())
    logger.info(f"Available tickers: {len(tickers)}")

    sample_ticker = tickers[len(tickers) // 2]  # middle ticker
    df_stock = train[train["ticker"] == sample_ticker].reset_index(drop=True)
    logger.info(f"Sample stock: {sample_ticker}, {len(df_stock)} bars")

    rqa_cfg = RQAConfig(window=60, step=20, recurrence_rate=0.1,
                        embed=EmbeddingConfig(m=4, tau=2), mode="joint")
    rqa_cols = ("log_return", "rv")

    arr_full = df_stock[list(rqa_cols)].to_numpy(dtype=float)
    arr_full = _standardize_cols(arr_full)
    X_full = _delay_embed_joint(arr_full, rqa_cfg.embed)
    logger.info(f"Full embedded series: N={X_full.shape[0]}, d={X_full.shape[1]}")

    eps = estimate_epsilon_from_train(
        df_stock[list(rqa_cols)].to_numpy(dtype=float), rqa_cfg)
    logger.info(f"Epsilon: {eps:.6f}")

    print("\n--- A. Single-Stock Benchmark (accuracy + timing) ---")
    n_max = min(X_full.shape[0], 3000)
    n_values = [v for v in [50, 100, 200, 500, 1000, 2000, 3000] if v <= n_max]
    single_results = benchmark_single_stock(X_full, eps, logger, n_values=n_values)

    res = single_results[-1]
    print(f"\n  Accuracy comparison at N={res['N']}:")
    print(f"  {'Measure':<8} {'Exact':>10} {'woRP':>10} {'Samp 4N':>10} {'Samp N':>10} {'Samp 0.2N':>10}")
    print("  " + "-" * 58)
    for m in ["rr", "det", "lam", "lmax", "tt", "entr"]:
        exact_v = res["exact"][m]
        worp_v = res["worp"][m]
        s4 = res["samp"][4.0]["measures"][m] if 4.0 in res["samp"] else float("nan")
        s1 = res["samp"][1.0]["measures"][m] if 1.0 in res["samp"] else float("nan")
        s02 = res["samp"][0.2]["measures"][m] if 0.2 in res["samp"] else float("nan")
        print(f"  {m:<8} {exact_v:>10.4f} {worp_v:>10.4f} {s4:>10.4f} {s1:>10.4f} {s02:>10.4f}")

    print("\n--- B. Windowed Pipeline Benchmark (per-stock) ---")
    windowed_results = benchmark_windowed_pipeline(df_stock, rqa_cfg, eps, logger)

    multi_stock_time = None
    n_bench = args.max_tickers if args.max_tickers is not None else len(tickers)
    if n_bench > 1:
        print(f"\n--- C. Multi-Stock Pipeline ({n_bench} stocks) ---")
        subset_tickers = tickers[:n_bench]

        t0 = time.time()
        for i, ticker in enumerate(subset_tickers):
            df_t = train[train["ticker"] == ticker].reset_index(drop=True)
            rqa_features_rolling(df_t, rqa_cols, rqa_cfg, prefix="rqa", eps_fixed=eps)
            if (i + 1) % max(1, n_bench // 5) == 0:
                logger.info(f"  Current exact: {i+1}/{n_bench} "
                            f"({time.time()-t0:.1f}s)")
        t_multi_current = time.time() - t0

        t0 = time.time()
        for i, ticker in enumerate(subset_tickers):
            df_t = train[train["ticker"] == ticker].reset_index(drop=True)
            rqa_samp_features_rolling(df_t, rqa_cols, rqa_cfg, M_factor=4.0,
                                      prefix="rqa_s", eps_fixed=eps)
            if (i + 1) % max(1, n_bench // 5) == 0:
                logger.info(f"  Samp M=4N: {i+1}/{n_bench} "
                            f"({time.time()-t0:.1f}s)")
        t_multi_samp = time.time() - t0

        logger.info(f"  {n_bench} stocks — Exact: {t_multi_current:.1f}s, "
                    f"Samp M=4N: {t_multi_samp:.1f}s "
                    f"(speedup: {t_multi_current/max(t_multi_samp,1e-6):.1f}x)")

        if n_bench < len(tickers):
            proj_exact = t_multi_current / n_bench * 503
            proj_samp = t_multi_samp / n_bench * 503
            logger.info(f"  Projected 503 stocks — Exact: {proj_exact:.0f}s ({proj_exact/60:.1f}min), "
                        f"Samp: {proj_samp:.0f}s ({proj_samp/60:.1f}min)")
        else:
            proj_exact = t_multi_current
            proj_samp = t_multi_samp
            logger.info(f"  FULL 503 stocks — Exact: {proj_exact:.0f}s ({proj_exact/60:.1f}min), "
                        f"Samp: {proj_samp:.0f}s ({proj_samp/60:.1f}min)")
        multi_stock_time = proj_exact

    print("\n--- Generating figures ---")
    plot_accuracy_vs_M(single_results, "7_1_rqa_samp_accuracy")
    plot_timing_scaling(single_results, "7_2_rqa_timing_scaling")
    plot_pipeline_comparison(windowed_results, multi_stock_time, "7_3_rqa_pipeline_comparison")

    out_dir = repo_root() / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "benchmark_scalable_rqa.txt"

    lines = []
    lines.append("=" * 80)
    lines.append("  SCALABLE RQA BENCHMARK — DATASET 3")
    lines.append("  Methods: Standard (full RP) | RQA_woRP (Marwan 2025) | RQA_Samp (Marwan 2025)")
    lines.append("=" * 80)

    lines.append(f"\n\nA. SINGLE-STOCK TIMING (stock: {sample_ticker}, {len(df_stock)} bars)\n")
    lines.append(f"  {'N':>6} {'Standard':>10} {'woRP':>10} {'Samp 4N':>10} {'Samp N':>10} {'Samp 0.2N':>10}")
    lines.append("  " + "-" * 58)
    for r in single_results:
        t_std = r["t_standard"]
        t_worp = r["t_worp"]
        t_s4 = r["samp"].get(4.0, {}).get("time", float("nan"))
        t_s1 = r["samp"].get(1.0, {}).get("time", float("nan"))
        t_s02 = r["samp"].get(0.2, {}).get("time", float("nan"))
        lines.append(f"  {r['N']:>6} {t_std:>10.4f}s {t_worp:>10.4f}s "
                     f"{t_s4:>10.4f}s {t_s1:>10.4f}s {t_s02:>10.4f}s")

    lines.append(f"\n\nB. ACCURACY AT N={single_results[-1]['N']} (vs exact standard RQA)\n")
    res = single_results[-1]
    lines.append(f"  {'Measure':<8} {'Exact':>10} {'woRP':>10} {'Samp 4N':>10} {'Samp N':>10} {'Samp 0.2N':>10}")
    lines.append("  " + "-" * 58)
    for m in ["rr", "det", "lam", "lmax", "tt", "entr"]:
        exact_v = res["exact"][m]
        worp_v = res["worp"][m]
        s4 = res["samp"].get(4.0, {}).get("measures", {}).get(m, float("nan"))
        s1 = res["samp"].get(1.0, {}).get("measures", {}).get(m, float("nan"))
        s02 = res["samp"].get(0.2, {}).get("measures", {}).get(m, float("nan"))
        lines.append(f"  {m:<8} {exact_v:>10.4f} {worp_v:>10.4f} {s4:>10.4f} {s1:>10.4f} {s02:>10.4f}")

    lines.append(f"\n\nC. WINDOWED PIPELINE TIMING (per stock, w={rqa_cfg.window}, s={rqa_cfg.step})\n")
    lines.append(f"  Windowed exact (current):  {windowed_results['t_current']:.2f}s")
    lines.append(f"  Windowed Samp M=4N:        {windowed_results['t_samp4']:.2f}s "
                 f"(speedup: {windowed_results['t_current']/max(windowed_results['t_samp4'],1e-6):.1f}x)")
    lines.append(f"  Windowed Samp M=N:         {windowed_results['t_samp1']:.2f}s "
                 f"(speedup: {windowed_results['t_current']/max(windowed_results['t_samp1'],1e-6):.1f}x)")

    lines.append(f"\n  Projected 503 stocks:")
    for label, t in [("Exact", windowed_results["t_current"]),
                     ("Samp M=4N", windowed_results["t_samp4"]),
                     ("Samp M=N", windowed_results["t_samp1"])]:
        proj = t * 503
        lines.append(f"    {label:<20}: {proj:.0f}s ({proj/60:.1f} min)")

    results_text = "\n".join(lines)
    out_path.write_text(results_text, encoding="utf-8")
    logger.info(f"\nResults saved to {out_path}")
    print(results_text)

    print("\n" + "=" * 70)
    print("  DONE — Check figures/ and results/ directories")
    print("=" * 70)


if __name__ == "__main__":
    main()