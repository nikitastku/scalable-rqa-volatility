"""
Test RQA_Samp in the classification pipeline.

Supervisor feedback: "If you do this now, if you add this to your pipeline,
how big of an impact does that have on your performance? Do you completely
wipe it out, or is there still something salvageable?"

This script answers that directly:
  1. Compute exact windowed RQA features (current pipeline)
  2. Compute RQA_Samp windowed features at M=4N, M=N, M=0.2N
  3. Train RF Std+RQA with each feature set
  4. Compare AUC, F1, and per-stock metrics
  5. Also run Wilcoxon signed-rank test: exact vs sampled
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score

from scalable_rqa_volatility.logging_utils import get_logger
from scalable_rqa_volatility.recurrence.embeddings import EmbeddingConfig
from scalable_rqa_volatility.recurrence.rqa import (
    RQAConfig, rqa_features_rolling, estimate_epsilon_from_train,
    _standardize_cols, _delay_embed_joint, _pairwise_distances,
    _epsilon_for_rr,
)

from scalable_rqa_volatility.recurrence.rqa import (
    _recurrence_matrix, _delay_embed_joint, _standardize_cols,
)

STD_FEATURE_COLS = [
    "ret_abs", "ret_sq", "rv",
    "ret_mean_30", "ret_std_30", "ret_abs_mean_30", "rv_mean_30", "rv_std_30",
    "ret_mean_120", "ret_std_120", "ret_abs_mean_120", "rv_mean_120", "rv_std_120",
    "ret_mean_390", "ret_std_390", "ret_abs_mean_390", "rv_mean_390", "rv_std_390",
]

COLORS = {
    "primary": "#1E2761", "accent": "#3B82F6", "green": "#10B981",
    "red": "#EF4444", "orange": "#F59E0B", "purple": "#8B5CF6",
    "gray": "#94A3B8",
}
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "legend.fontsize": 9, "figure.figsize": (12, 6),
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


def rqa_samp_single(X, eps, M, lmin=2, vmin=2, seed=42):
    """Sampled RQA with corrected DET/LAM estimation."""
    n = X.shape[0]
    rng = np.random.RandomState(seed)
    eps2 = eps * eps
    sq_norms = np.sum(X * X, axis=1)

    def is_recurrent(i, j):
        d2 = float(sq_norms[i] + sq_norms[j] - 2.0 * np.dot(X[i], X[j]))
        return d2 <= eps2

    pairs_i = rng.randint(0, n, size=M)
    pairs_j = rng.randint(0, n, size=M)
    same = pairs_i == pairs_j
    while same.any():
        pairs_j[same] = rng.randint(0, n, size=int(same.sum()))
        same = pairs_i == pairs_j

    sampled_recurrent = 0
    diag_lengths = []
    vert_lengths = []
    sample_diag_lengths = []
    sample_vert_lengths = []
    seen_diag_starts = set()
    seen_vert_starts = set()
    diag_length_cache = {}
    vert_length_cache = {}

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
    det = float(np.sum(sample_diag >= lmin) / sampled_recurrent) if sampled_recurrent > 0 and sample_diag.size > 0 else 0.0
    diag_ge = diag_arr[diag_arr >= lmin]
    lmax = float(diag_ge.max()) if diag_ge.size else 0.0

    vert_arr = np.array(vert_lengths, dtype=int) if vert_lengths else np.array([], dtype=int)
    sample_vert = np.array(sample_vert_lengths, dtype=int) if sample_vert_lengths else np.array([], dtype=int)
    lam = float(np.sum(sample_vert >= vmin) / sampled_recurrent) if sampled_recurrent > 0 and sample_vert.size > 0 else 0.0

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


def rqa_samp_features_rolling(df, cols, cfg, M_factor, prefix, eps_fixed=None):
    """Windowed RQA using sampling within each window."""
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
        feats = rqa_samp_single(X, eps, M=M)
        for i, f in enumerate(feature_names):
            out_np[t, i] = float(feats[f])

    out = pd.DataFrame(out_np, index=df.index, columns=out_cols)
    return out.ffill()


def compute_rqa_per_stock(df, rqa_cfg, rqa_cols, eps_fixed, method="exact", M_factor=4.0):
    """Compute RQA per stock — exact or sampled."""
    all_rqa = []
    tickers = sorted(df["ticker"].unique())

    for ticker in tickers:
        mask = df["ticker"] == ticker
        df_ticker = df.loc[mask].reset_index(drop=True)

        if method == "exact":
            rqa_feats = rqa_features_rolling(
                df_ticker, rqa_cols, rqa_cfg, prefix="rqa", eps_fixed=eps_fixed)
        else:
            prefix = f"rqa_s{M_factor:.0f}"
            rqa_feats = rqa_samp_features_rolling(
                df_ticker, rqa_cols, rqa_cfg, M_factor=M_factor,
                prefix=prefix, eps_fixed=eps_fixed)

        rqa_feats.index = df.index[mask]
        all_rqa.append(rqa_feats)

    return pd.concat(all_rqa).sort_index()


def build_xy_per_stock(df, X_df):
    all_X, all_y, all_tickers = [], [], []
    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker
        df_t = df.loc[mask].reset_index(drop=True)
        X_t = X_df.loc[mask].reset_index(drop=True)
        y_next = df_t["regime"].shift(-1).to_numpy()
        ok = pd.notna(y_next)
        X_arr = X_t.to_numpy(dtype=float)[ok]
        y_arr = y_next[ok].astype(int)
        keep = np.isfinite(X_arr).all(axis=1) & np.isfinite(y_arr)
        if keep.sum() > 0:
            all_X.append(X_arr[keep])
            all_y.append(y_arr[keep])
            all_tickers.extend([ticker] * int(keep.sum()))
    return np.concatenate(all_X), np.concatenate(all_y), np.array(all_tickers)


def f1_fast(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    denom = 2 * tp + fp + fn
    return float((2 * tp) / denom) if denom else 0.0


def best_threshold(y_true, prob):
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    prob = np.asarray(prob, dtype=float).reshape(-1)
    if y_true.size == 0:
        return 0.5
    target = float(np.clip(np.mean(y_true), 0.05, 0.95))
    if prob.size > 100000:
        grid = np.unique(np.quantile(prob, np.linspace(0.01, 0.99, 499)))
    else:
        grid = np.unique(prob)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        y_pred = (prob >= float(t)).astype(int)
        pos = float(np.mean(y_pred))
        if not (target * 0.5 <= pos <= min(0.99, target * 1.5)):
            continue
        f1 = f1_fast(y_true, y_pred)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    if best_f1 < 0:
        best_t = float(np.quantile(prob, 1.0 - target))
    return best_t


def per_stock_metrics(y_true, y_score, y_pred, tickers):
    unique_tickers = sorted(set(tickers))
    stock_aucs, stock_f1s, valid_tickers = [], [], []
    for t in unique_tickers:
        mask = tickers == t
        yt, ys, yp = y_true[mask], y_score[mask], y_pred[mask]
        if len(np.unique(yt)) < 2 or len(yt) < 10:
            continue
        try:
            auc = roc_auc_score(yt, ys)
            f1 = f1_score(yt, yp, zero_division=0)
            stock_aucs.append(auc)
            stock_f1s.append(f1)
            valid_tickers.append(t)
        except ValueError:
            continue
    return np.array(stock_aucs), np.array(stock_f1s), valid_tickers


def train_evaluate(name, Xtr, ytr, Xva, yva, Xte, yte, tickers_te, logger):
    """Train RF, tune threshold, evaluate on test set with per-stock metrics."""
    rf = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5, n_jobs=-1,
        class_weight="balanced_subsample", random_state=42,
    )
    t0 = time.time()
    rf.fit(Xtr, ytr)
    fit_time = time.time() - t0

    p_va = rf.predict_proba(Xva)[:, 1]
    thr = best_threshold(yva, p_va)
    p_te = rf.predict_proba(Xte)[:, 1]
    y_pred = (p_te >= thr).astype(int)

    auc_pooled = roc_auc_score(yte, p_te)
    f1_pooled = f1_score(yte, y_pred, zero_division=0)

    aucs, f1s, valid = per_stock_metrics(yte, p_te, y_pred, tickers_te)

    logger.info(f"  {name}: AUC={auc_pooled:.6f}, F1={f1_pooled:.4f}, "
                f"per-stock AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}, "
                f"fit={fit_time:.1f}s")

    return {
        "name": name,
        "auc_pooled": auc_pooled,
        "f1_pooled": f1_pooled,
        "per_stock_aucs": aucs,
        "per_stock_f1s": f1s,
        "valid_tickers": valid,
        "y_true": yte,
        "y_score": p_te,
        "y_pred": y_pred,
        "tickers": tickers_te,
        "fit_time": fit_time,
    }


def plot_comparison(results_dict, fig_name):
    """Compare classification performance across feature computation methods."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    names = list(results_dict.keys())
    colors_list = [COLORS["accent"], COLORS["green"], COLORS["purple"], COLORS["orange"]]

    ax = axes[0]
    aucs = [results_dict[n]["auc_pooled"] for n in names]
    bars = ax.barh(range(len(names)), aucs, color=colors_list[:len(names)],
                   edgecolor="white", height=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Pooled ROC-AUC")
    ax.set_title("A. Pooled AUC")
    ax.set_xlim(min(aucs) - 0.001, max(aucs) + 0.001)
    ax.invert_yaxis()
    for bar, auc in zip(bars, aucs):
        ax.text(auc + 0.0001, bar.get_y() + bar.get_height() / 2,
                f"{auc:.6f}", va="center", fontsize=9)

    ax = axes[1]
    data = [results_dict[n]["per_stock_aucs"] for n in names]
    bp = ax.boxplot(data, labels=names, vert=True, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors_list[:len(names)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel("Per-stock AUC")
    ax.set_title("B. Per-Stock AUC Distribution")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    exact_aucs = results_dict[names[0]]["per_stock_aucs"]
    exact_valid = results_dict[names[0]]["valid_tickers"]
    for i, n in enumerate(names[1:], 1):
        samp_aucs = results_dict[n]["per_stock_aucs"]
        samp_valid = results_dict[n]["valid_tickers"]
        common = sorted(set(exact_valid) & set(samp_valid))
        idx_e = {t: j for j, t in enumerate(exact_valid)}
        idx_s = {t: j for j, t in enumerate(samp_valid)}
        diff = np.array([samp_aucs[idx_s[t]] - exact_aucs[idx_e[t]] for t in common])
        ax.hist(diff, bins=40, alpha=0.5, color=colors_list[i],
                label=f"{n}\nmean={np.mean(diff):+.5f}")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("ΔAUC (sampled - exact)")
    ax.set_ylabel("Count")
    ax.set_title("C. Per-Stock AUC Change")
    ax.legend(fontsize=8)

    fig.suptitle("Classification Performance: Exact vs Sampled RQA Features",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    savefig(fig, fig_name)


def main():
    logger = get_logger()

    print("=" * 70)
    print("  CLASSIFICATION WITH RQA_Samp vs EXACT — DATASET 3")
    print("  Question: Does using sampled features hurt classification?")
    print("=" * 70)

    logger.info("Loading D3 splits...")
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")
    full = pd.concat([train, val, test], ignore_index=True)
    n_tr, n_va = len(train), len(val)
    logger.info(f"Train: {n_tr:,}, Val: {n_va:,}, Test: {len(test):,}")

    rqa_cfg = RQAConfig(window=60, step=20, recurrence_rate=0.1,
                        embed=EmbeddingConfig(m=4, tau=2), mode="joint")
    rqa_cols = ("log_return", "rv")

    tickers = sorted(train["ticker"].unique())
    sample_tickers = tickers[:min(10, len(tickers))]
    sample_blocks = [train[train["ticker"] == t][list(rqa_cols)].to_numpy(dtype=float)
                     for t in sample_tickers]
    eps_fixed = estimate_epsilon_from_train(np.concatenate(sample_blocks), rqa_cfg)
    logger.info(f"Epsilon: {eps_fixed:.6f}")

    logger.info("Building standard feature arrays...")
    X_std_tr, y_tr, tickers_tr = build_xy_per_stock(train, train[STD_FEATURE_COLS].copy())
    X_std_va, y_va, tickers_va = build_xy_per_stock(val, val[STD_FEATURE_COLS].copy())
    X_std_te, y_te, tickers_te = build_xy_per_stock(test, test[STD_FEATURE_COLS].copy())

    methods = [
        ("Exact", "exact", None),
        ("Samp M=4N", "samp", 4.0),
        ("Samp M=N", "samp", 1.0),
        ("Samp M=0.2N", "samp", 0.2),
    ]

    results_dict = {}

    for method_name, method_type, M_factor in methods:
        logger.info(f"\n--- {method_name} ---")

        t0 = time.time()
        logger.info(f"  Computing RQA features ({method_name})...")
        X_rqa_all = compute_rqa_per_stock(
            full, rqa_cfg, rqa_cols, eps_fixed,
            method=method_type, M_factor=M_factor if M_factor else 4.0,
        )
        comp_time = time.time() - t0
        logger.info(f"  RQA computation: {comp_time:.1f}s")

        X_rqa_tr_df = X_rqa_all.iloc[:n_tr].reset_index(drop=True)
        X_rqa_va_df = X_rqa_all.iloc[n_tr:n_tr + n_va].reset_index(drop=True)
        X_rqa_te_df = X_rqa_all.iloc[n_tr + n_va:].reset_index(drop=True)

        X_rqa_tr, y_tr2, tickers_tr2 = build_xy_per_stock(train, X_rqa_tr_df)
        X_rqa_va, y_va2, _ = build_xy_per_stock(val, X_rqa_va_df)
        X_rqa_te, y_te2, tickers_te2 = build_xy_per_stock(test, X_rqa_te_df)

        ntr = min(len(y_tr), len(y_tr2))
        nva = min(len(y_va), len(y_va2))
        nte = min(len(y_te), len(y_te2))

        X_comb_tr = np.concatenate([X_std_tr[:ntr], X_rqa_tr[:ntr]], axis=1)
        X_comb_va = np.concatenate([X_std_va[:nva], X_rqa_va[:nva]], axis=1)
        X_comb_te = np.concatenate([X_std_te[:nte], X_rqa_te[:nte]], axis=1)
        y_tr_a, y_va_a, y_te_a = y_tr[:ntr], y_va[:nva], y_te[:nte]
        tickers_te_a = tickers_te[:nte]

        result = train_evaluate(
            method_name, X_comb_tr, y_tr_a, X_comb_va, y_va_a,
            X_comb_te, y_te_a, tickers_te_a, logger,
        )
        result["comp_time"] = comp_time
        results_dict[method_name] = result

    logger.info("\n--- Wilcoxon signed-rank tests (per-stock AUC) ---")

    exact_r = results_dict["Exact"]
    wilcoxon_results = {}

    for samp_name in ["Samp M=4N", "Samp M=N", "Samp M=0.2N"]:
        samp_r = results_dict[samp_name]
        common = sorted(set(exact_r["valid_tickers"]) & set(samp_r["valid_tickers"]))
        idx_e = {t: i for i, t in enumerate(exact_r["valid_tickers"])}
        idx_s = {t: i for i, t in enumerate(samp_r["valid_tickers"])}

        aucs_e = np.array([exact_r["per_stock_aucs"][idx_e[t]] for t in common])
        aucs_s = np.array([samp_r["per_stock_aucs"][idx_s[t]] for t in common])
        diff = aucs_s - aucs_e

        try:
            stat, p_two = scipy_stats.wilcoxon(diff, alternative="two-sided")
        except ValueError:
            p_two = 1.0

        sig = "***" if p_two < 0.001 else "**" if p_two < 0.01 else "*" if p_two < 0.05 else "n.s."

        wilcoxon_results[samp_name] = {
            "mean_diff": float(np.mean(diff)),
            "median_diff": float(np.median(diff)),
            "p_value": float(p_two),
            "sig": sig,
            "n_stocks": len(common),
        }

        logger.info(f"  Exact vs {samp_name}: mean dAUC={np.mean(diff):+.6f}, "
                    f"median={np.median(diff):+.6f}, p={p_two:.4f} {sig}")

    print("\n--- Generating figures ---")
    plot_comparison(results_dict, "8_1_samp_vs_exact_classification")

    out_dir = repo_root() / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "classify_rqa_samp_d3.txt"

    lines = []
    lines.append("=" * 80)
    lines.append("  CLASSIFICATION WITH RQA_Samp vs EXACT — DATASET 3")
    lines.append("  Question: Does using sampled features hurt classification?")
    lines.append("=" * 80)

    lines.append(f"\n\nA. POOLED PERFORMANCE\n")
    lines.append(f"  {'Method':<20} {'AUC':>10} {'F1':>8} {'RQA time':>10} {'RF time':>10}")
    lines.append("  " + "-" * 60)
    for name, r in results_dict.items():
        lines.append(f"  {name:<20} {r['auc_pooled']:>10.6f} {r['f1_pooled']:>8.4f} "
                     f"{r['comp_time']:>9.1f}s {r['fit_time']:>9.1f}s")

    lines.append(f"\n\nB. PER-STOCK METRICS (mean ± std across stocks)\n")
    lines.append(f"  {'Method':<20} {'AUC':>18} {'F1':>18} {'N stocks':>10}")
    lines.append("  " + "-" * 68)
    for name, r in results_dict.items():
        aucs = r["per_stock_aucs"]
        f1s = r["per_stock_f1s"]
        lines.append(f"  {name:<20} {np.mean(aucs):.4f}±{np.std(aucs):.4f}"
                     f"      {np.mean(f1s):.4f}±{np.std(f1s):.4f}"
                     f"      {len(r['valid_tickers']):>5}")

    lines.append(f"\n\nC. WILCOXON SIGNED-RANK TESTS (Exact vs Sampled, per-stock AUC)\n")
    lines.append(f"  {'Comparison':<30} {'Mean dAUC':>12} {'Median dAUC':>14} {'p-value':>10} {'Sig':>5}")
    lines.append("  " + "-" * 73)
    for samp_name, w in wilcoxon_results.items():
        lines.append(f"  Exact vs {samp_name:<19} {w['mean_diff']:>+12.6f} "
                     f"{w['median_diff']:>+14.6f} {w['p_value']:>10.4f} {w['sig']:>5}")

    lines.append(f"\n\nD. AUC DIFFERENCES (vs Exact)\n")
    exact_auc = results_dict["Exact"]["auc_pooled"]
    for name, r in results_dict.items():
        if name == "Exact":
            continue
        delta = r["auc_pooled"] - exact_auc
        pct = delta / exact_auc * 100
        lines.append(f"  {name:<20}: ΔAUC = {delta:+.6f} ({pct:+.4f}%)")

    lines.append(f"\n\n{'='*80}")
    lines.append(f"  INTERPRETATION")
    lines.append(f"{'='*80}\n")

    samp4_delta = results_dict["Samp M=4N"]["auc_pooled"] - exact_auc
    samp1_delta = results_dict["Samp M=N"]["auc_pooled"] - exact_auc
    samp02_delta = results_dict["Samp M=0.2N"]["auc_pooled"] - exact_auc

    lines.append(f"  RQA_Samp at M=4N: ΔAUC = {samp4_delta:+.6f} vs exact.")
    if abs(samp4_delta) < 0.0005:
        lines.append("  → Negligible impact on classification. Safe to use for 1.6× speedup.")
    elif samp4_delta < -0.001:
        lines.append("  → Noticeable degradation. Exact features preferred.")
    else:
        lines.append("  → Small difference. Acceptable trade-off for speed.")

    lines.append(f"\n  RQA_Samp at M=N:  ΔAUC = {samp1_delta:+.6f} vs exact.")
    lines.append(f"  RQA_Samp at M=0.2N: ΔAUC = {samp02_delta:+.6f} vs exact.")

    w4 = wilcoxon_results.get("Samp M=4N", {})
    if w4.get("sig") == "n.s.":
        lines.append("\n  Wilcoxon test confirms: no statistically significant difference")
        lines.append("  between exact and Samp M=4N at the per-stock level.")

    results_text = "\n".join(lines)
    out_path.write_text(results_text, encoding="utf-8")
    logger.info(f"\nResults saved to {out_path}")
    print(results_text)

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()