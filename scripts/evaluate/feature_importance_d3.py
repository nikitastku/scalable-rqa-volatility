"""
feature_importance_d3.py — Feature importance analysis for Dataset 3.

Answers the supervisor's two questions:
  1. WHY do RQA features help? (Which RQA measures are most important?)
  2. Can we throw some away and still get the same performance?

Methods:
  A. Gini importance (MDI) — from the trained RF model
  B. Permutation importance — on the test set (model-agnostic)
  C. Ablation — drop each RQA feature one at a time, measure AUC change
  D. Feature group analysis — std vs RQA contribution breakdown
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scalable_rqa_volatility.logging_utils import get_logger
from scalable_rqa_volatility.recurrence.beta_rqa import BetaRQAConfig, beta_rqa_features_rolling
from scalable_rqa_volatility.recurrence.embeddings import EmbeddingConfig
from scalable_rqa_volatility.recurrence.rqa import (
    RQAConfig, rqa_features_rolling, estimate_epsilon_from_train,
)

STD_FEATURE_COLS = [
    "ret_abs", "ret_sq", "rv",
    "ret_mean_30", "ret_std_30", "ret_abs_mean_30", "rv_mean_30", "rv_std_30",
    "ret_mean_120", "ret_std_120", "ret_abs_mean_120", "rv_mean_120", "rv_std_120",
    "ret_mean_390", "ret_std_390", "ret_abs_mean_390", "rv_mean_390", "rv_std_390",
]

RQA_FEATURE_NAMES = ["rr", "det", "lam", "lmax", "tt", "entr"]
BETA_RQA_FEATURE_NAMES = ["rr", "det", "lam", "lmax", "tt", "entr",
                           "lam_h", "tt_h", "delta_lam", "delta_tt"]

COLORS = {
    "primary": "#1E2761",
    "accent": "#3B82F6",
    "green": "#10B981",
    "red": "#EF4444",
    "orange": "#F59E0B",
    "purple": "#8B5CF6",
    "gray": "#94A3B8",
    "lightgray": "#F1F5F9",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.figsize": (10, 6),
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


def compute_rqa_per_stock(df, rqa_cfg, rqa_cols):
    all_rqa = []
    tickers = sorted(df["ticker"].unique())
    sample_tickers = tickers[:min(10, len(tickers))]
    sample_blocks = [df[df["ticker"] == t][list(rqa_cols)].to_numpy(dtype=float)
                     for t in sample_tickers]
    combined_sample = np.concatenate(sample_blocks, axis=0)
    eps_fixed = estimate_epsilon_from_train(combined_sample, rqa_cfg)

    for ticker in tickers:
        mask = df["ticker"] == ticker
        df_ticker = df.loc[mask].reset_index(drop=True)
        rqa_feats = rqa_features_rolling(df_ticker, rqa_cols, rqa_cfg,
                                         prefix="rqa", eps_fixed=eps_fixed)
        rqa_feats.index = df.index[mask]
        all_rqa.append(rqa_feats)
    return pd.concat(all_rqa).sort_index()


def compute_beta_rqa_per_stock(df, beta_cfg, cols):
    all_feats = []
    for ticker in sorted(df["ticker"].unique()):
        mask = df["ticker"] == ticker
        df_ticker = df.loc[mask].reset_index(drop=True)
        feats = beta_rqa_features_rolling(df_ticker, cols, beta_cfg, prefix="beta_rqa")
        feats.index = df.index[mask]
        all_feats.append(feats)
    return pd.concat(all_feats).sort_index()


def build_xy_per_stock(df, X_df):
    all_X, all_y = [], []
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
    return np.concatenate(all_X), np.concatenate(all_y)


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


def train_rf_and_get_importances(
    Xtr, ytr, Xva, yva, Xte, yte,
    feature_names, logger, model_name="RF",
):
    """Train RF, return Gini importances + permutation importances + metrics."""
    rf = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5, n_jobs=-1,
        class_weight="balanced_subsample", random_state=42,
    )
    logger.info(f"  Training {model_name} ({Xtr.shape[1]} features, {len(ytr):,} samples)...")
    t0 = time.time()
    rf.fit(Xtr, ytr)
    fit_time = time.time() - t0
    logger.info(f"  Fit time: {fit_time:.1f}s")

    p_va = rf.predict_proba(Xva)[:, 1]
    thr = best_threshold(yva, p_va)

    p_te = rf.predict_proba(Xte)[:, 1]
    y_pred = (p_te >= thr).astype(int)
    auc = roc_auc_score(yte, p_te)
    f1 = f1_score(yte, y_pred, zero_division=0)

    gini_imp = rf.feature_importances_

    logger.info(f"  Computing permutation importances (10 repeats, may take a few minutes)...")
    t0 = time.time()
    perm_result = permutation_importance(
        rf, Xte, yte, n_repeats=10, random_state=42,
        scoring="roc_auc", n_jobs=-1,
    )
    perm_time = time.time() - t0
    logger.info(f"  Permutation importance time: {perm_time:.1f}s")

    perm_imp_mean = perm_result.importances_mean
    perm_imp_std = perm_result.importances_std

    return {
        "model": rf,
        "auc": auc,
        "f1": f1,
        "threshold": thr,
        "gini": dict(zip(feature_names, gini_imp)),
        "perm_mean": dict(zip(feature_names, perm_imp_mean)),
        "perm_std": dict(zip(feature_names, perm_imp_std)),
        "feature_names": feature_names,
    }


def run_ablation(
    rf_model, Xtr, ytr, Xva, yva, Xte, yte,
    feature_names, rqa_feature_indices, rqa_feature_labels,
    logger,
):
    """
    Leave-one-out ablation: for each RQA feature, remove it and retrain.
    Returns dict mapping feature label -> AUC when that feature is removed.
    """
    p_va = rf_model.predict_proba(Xva)[:, 1]
    thr = best_threshold(yva, p_va)
    p_te = rf_model.predict_proba(Xte)[:, 1]
    baseline_auc = roc_auc_score(yte, p_te)
    baseline_f1 = f1_score(yte, (p_te >= thr).astype(int), zero_division=0)

    ablation_results = {"baseline": {"auc": baseline_auc, "f1": baseline_f1}}

    for idx, label in zip(rqa_feature_indices, rqa_feature_labels):
        keep = [i for i in range(Xtr.shape[1]) if i != idx]
        Xtr_abl = Xtr[:, keep]
        Xva_abl = Xva[:, keep]
        Xte_abl = Xte[:, keep]

        rf_abl = RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, n_jobs=-1,
            class_weight="balanced_subsample", random_state=42,
        )
        rf_abl.fit(Xtr_abl, ytr)

        p_va_abl = rf_abl.predict_proba(Xva_abl)[:, 1]
        thr_abl = best_threshold(yva, p_va_abl)
        p_te_abl = rf_abl.predict_proba(Xte_abl)[:, 1]
        auc_abl = roc_auc_score(yte, p_te_abl)
        f1_abl = f1_score(yte, (p_te_abl >= thr_abl).astype(int), zero_division=0)

        ablation_results[label] = {
            "auc": auc_abl,
            "f1": f1_abl,
            "delta_auc": auc_abl - baseline_auc,
            "delta_f1": f1_abl - baseline_f1,
        }
        logger.info(f"    Drop {label:>20s}: AUC={auc_abl:.6f} (Δ={auc_abl - baseline_auc:+.6f})")

    std_only_cols = [i for i in range(Xtr.shape[1]) if i not in rqa_feature_indices]
    rf_std = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5, n_jobs=-1,
        class_weight="balanced_subsample", random_state=42,
    )
    rf_std.fit(Xtr[:, std_only_cols], ytr)
    p_va_std = rf_std.predict_proba(Xva[:, std_only_cols])[:, 1]
    thr_std = best_threshold(yva, p_va_std)
    p_te_std = rf_std.predict_proba(Xte[:, std_only_cols])[:, 1]
    auc_std = roc_auc_score(yte, p_te_std)
    f1_std = f1_score(yte, (p_te_std >= thr_std).astype(int), zero_division=0)
    ablation_results["ALL_RQA_removed"] = {
        "auc": auc_std,
        "f1": f1_std,
        "delta_auc": auc_std - baseline_auc,
        "delta_f1": f1_std - baseline_f1,
    }
    logger.info(f"    Drop ALL RQA features: AUC={auc_std:.6f} (Δ={auc_std - baseline_auc:+.6f})")

    return ablation_results


def plot_feature_importances(results, model_label, fig_name):
    """Combined Gini + permutation importance bar chart."""
    feature_names = results["feature_names"]
    gini = np.array([results["gini"][f] for f in feature_names])
    perm = np.array([results["perm_mean"][f] for f in feature_names])
    perm_std = np.array([results["perm_std"][f] for f in feature_names])

    order = np.argsort(perm)[::-1]
    top_n = min(30, len(feature_names))  
    order = order[:top_n]

    names_sorted = [feature_names[i] for i in order]
    gini_sorted = gini[order]
    perm_sorted = perm[order]
    perm_std_sorted = perm_std[order]

    colors = []
    for name in names_sorted:
        if "rqa" in name.lower() or "beta" in name.lower():
            colors.append(COLORS["green"])
        else:
            colors.append(COLORS["accent"])

    fig, axes = plt.subplots(1, 2, figsize=(16, max(8, top_n * 0.35)))

    ax = axes[0]
    y_pos = np.arange(top_n)
    ax.barh(y_pos, gini_sorted, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_sorted, fontsize=8)
    ax.set_xlabel("Gini Importance (MDI)")
    ax.set_title(f"{model_label}: Gini Importance")
    ax.invert_yaxis()

    ax = axes[1]
    ax.barh(y_pos, perm_sorted, xerr=perm_std_sorted, color=colors,
            edgecolor="white", height=0.7, capsize=2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_sorted, fontsize=8)
    ax.set_xlabel("Permutation Importance (ΔAUC)")
    ax.set_title(f"{model_label}: Permutation Importance")
    ax.invert_yaxis()

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["accent"], label="Standard features"),
        Patch(facecolor=COLORS["green"], label="RQA features"),
    ]
    axes[1].legend(handles=legend_elements, loc="lower right")

    fig.suptitle(f"Dataset 3: Feature Importance Analysis — {model_label}\n"
                 f"(AUC={results['auc']:.4f}, {len(feature_names)} features)",
                 fontsize=14)
    fig.tight_layout()
    savefig(fig, fig_name)


def plot_ablation(ablation_results, fig_name, model_label="RF Std+RQA"):
    """Bar chart showing AUC change when each RQA feature is removed."""
    baseline_auc = ablation_results["baseline"]["auc"]

    features = [k for k in ablation_results if k not in ("baseline", "ALL_RQA_removed")]
    deltas = [ablation_results[k]["delta_auc"] for k in features]

    features.append("ALL RQA removed")
    deltas.append(ablation_results["ALL_RQA_removed"]["delta_auc"])

    order = np.argsort(deltas)
    features_sorted = [features[i] for i in order]
    deltas_sorted = [deltas[i] for i in order]

    colors = [COLORS["red"] if d < 0 else COLORS["green"] for d in deltas_sorted]
    for i, f in enumerate(features_sorted):
        if f == "ALL RQA removed":
            colors[i] = COLORS["orange"]

    fig, ax = plt.subplots(figsize=(10, max(5, len(features_sorted) * 0.4)))
    y_pos = np.arange(len(features_sorted))
    ax.barh(y_pos, deltas_sorted, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features_sorted, fontsize=9)
    ax.set_xlabel("ΔAUC (vs full model)")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.invert_yaxis()

    for i, (d, f) in enumerate(zip(deltas_sorted, features_sorted)):
        ax.text(d + (0.00002 if d >= 0 else -0.00002), i,
                f"{d:+.5f}", va="center", ha="left" if d >= 0 else "right",
                fontsize=8)

    ax.set_title(f"Dataset 3: Leave-One-Out Ablation — {model_label}\n"
                 f"Baseline AUC = {baseline_auc:.6f} | "
                 f"Negative Δ = removing hurts (feature is useful)",
                 fontsize=12)

    fig.tight_layout()
    savefig(fig, fig_name)


def plot_feature_group_contribution(results_std, results_rqa, results_beta, fig_name):
    """Bar chart comparing std-only, std+rqa, std+beta model AUCs."""
    models = []
    aucs = []
    colors_list = []

    models.append("RF Std-only")
    aucs.append(results_std["auc"])
    colors_list.append(COLORS["accent"])

    if results_rqa is not None:
        models.append("RF Std+RQA")
        aucs.append(results_rqa["auc"])
        colors_list.append(COLORS["green"])

    if results_beta is not None:
        models.append("RF Std+β-RQA")
        aucs.append(results_beta["auc"])
        colors_list.append(COLORS["purple"])

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(range(len(models)), aucs, color=colors_list, edgecolor="white", height=0.5)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel("ROC-AUC")
    ax.set_xlim(min(aucs) - 0.002, max(aucs) + 0.002)
    ax.invert_yaxis()

    for bar, auc in zip(bars, aucs):
        ax.text(auc + 0.0002, bar.get_y() + bar.get_height() / 2,
                f"{auc:.5f}", va="center", fontsize=10)

    ax.set_title("Dataset 3: Feature Group Contribution (pooled AUC)", fontsize=13)
    fig.tight_layout()
    savefig(fig, fig_name)



def main():
    logger = get_logger()
    parser = argparse.ArgumentParser(description="Feature importance analysis for D3")
    parser.add_argument("--skip_beta", action="store_true",
                        help="Skip β-RQA analysis (faster)")
    args = parser.parse_args()

    print("=" * 70)
    print("  FEATURE IMPORTANCE ANALYSIS — DATASET 3")
    print("  Answers: Why do RQA features help? Which ones matter most?")
    print("=" * 70)

    logger.info("Loading D3 splits...")
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")
    full = pd.concat([train, val, test], ignore_index=True)
    n_tr, n_va = len(train), len(val)
    logger.info(f"Train: {n_tr:,}, Val: {n_va:,}, Test: {len(test):,}")

    logger.info("Building standard feature arrays...")
    X_std_tr, y_tr = build_xy_per_stock(train, train[STD_FEATURE_COLS].copy())
    X_std_va, y_va = build_xy_per_stock(val, val[STD_FEATURE_COLS].copy())
    X_std_te, y_te = build_xy_per_stock(test, test[STD_FEATURE_COLS].copy())
    logger.info(f"Std: {X_std_tr.shape[1]} features, {len(y_tr):,} train samples")

    logger.info("Computing RQA features (default config, step=20)...")
    rqa_cfg = RQAConfig(window=60, step=20, recurrence_rate=0.1,
                        embed=EmbeddingConfig(m=4, tau=2), mode="joint")
    rqa_cols = ("log_return", "rv")

    t0 = time.time()
    X_rqa_all = compute_rqa_per_stock(full, rqa_cfg, rqa_cols)
    rqa_time = time.time() - t0
    logger.info(f"RQA computation: {rqa_time:.1f}s, {X_rqa_all.shape[1]} features")

    X_rqa_tr_df = X_rqa_all.iloc[:n_tr].reset_index(drop=True)
    X_rqa_va_df = X_rqa_all.iloc[n_tr:n_tr + n_va].reset_index(drop=True)
    X_rqa_te_df = X_rqa_all.iloc[n_tr + n_va:].reset_index(drop=True)

    X_rqa_tr, y_tr2 = build_xy_per_stock(train, X_rqa_tr_df)
    X_rqa_va, y_va2 = build_xy_per_stock(val, X_rqa_va_df)
    X_rqa_te, y_te2 = build_xy_per_stock(test, X_rqa_te_df)

    rqa_col_names = list(X_rqa_all.columns)

    ntr = min(len(y_tr), len(y_tr2))
    nva = min(len(y_va), len(y_va2))
    nte = min(len(y_te), len(y_te2))
    X_std_tr_a, y_tr = X_std_tr[:ntr], y_tr[:ntr]
    X_rqa_tr = X_rqa_tr[:ntr]
    X_std_va_a, y_va = X_std_va[:nva], y_va[:nva]
    X_rqa_va = X_rqa_va[:nva]
    X_std_te_a, y_te = X_std_te[:nte], y_te[:nte]
    X_rqa_te = X_rqa_te[:nte]

    X_comb_tr = np.concatenate([X_std_tr_a, X_rqa_tr], axis=1)
    X_comb_va = np.concatenate([X_std_va_a, X_rqa_va], axis=1)
    X_comb_te = np.concatenate([X_std_te_a, X_rqa_te], axis=1)
    comb_feature_names = STD_FEATURE_COLS + rqa_col_names

    logger.info(f"Combined: {len(comb_feature_names)} features = "
                f"{len(STD_FEATURE_COLS)} std + {len(rqa_col_names)} RQA")

    print("\n--- A. RF Standard-only baseline ---")
    results_std = train_rf_and_get_importances(
        X_std_tr_a, y_tr, X_std_va_a, y_va, X_std_te_a, y_te,
        STD_FEATURE_COLS, logger, model_name="RF Std-only",
    )
    logger.info(f"RF Std-only: AUC={results_std['auc']:.6f}, F1={results_std['f1']:.4f}")

    print("\n--- B. RF Std+RQA (Gini + Permutation importances) ---")
    results_rqa = train_rf_and_get_importances(
        X_comb_tr, y_tr, X_comb_va, y_va, X_comb_te, y_te,
        comb_feature_names, logger, model_name="RF Std+RQA",
    )
    logger.info(f"RF Std+RQA: AUC={results_rqa['auc']:.6f}, F1={results_rqa['f1']:.4f}")

    print("\n--- C. Leave-one-out ablation of RQA features ---")
    rqa_indices = list(range(len(STD_FEATURE_COLS), len(comb_feature_names)))
    ablation_rqa = run_ablation(
        results_rqa["model"], X_comb_tr, y_tr, X_comb_va, y_va, X_comb_te, y_te,
        comb_feature_names, rqa_indices, rqa_col_names, logger,
    )

    results_beta = None
    ablation_beta = None
    if not args.skip_beta:
        print("\n--- D. RF Std+β-RQA (β=2.0) ---")
        beta_cfg = BetaRQAConfig(window=60, step=20, recurrence_rate=0.1,
                                 embed=EmbeddingConfig(m=4, tau=2),
                                 beta=2.0, transform="minmax")
        logger.info("Computing β-RQA features...")
        t0 = time.time()
        X_beta_all = compute_beta_rqa_per_stock(full, beta_cfg, ("log_return",))
        logger.info(f"β-RQA computation: {time.time() - t0:.1f}s")

        X_beta_tr_df = X_beta_all.iloc[:n_tr].reset_index(drop=True)
        X_beta_va_df = X_beta_all.iloc[n_tr:n_tr + n_va].reset_index(drop=True)
        X_beta_te_df = X_beta_all.iloc[n_tr + n_va:].reset_index(drop=True)

        X_beta_tr, y_tr3 = build_xy_per_stock(train, X_beta_tr_df)
        X_beta_va, y_va3 = build_xy_per_stock(val, X_beta_va_df)
        X_beta_te, y_te3 = build_xy_per_stock(test, X_beta_te_df)

        beta_col_names = list(X_beta_all.columns)

        nb_tr = min(ntr, len(y_tr3))
        nb_va = min(nva, len(y_va3))
        nb_te = min(nte, len(y_te3))

        X_comb_beta_tr = np.concatenate([X_std_tr_a[:nb_tr], X_beta_tr[:nb_tr]], axis=1)
        X_comb_beta_va = np.concatenate([X_std_va_a[:nb_va], X_beta_va[:nb_va]], axis=1)
        X_comb_beta_te = np.concatenate([X_std_te_a[:nb_te], X_beta_te[:nb_te]], axis=1)
        y_tr_b, y_va_b, y_te_b = y_tr[:nb_tr], y_va[:nb_va], y_te[:nb_te]
        beta_comb_names = STD_FEATURE_COLS + beta_col_names

        results_beta = train_rf_and_get_importances(
            X_comb_beta_tr, y_tr_b, X_comb_beta_va, y_va_b, X_comb_beta_te, y_te_b,
            beta_comb_names, logger, model_name="RF Std+β-RQA",
        )
        logger.info(f"RF Std+β-RQA: AUC={results_beta['auc']:.6f}, F1={results_beta['f1']:.4f}")

        print("\n--- E. Leave-one-out ablation of β-RQA features ---")
        beta_indices = list(range(len(STD_FEATURE_COLS), len(beta_comb_names)))
        ablation_beta = run_ablation(
            results_beta["model"], X_comb_beta_tr, y_tr_b,
            X_comb_beta_va, y_va_b, X_comb_beta_te, y_te_b,
            beta_comb_names, beta_indices, beta_col_names, logger,
        )

    print("\n--- Generating figures ---")
    plot_feature_importances(results_rqa, "RF Std+RQA", "6_1_feature_importance_rqa_d3")

    if results_beta is not None:
        plot_feature_importances(results_beta, "RF Std+β-RQA", "6_1b_feature_importance_beta_d3")

    plot_ablation(ablation_rqa, "6_2_ablation_rqa_d3", "RF Std+RQA")

    if ablation_beta is not None:
        plot_ablation(ablation_beta, "6_2b_ablation_beta_d3", "RF Std+β-RQA")

    plot_feature_group_contribution(results_std, results_rqa, results_beta,
                                    "6_3_feature_group_contribution_d3")

    print("\n--- Saving text results ---")
    out_dir = repo_root() / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "feature_importance_d3.txt"

    lines = []
    lines.append("=" * 80)
    lines.append("  FEATURE IMPORTANCE ANALYSIS — DATASET 3")
    lines.append("  Answers: Why do RQA features help? Which ones matter most?")
    lines.append("=" * 80)

    lines.append(f"\n\nA. MODEL PERFORMANCE SUMMARY\n")
    lines.append(f"  RF Std-only:   AUC = {results_std['auc']:.6f}  F1 = {results_std['f1']:.4f}")
    lines.append(f"  RF Std+RQA:    AUC = {results_rqa['auc']:.6f}  F1 = {results_rqa['f1']:.4f}")
    if results_beta:
        lines.append(f"  RF Std+β-RQA:  AUC = {results_beta['auc']:.6f}  F1 = {results_beta['f1']:.4f}")
    lines.append(f"  ΔAUC (RQA):    {results_rqa['auc'] - results_std['auc']:+.6f}")
    if results_beta:
        lines.append(f"  ΔAUC (β-RQA):  {results_beta['auc'] - results_std['auc']:+.6f}")

    lines.append(f"\n\nB. GINI IMPORTANCE — RF Std+RQA (top 25 features)\n")
    gini_sorted = sorted(results_rqa["gini"].items(), key=lambda x: x[1], reverse=True)
    lines.append(f"  {'Feature':<35} {'Gini':>10} {'Type':>10}")
    lines.append("  " + "-" * 57)
    for f, v in gini_sorted[:25]:
        ftype = "RQA" if "rqa" in f.lower() else "Standard"
        lines.append(f"  {f:<35} {v:>10.6f} {ftype:>10}")

    lines.append(f"\n\nC. PERMUTATION IMPORTANCE — RF Std+RQA (top 25 features)\n")
    perm_sorted = sorted(results_rqa["perm_mean"].items(), key=lambda x: x[1], reverse=True)
    lines.append(f"  {'Feature':<35} {'Perm ΔAUC':>12} {'Std':>10} {'Type':>10}")
    lines.append("  " + "-" * 69)
    for f, v in perm_sorted[:25]:
        ftype = "RQA" if "rqa" in f.lower() else "Standard"
        std = results_rqa["perm_std"][f]
        lines.append(f"  {f:<35} {v:>12.6f} {std:>10.6f} {ftype:>10}")

    lines.append(f"\n\nD. RQA FEATURE RANKING (by permutation importance)\n")
    rqa_feats_only = [(f, results_rqa["perm_mean"][f], results_rqa["perm_std"][f])
                      for f in rqa_col_names]
    rqa_feats_only.sort(key=lambda x: x[1], reverse=True)
    lines.append(f"  {'RQA Feature':<35} {'Perm ΔAUC':>12} {'Std':>10}")
    lines.append("  " + "-" * 59)
    for f, v, s in rqa_feats_only:
        lines.append(f"  {f:<35} {v:>12.6f} {s:>10.6f}")

    lines.append(f"\n\nE. ABLATION — Drop each RQA feature from RF Std+RQA\n")
    lines.append(f"  Baseline AUC: {ablation_rqa['baseline']['auc']:.6f}")
    lines.append(f"  {'Dropped Feature':<35} {'AUC':>10} {'ΔAUC':>12} {'ΔF1':>10}")
    lines.append("  " + "-" * 69)
    for k in sorted(ablation_rqa.keys()):
        if k == "baseline":
            continue
        v = ablation_rqa[k]
        lines.append(f"  {k:<35} {v['auc']:>10.6f} {v.get('delta_auc', 0):>+12.6f} "
                     f"{v.get('delta_f1', 0):>+10.4f}")

    if results_beta and ablation_beta:
        lines.append(f"\n\nF. β-RQA FEATURE RANKING (by permutation importance)\n")
        beta_feats_only = [(f, results_beta["perm_mean"][f], results_beta["perm_std"][f])
                           for f in beta_col_names]
        beta_feats_only.sort(key=lambda x: x[1], reverse=True)
        lines.append(f"  {'β-RQA Feature':<35} {'Perm ΔAUC':>12} {'Std':>10}")
        lines.append("  " + "-" * 59)
        for f, v, s in beta_feats_only:
            lines.append(f"  {f:<35} {v:>12.6f} {s:>10.6f}")

        lines.append(f"\n\nG. ABLATION — Drop each β-RQA feature from RF Std+β-RQA\n")
        lines.append(f"  Baseline AUC: {ablation_beta['baseline']['auc']:.6f}")
        lines.append(f"  {'Dropped Feature':<35} {'AUC':>10} {'ΔAUC':>12}")
        lines.append("  " + "-" * 59)
        for k in sorted(ablation_beta.keys()):
            if k == "baseline":
                continue
            v = ablation_beta[k]
            lines.append(f"  {k:<35} {v['auc']:>10.6f} {v.get('delta_auc', 0):>+12.6f}")

    lines.append(f"\n\n{'=' * 80}")
    lines.append(f"  INTERPRETATION (for thesis discussion)")
    lines.append(f"{'=' * 80}\n")
    lines.append("  The feature importance analysis reveals which RQA measures contribute")
    lines.append("  to the statistically significant improvement observed in Dataset 3.")
    lines.append("  Examine the permutation importance and ablation results above to")
    lines.append("  identify: (1) which RQA features have the highest individual impact,")
    lines.append("  (2) whether any can be removed without loss, and (3) whether the")
    lines.append("  improvement is concentrated in a few measures or distributed across all.")
    lines.append("")
    lines.append("  Key questions to answer from these results:")
    lines.append("  - Which RQA feature has the highest permutation importance?")
    lines.append("  - Does removing any single RQA feature cause a notable AUC drop?")
    lines.append("  - Is the total RQA contribution (ALL_RQA_removed ΔAUC) larger than")
    lines.append("    any individual feature's contribution? (If so, features are complementary.)")
    lines.append("  - Do the Dreesen 2025 horizontal measures (LAM_h, TT_h, ΔLAM, ΔTT)")
    lines.append("    contribute meaningfully in the β-RQA model?")

    results_text = "\n".join(lines)
    out_path.write_text(results_text, encoding="utf-8")
    logger.info(f"\nResults saved to {out_path}")
    print(results_text)

    print("\n" + "=" * 70)
    print("  DONE — Check figures/ and results/ directories")
    print("=" * 70)


if __name__ == "__main__":
    main()