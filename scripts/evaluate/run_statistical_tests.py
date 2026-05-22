"""
run_statistical_tests.py — Statistical significance testing for all datasets.

For each dataset, re-runs key model comparisons, saves predictions,
then computes:
  - Bootstrap 95% confidence intervals for AUC and F1
  - Paired bootstrap test for model comparisons (D1/D2)
  - Per-stock metrics with mean ± std and Wilcoxon signed-rank test (D3)

Usage:
  python scripts/run_statistical_tests.py --dataset 1
  python scripts/run_statistical_tests.py --dataset 2
  python scripts/run_statistical_tests.py --dataset 3

Output: results/statistical_tests_d{N}.txt
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scalable_rqa_volatility.logging_utils import get_logger
from scalable_rqa_volatility.recurrence.beta_rqa import BetaRQAConfig, beta_rqa_features_rolling
from scalable_rqa_volatility.recurrence.embeddings import EmbeddingConfig
from scalable_rqa_volatility.recurrence.rqa import RQAConfig, rqa_features_rolling, estimate_epsilon_from_train
from scalable_rqa_volatility.volatility.features_standard import StandardFeatureConfig, standard_features


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_split(dataset: int, name: str) -> pd.DataFrame:
    return pd.read_parquet(repo_root() / "data" / "processed" / f"dataset{dataset}_{name}.parquet")


def bootstrap_metric(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray,
                     n_boot: int = 2000, seed: int = 42) -> dict:
    """Bootstrap 95% CI for AUC and F1."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aucs, f1s = [], []

    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yt, ys, yp = y_true[idx], y_score[idx], y_pred[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(yt, ys))
        except ValueError:
            pass
        f1s.append(f1_score(yt, yp, zero_division=0))

    aucs, f1s = np.array(aucs), np.array(f1s)
    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "auc_ci_lo": float(np.percentile(aucs, 2.5)),
        "auc_ci_hi": float(np.percentile(aucs, 97.5)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "f1_ci_lo": float(np.percentile(f1s, 2.5)),
        "f1_ci_hi": float(np.percentile(f1s, 97.5)),
    }


def paired_bootstrap_test(y_true: np.ndarray,
                          score_a: np.ndarray, score_b: np.ndarray,
                          n_boot: int = 2000, seed: int = 42) -> dict:
    """Paired bootstrap test: is model B better than model A (by AUC)?"""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    diffs = []

    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            auc_a = roc_auc_score(yt, score_a[idx])
            auc_b = roc_auc_score(yt, score_b[idx])
            diffs.append(auc_b - auc_a)
        except ValueError:
            pass

    diffs = np.array(diffs)
    p_value = float(np.mean(diffs <= 0))  

    return {
        "mean_auc_diff": float(np.mean(diffs)),
        "std_auc_diff": float(np.std(diffs)),
        "ci_lo": float(np.percentile(diffs, 2.5)),
        "ci_hi": float(np.percentile(diffs, 97.5)),
        "p_value_b_better": float(p_value),
    }


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
        f = f1_fast(y_true, y_pred)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    if best_f1 < 0:
        best_t = float(np.quantile(prob, 1.0 - target))
    return best_t


def build_xy_d12(df, X_df, label_col="regime"):
    """Build X, y for D1/D2 (predict next-step regime)."""
    y = df[label_col].shift(-1)
    mask = y.notna().to_numpy()
    X2 = X_df.iloc[:len(mask)].copy().iloc[mask].replace([np.inf, -np.inf], np.nan)
    y2 = y.iloc[:len(mask)].iloc[mask].astype(int)
    keep = X2.notna().all(axis=1).to_numpy()
    return X2.iloc[keep].to_numpy(dtype=float), y2.iloc[keep].to_numpy()


def run_d12(dataset: int, logger):
    logger.info(f"=== DATASET {dataset} STATISTICAL TESTS ===")

    train = load_split(dataset, "train").reset_index(drop=True)
    val = load_split(dataset, "val").reset_index(drop=True)
    test = load_split(dataset, "test").reset_index(drop=True)
    full = pd.concat([train, val, test], ignore_index=True)
    n_tr, n_va, n_te = len(train), len(val), len(test)

    logger.info(f"Train: {n_tr}, Val: {n_va}, Test: {n_te}")

    std_cfg = StandardFeatureConfig(windows=(5, 22, 60))
    X_std_full = standard_features(full, std_cfg)
    X_std_tr = X_std_full.iloc[:n_tr].reset_index(drop=True)
    X_std_va = X_std_full.iloc[n_tr:n_tr+n_va].reset_index(drop=True)
    X_std_te = X_std_full.iloc[n_tr+n_va:].reset_index(drop=True)

    Xtr_std, ytr = build_xy_d12(train, X_std_tr)
    Xva_std, yva = build_xy_d12(val, X_std_va)
    Xte_std, yte = build_xy_d12(test, X_std_te)

    logger.info(f"Std features: {Xtr_std.shape[1]}, Test samples: {len(yte)}")

    rqa_mode = "per_series" if dataset == 2 else "joint"
    rqa_cfg = RQAConfig(window=90, step=5, recurrence_rate=0.1,
                        embed=EmbeddingConfig(m=4, tau=2), mode=rqa_mode)
    rqa_cols = ("log_return", "rv")
    eps_fixed = estimate_epsilon_from_train(
        full.iloc[:n_tr][list(rqa_cols)].to_numpy(dtype=float), rqa_cfg)
    X_rqa_full = rqa_features_rolling(full, rqa_cols, rqa_cfg, prefix="rqa", eps_fixed=eps_fixed)
    X_rqa_tr = X_rqa_full.iloc[:n_tr].reset_index(drop=True)
    X_rqa_va = X_rqa_full.iloc[n_tr:n_tr+n_va].reset_index(drop=True)
    X_rqa_te = X_rqa_full.iloc[n_tr+n_va:].reset_index(drop=True)

    Xtr_rqa, ytr_rqa = build_xy_d12(train, X_rqa_tr)
    Xva_rqa, yva_rqa = build_xy_d12(val, X_rqa_va)
    Xte_rqa, yte_rqa = build_xy_d12(test, X_rqa_te)

    def align(*arrays_and_ys):
        n = min(len(a) for a in arrays_and_ys)
        return tuple(a[:n] for a in arrays_and_ys)

    Xtr_std_a, Xtr_rqa_a, ytr_a = align(Xtr_std, Xtr_rqa, ytr)
    Xva_std_a, Xva_rqa_a, yva_a = align(Xva_std, Xva_rqa, yva)
    Xte_std_a, Xte_rqa_a, yte_a = align(Xte_std, Xte_rqa, yte)

    Xtr_comb_rqa = np.concatenate([Xtr_std_a, Xtr_rqa_a], axis=1)
    Xva_comb_rqa = np.concatenate([Xva_std_a, Xva_rqa_a], axis=1)
    Xte_comb_rqa = np.concatenate([Xte_std_a, Xte_rqa_a], axis=1)

    beta_values = [4.0, 5.0] if dataset == 2 else [1.5]
    beta_runs = {}

    for beta_value in beta_values:
        logger.info(f"Computing β-RQA features (β={beta_value})...")
        beta_cfg = BetaRQAConfig(window=90, step=5, recurrence_rate=0.1,
                                 embed=EmbeddingConfig(m=4, tau=2),
                                 beta=beta_value, transform="minmax")
        X_beta_full = beta_rqa_features_rolling(
            full, ("log_return",), beta_cfg, prefix="beta_rqa")
        X_beta_tr = X_beta_full.iloc[:n_tr].reset_index(drop=True)
        X_beta_va = X_beta_full.iloc[n_tr:n_tr+n_va].reset_index(drop=True)
        X_beta_te = X_beta_full.iloc[n_tr+n_va:].reset_index(drop=True)

        Xtr_beta, ytr_beta = build_xy_d12(train, X_beta_tr)
        Xva_beta, yva_beta = build_xy_d12(val, X_beta_va)
        Xte_beta, yte_beta = build_xy_d12(test, X_beta_te)

        Xtr_std_b, Xtr_beta_b, ytr_b = align(Xtr_std, Xtr_beta, ytr)
        Xva_std_b, Xva_beta_b, yva_b = align(Xva_std, Xva_beta, yva)
        Xte_std_b, Xte_beta_b, yte_b = align(Xte_std, Xte_beta, yte)

        beta_runs[beta_value] = {
            "Xtr": np.concatenate([Xtr_std_b, Xtr_beta_b], axis=1),
            "Xva": np.concatenate([Xva_std_b, Xva_beta_b], axis=1),
            "Xte": np.concatenate([Xte_std_b, Xte_beta_b], axis=1),
            "ytr": ytr_b,
            "yva": yva_b,
            "yte": yte_b,
        }

    predictions = {}
    display_names = {}

    def train_and_predict(name, model, Xtr, ytr, Xva, yva, Xte, yte, display_name=None):
        model.fit(Xtr, ytr)
        p_va = model.predict_proba(Xva)[:, 1]
        thr = best_threshold(yva, p_va)
        p_te = model.predict_proba(Xte)[:, 1]
        pred = (p_te >= thr).astype(int)
        auc = roc_auc_score(yte, p_te)
        f1 = f1_score(yte, pred, zero_division=0)
        predictions[name] = {"y_true": yte, "y_score": p_te, "y_pred": pred,
                             "auc": auc, "f1": f1, "threshold": thr}
        display_names[name] = display_name or name
        logger.info(f"  {display_names[name]}: AUC={auc:.4f}, F1={f1:.4f}")

    rf = lambda: RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                        n_jobs=-1, class_weight="balanced_subsample", random_state=42)
    lr = lambda: Pipeline([("scaler", StandardScaler()),
                           ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))])

    logger.info("Training models...")
    train_and_predict("rf_std", rf(), Xtr_std, ytr, Xva_std, yva, Xte_std, yte)
    train_and_predict("rf_std_rqa", rf(), Xtr_comb_rqa, ytr_a, Xva_comb_rqa, yva_a, Xte_comb_rqa, yte_a)

    for beta_value, run in beta_runs.items():
        suffix = str(beta_value).replace(".", "_")
        name = "rf_std_beta" if len(beta_values) == 1 else f"rf_std_beta_b{suffix}"
        label = "rf_std_beta" if len(beta_values) == 1 else f"rf_std_beta (β={beta_value})"
        train_and_predict(name, rf(), run["Xtr"], run["ytr"], run["Xva"], run["yva"], run["Xte"], run["yte"], label)

    train_and_predict("lr_std", lr(), Xtr_std, ytr, Xva_std, yva, Xte_std, yte)
    train_and_predict("lr_std_rqa", lr(), Xtr_comb_rqa, ytr_a, Xva_comb_rqa, yva_a, Xte_comb_rqa, yte_a)

    for beta_value, run in beta_runs.items():
        suffix = str(beta_value).replace(".", "_")
        name = "lr_std_beta" if len(beta_values) == 1 else f"lr_std_beta_b{suffix}"
        label = "lr_std_beta" if len(beta_values) == 1 else f"lr_std_beta (β={beta_value})"
        train_and_predict(name, lr(), run["Xtr"], run["ytr"], run["Xva"], run["yva"], run["Xte"], run["yte"], label)

    logger.info("Training HAR-RV...")
    har_feats = pd.DataFrame({
        "rv_d": full["rv"].shift(1),
        "rv_w": full["rv"].rolling(5).mean().shift(1),
        "rv_m": full["rv"].rolling(22).mean().shift(1),
    })
    har_y = full["regime"].shift(-1)
    har_rv_next = full["rv"].shift(-1)

    mask = har_feats.notna().all(axis=1) & har_y.notna() & har_rv_next.notna()
    har_X = har_feats[mask].to_numpy(dtype=float)
    har_regime = har_y[mask].astype(int).to_numpy()
    har_rv = har_rv_next[mask].astype(float).to_numpy()
    har_idx = np.where(mask.to_numpy())[0]

    tr_mask = har_idx < n_tr
    va_mask = (har_idx >= n_tr) & (har_idx < n_tr + n_va)
    te_mask = har_idx >= n_tr + n_va

    har_model = LinearRegression()
    har_model.fit(har_X[tr_mask], har_rv[tr_mask])
    har_pred_va = har_model.predict(har_X[va_mask])
    har_pred_te = har_model.predict(har_X[te_mask])
    har_yte = har_regime[te_mask]

    har_b = best_threshold(har_regime[va_mask], har_pred_va)
    har_pred_class = (har_pred_te >= har_b).astype(int)
    predictions["har_rv"] = {
        "y_true": har_yte, "y_score": har_pred_te, "y_pred": har_pred_class,
        "auc": roc_auc_score(har_yte, har_pred_te),
        "f1": f1_score(har_yte, har_pred_class, zero_division=0),
    }
    display_names["har_rv"] = "har_rv"
    logger.info(f"  har_rv: AUC={predictions['har_rv']['auc']:.4f}, F1={predictions['har_rv']['f1']:.4f}")

    logger.info("Computing bootstrap confidence intervals (2000 resamples)...")
    results_lines = []
    results_lines.append(f"{'='*80}")
    results_lines.append(f"  DATASET {dataset} — STATISTICAL SIGNIFICANCE TESTS")
    results_lines.append(f"  Bootstrap: 2000 resamples, 95% confidence intervals")
    if dataset == 2:
        results_lines.append(f"  β-RQA rows include β=4.0 and β=5.0")
    results_lines.append(f"{'='*80}\n")

    results_lines.append("A. BOOTSTRAP CONFIDENCE INTERVALS\n")
    results_lines.append(f"{'Model':<30} {'AUC':>8} {'AUC 95% CI':>20} {'F1':>8} {'F1 95% CI':>20}")
    results_lines.append("-" * 90)

    model_order = ["rf_std", "rf_std_rqa"]
    for beta_value in beta_values:
        suffix = str(beta_value).replace(".", "_")
        model_order.append("rf_std_beta" if len(beta_values) == 1 else f"rf_std_beta_b{suffix}")
    model_order += ["lr_std", "lr_std_rqa"]
    for beta_value in beta_values:
        suffix = str(beta_value).replace(".", "_")
        model_order.append("lr_std_beta" if len(beta_values) == 1 else f"lr_std_beta_b{suffix}")
    model_order.append("har_rv")

    for name in model_order:
        p = predictions[name]
        bs = bootstrap_metric(p["y_true"], p["y_score"], p["y_pred"])
        results_lines.append(
            f"{display_names[name]:<30} {bs['auc_mean']:>8.4f} [{bs['auc_ci_lo']:.4f}, {bs['auc_ci_hi']:.4f}]"
            f" {bs['f1_mean']:>8.4f} [{bs['f1_ci_lo']:.4f}, {bs['f1_ci_hi']:.4f}]"
        )

    results_lines.append(f"\n\nB. PAIRED BOOTSTRAP TESTS (AUC difference, H1: B > A)\n")
    results_lines.append(f"{'Comparison':<55} {'Mean dAUC':>12} {'95% CI':>22} {'p-value':>10}")
    results_lines.append("-" * 105)

    comparisons = [
        ("rf_std", "rf_std_rqa", "RF Std → RF Std+RQA"),
    ]
    for beta_value in beta_values:
        suffix = str(beta_value).replace(".", "_")
        b_name = "rf_std_beta" if len(beta_values) == 1 else f"rf_std_beta_b{suffix}"
        comparisons.append(("rf_std", b_name, f"RF Std → RF Std+β-RQA (β={beta_value})"))
    comparisons.append(("lr_std", "lr_std_rqa", "LogReg Std → LogReg Std+RQA"))
    for beta_value in beta_values:
        suffix = str(beta_value).replace(".", "_")
        b_name = "lr_std_beta" if len(beta_values) == 1 else f"lr_std_beta_b{suffix}"
        comparisons.append(("lr_std", b_name, f"LogReg Std → LogReg Std+β-RQA (β={beta_value})"))

    for name_a, name_b, label in comparisons:
        pa, pb = predictions[name_a], predictions[name_b]
        n = min(len(pa["y_true"]), len(pb["y_true"]))
        test_res = paired_bootstrap_test(pa["y_true"][:n], pa["y_score"][:n], pb["y_score"][:n])
        sig = "***" if test_res["p_value_b_better"] < 0.001 else \
              "**" if test_res["p_value_b_better"] < 0.01 else \
              "*" if test_res["p_value_b_better"] < 0.05 else "n.s."
        results_lines.append(
            f"{label:<55} {test_res['mean_auc_diff']:>+12.5f} "
            f"[{test_res['ci_lo']:>+.5f}, {test_res['ci_hi']:>+.5f}] "
            f"{test_res['p_value_b_better']:>10.4f} {sig}"
        )

    out_dir = repo_root() / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"statistical_tests_d{dataset}.txt"
    results_text = "\n".join(results_lines)
    out_path.write_text(results_text, encoding="utf-8")
    logger.info(f"\nResults saved to {out_path}")
    print(results_text)

    pred_path = out_dir / f"predictions_d{dataset}.npz"
    np.savez(pred_path, **{f"{k}_{sub}": v[sub] for k, v in predictions.items() for sub in ["y_true", "y_score", "y_pred"]})
    logger.info(f"Predictions saved to {pred_path}")


STD_FEATURE_COLS_D3 = [
    "ret_abs", "ret_sq", "rv",
    "ret_mean_30", "ret_std_30", "ret_abs_mean_30", "rv_mean_30", "rv_std_30",
    "ret_mean_120", "ret_std_120", "ret_abs_mean_120", "rv_mean_120", "rv_std_120",
    "ret_mean_390", "ret_std_390", "ret_abs_mean_390", "rv_mean_390", "rv_std_390",
]


def compute_rqa_per_stock_d3(df, rqa_cfg, rqa_cols):
    """Compute RQA per stock for D3."""
    all_rqa = []
    tickers = sorted(df["ticker"].unique())
    sample_tickers = tickers[:min(10, len(tickers))]
    sample_blocks = [df[df["ticker"] == t][list(rqa_cols)].to_numpy(dtype=float) for t in sample_tickers]
    combined_sample = np.concatenate(sample_blocks, axis=0)
    eps_fixed = estimate_epsilon_from_train(combined_sample, rqa_cfg)

    for ticker in tickers:
        mask = df["ticker"] == ticker
        df_ticker = df.loc[mask].reset_index(drop=True)
        rqa_feats = rqa_features_rolling(df_ticker, rqa_cols, rqa_cfg, prefix="rqa", eps_fixed=eps_fixed)
        rqa_feats.index = df.index[mask]
        all_rqa.append(rqa_feats)
    return pd.concat(all_rqa).sort_index()


def compute_beta_rqa_per_stock_d3(df, beta_cfg, cols):
    from scalable_rqa_volatility.recurrence.beta_rqa import beta_rqa_features_rolling
    all_feats = []
    for ticker in sorted(df["ticker"].unique()):
        mask = df["ticker"] == ticker
        df_ticker = df.loc[mask].reset_index(drop=True)
        feats = beta_rqa_features_rolling(df_ticker, cols, beta_cfg, prefix="beta_rqa")
        feats.index = df.index[mask]
        all_feats.append(feats)
    return pd.concat(all_feats).sort_index()


def build_xy_per_stock_d3(df, X_df):
    """Build X, y per stock (no cross-stock leakage)."""
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


def per_stock_metrics(y_true, y_score, y_pred, tickers):
    """Compute AUC and F1 per stock."""
    unique_tickers = sorted(set(tickers))
    stock_aucs, stock_f1s = [], []
    valid_tickers = []

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


def run_d3(logger):
    """Run statistical tests for Dataset 3 (per-stock metrics + Wilcoxon)."""
    logger.info("=== DATASET 3 STATISTICAL TESTS ===")

    train = load_split(3, "train")
    val = load_split(3, "val")
    test = load_split(3, "test")
    full = pd.concat([train, val, test], ignore_index=True)
    n_tr, n_va = len(train), len(val)

    logger.info(f"Train: {n_tr:,}, Val: {n_va:,}, Test: {len(test):,}")

    X_std_tr = train[STD_FEATURE_COLS_D3].copy()
    X_std_va = val[STD_FEATURE_COLS_D3].copy()
    X_std_te = test[STD_FEATURE_COLS_D3].copy()

    Xtr_std, ytr, tickers_tr = build_xy_per_stock_d3(train, X_std_tr)
    Xva_std, yva, tickers_va = build_xy_per_stock_d3(val, X_std_va)
    Xte_std, yte, tickers_te = build_xy_per_stock_d3(test, X_std_te)

    logger.info(f"Std: Train={len(ytr):,}, Val={len(yva):,}, Test={len(yte):,}")

    logger.info("Computing RQA features (default, step=20)...")
    rqa_cfg = RQAConfig(window=60, step=20, recurrence_rate=0.1,
                        embed=EmbeddingConfig(m=4, tau=2), mode="joint")
    X_rqa_full = compute_rqa_per_stock_d3(full, rqa_cfg, ("log_return", "rv"))
    X_rqa_tr = X_rqa_full.iloc[:n_tr].reset_index(drop=True)
    X_rqa_va = X_rqa_full.iloc[n_tr:n_tr+n_va].reset_index(drop=True)
    X_rqa_te = X_rqa_full.iloc[n_tr+n_va:].reset_index(drop=True)

    Xtr_rqa, ytr_r, tickers_tr_r = build_xy_per_stock_d3(train, X_rqa_tr)
    Xva_rqa, yva_r, _ = build_xy_per_stock_d3(val, X_rqa_va)
    Xte_rqa, yte_r, tickers_te_r = build_xy_per_stock_d3(test, X_rqa_te)

    logger.info("Computing β-RQA features (β=2.0, step=20)...")
    beta_cfg = BetaRQAConfig(window=60, step=20, recurrence_rate=0.1,
                             embed=EmbeddingConfig(m=4, tau=2),
                             beta=2.0, transform="minmax")
    X_beta_full = compute_beta_rqa_per_stock_d3(full, beta_cfg, ("log_return",))
    X_beta_tr = X_beta_full.iloc[:n_tr].reset_index(drop=True)
    X_beta_va = X_beta_full.iloc[n_tr:n_tr+n_va].reset_index(drop=True)
    X_beta_te = X_beta_full.iloc[n_tr+n_va:].reset_index(drop=True)

    Xtr_beta, ytr_bt, tickers_tr_bt = build_xy_per_stock_d3(train, X_beta_tr)
    Xva_beta, yva_bt, _ = build_xy_per_stock_d3(val, X_beta_va)
    Xte_beta, yte_bt, tickers_te_bt = build_xy_per_stock_d3(test, X_beta_te)

    n_rqa = min(len(Xtr_std), len(Xtr_rqa))
    Xtr_comb_rqa = np.concatenate([Xtr_std[:n_rqa], Xtr_rqa[:n_rqa]], axis=1)
    Xva_comb_rqa = np.concatenate([Xva_std[:min(len(Xva_std), len(Xva_rqa))],
                                   Xva_rqa[:min(len(Xva_std), len(Xva_rqa))]], axis=1)
    n_te_rqa = min(len(Xte_std), len(Xte_rqa))
    Xte_comb_rqa = np.concatenate([Xte_std[:n_te_rqa], Xte_rqa[:n_te_rqa]], axis=1)
    yte_comb_rqa = yte[:n_te_rqa]
    tickers_te_comb_rqa = tickers_te[:n_te_rqa]

    n_beta = min(len(Xtr_std), len(Xtr_beta))
    Xtr_comb_beta = np.concatenate([Xtr_std[:n_beta], Xtr_beta[:n_beta]], axis=1)
    Xva_comb_beta = np.concatenate([Xva_std[:min(len(Xva_std), len(Xva_beta))],
                                    Xva_beta[:min(len(Xva_std), len(Xva_beta))]], axis=1)
    n_te_beta = min(len(Xte_std), len(Xte_beta))
    Xte_comb_beta = np.concatenate([Xte_std[:n_te_beta], Xte_beta[:n_te_beta]], axis=1)
    yte_comb_beta = yte[:n_te_beta]
    tickers_te_comb_beta = tickers_te[:n_te_beta]

    predictions = {}

    def train_predict_d3(name, model, Xtr, ytr_, Xva, yva_, Xte, yte_, tickers_te_):
        logger.info(f"  Training {name}...")
        model.fit(Xtr, ytr_)
        p_va = model.predict_proba(Xva)[:, 1]
        thr = best_threshold(yva_, p_va)
        p_te = model.predict_proba(Xte)[:, 1]
        pred = (p_te >= thr).astype(int)
        predictions[name] = {"y_true": yte_, "y_score": p_te, "y_pred": pred, "tickers": tickers_te_}
        auc = roc_auc_score(yte_, p_te)
        logger.info(f"    {name}: AUC={auc:.4f}")

    rf = lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                        n_jobs=-1, class_weight="balanced_subsample", random_state=42)
    lr = lambda: Pipeline([("scaler", StandardScaler()),
                           ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", n_jobs=-1))])

    train_predict_d3("rf_std", rf(), Xtr_std, ytr, Xva_std, yva, Xte_std, yte, tickers_te)
    train_predict_d3("rf_std_rqa", rf(), Xtr_comb_rqa, ytr[:n_rqa], Xva_comb_rqa,
                     yva[:min(len(yva), len(Xva_rqa))], Xte_comb_rqa, yte_comb_rqa, tickers_te_comb_rqa)
    train_predict_d3("rf_std_beta", rf(), Xtr_comb_beta, ytr[:n_beta], Xva_comb_beta,
                     yva[:min(len(yva), len(Xva_beta))], Xte_comb_beta, yte_comb_beta, tickers_te_comb_beta)
    train_predict_d3("lr_std", lr(), Xtr_std, ytr, Xva_std, yva, Xte_std, yte, tickers_te)
    train_predict_d3("lr_std_rqa", lr(), Xtr_comb_rqa, ytr[:n_rqa], Xva_comb_rqa,
                     yva[:min(len(yva), len(Xva_rqa))], Xte_comb_rqa, yte_comb_rqa, tickers_te_comb_rqa)
    train_predict_d3("lr_std_beta", lr(), Xtr_comb_beta, ytr[:n_beta], Xva_comb_beta,
                     yva[:min(len(yva), len(Xva_beta))], Xte_comb_beta, yte_comb_beta, tickers_te_comb_beta)

    logger.info("Computing per-stock metrics...")
    results_lines = []
    results_lines.append(f"{'='*80}")
    results_lines.append(f"  DATASET 3 — STATISTICAL SIGNIFICANCE TESTS")
    results_lines.append(f"  Per-stock metrics (503 stocks) + Wilcoxon signed-rank test")
    results_lines.append(f"{'='*80}\n")

    results_lines.append("A. PER-STOCK METRICS (mean ± std across stocks)\n")
    results_lines.append(f"{'Model':<25} {'AUC':>12} {'F1':>12} {'N stocks':>10}")
    results_lines.append("-" * 65)

    stock_results = {}
    for name, p in predictions.items():
        aucs, f1s, valid = per_stock_metrics(p["y_true"], p["y_score"], p["y_pred"], p["tickers"])
        stock_results[name] = {"aucs": aucs, "f1s": f1s, "valid": valid}
        results_lines.append(
            f"{name:<25} {np.mean(aucs):.4f}±{np.std(aucs):.4f} "
            f"{np.mean(f1s):.4f}±{np.std(f1s):.4f} {len(valid):>10}"
        )

    results_lines.append(f"\n\nB. WILCOXON SIGNED-RANK TESTS (per-stock AUC, H1: B > A)\n")
    results_lines.append(f"{'Comparison':<45} {'Mean dAUC':>12} {'Median dAUC':>14} {'p-value':>10} {'Sig':>5}")
    results_lines.append("-" * 90)

    comparisons_d3 = [
        ("rf_std", "rf_std_rqa", "RF Std → RF Std+RQA"),
        ("rf_std", "rf_std_beta", "RF Std → RF Std+β-RQA (β=2.0)"),
        ("lr_std", "lr_std_rqa", "LogReg Std → LogReg Std+RQA"),
        ("lr_std", "lr_std_beta", "LogReg Std → LogReg Std+β-RQA (β=2.0)"),
    ]

    for name_a, name_b, label in comparisons_d3:
        sa, sb = stock_results[name_a], stock_results[name_b]
        common = sorted(set(sa["valid"]) & set(sb["valid"]))
        idx_a = {t: i for i, t in enumerate(sa["valid"])}
        idx_b = {t: i for i, t in enumerate(sb["valid"])}
        aucs_a = np.array([sa["aucs"][idx_a[t]] for t in common])
        aucs_b = np.array([sb["aucs"][idx_b[t]] for t in common])
        diff = aucs_b - aucs_a

        try:
            stat, p_two = scipy_stats.wilcoxon(diff, alternative="greater")
            p_val = p_two
        except ValueError:
            p_val = 1.0

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        results_lines.append(
            f"{label:<45} {np.mean(diff):>+12.5f} {np.median(diff):>+14.5f} {p_val:>10.4f} {sig:>5}"
        )

    results_lines.append(f"\n\nC. BOOTSTRAP CONFIDENCE INTERVALS (pooled across all stocks)\n")
    results_lines.append(f"{'Model':<25} {'AUC':>8} {'AUC 95% CI':>20} {'F1':>8} {'F1 95% CI':>20}")
    results_lines.append("-" * 85)

    for name, p in predictions.items():
        bs = bootstrap_metric(p["y_true"], p["y_score"], p["y_pred"])
        results_lines.append(
            f"{name:<25} {bs['auc_mean']:>8.4f} [{bs['auc_ci_lo']:.4f}, {bs['auc_ci_hi']:.4f}]"
            f" {bs['f1_mean']:>8.4f} [{bs['f1_ci_lo']:.4f}, {bs['f1_ci_hi']:.4f}]"
        )

    out_dir = repo_root() / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"statistical_tests_d3.txt"
    results_text = "\n".join(results_lines)
    out_path.write_text(results_text, encoding="utf-8")
    logger.info(f"\nResults saved to {out_path}")
    print(results_text)


def main():
    logger = get_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=int, required=True, choices=[1, 2, 3])
    args = parser.parse_args()

    if args.dataset in (1, 2):
        run_d12(args.dataset, logger)
    else:
        run_d3(logger)


if __name__ == "__main__":
    main()