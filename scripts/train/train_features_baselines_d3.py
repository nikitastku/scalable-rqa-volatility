"""
Feature baselines for Dataset 3 (intraday multi-stock).

Key adaptations from Datasets 1/2:
  - Standard features are PRECOMPUTED in the parquet (no recomputation needed)
  - RQA is computed per-stock with step=20 (4x faster than step=5)
  - RQA window=60 bars (~2 hours of intraday recurrence)
  - Pooled across all 503 stocks for training
  - Uses --rqa_config CLI for hyperparameter sweep
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scalable_rqa_volatility.evaluation.metrics import classification_metrics
from scalable_rqa_volatility.logging_utils import get_logger
from scalable_rqa_volatility.recurrence.embeddings import EmbeddingConfig
from scalable_rqa_volatility.recurrence.rqa import RQAConfig, rqa_features_rolling, estimate_epsilon_from_train

RQA_CONFIGS = {
    "default":       RQAConfig(window=60, step=20, recurrence_rate=0.1, embed=EmbeddingConfig(m=4, tau=2), mode="joint"),
    "small_window":  RQAConfig(window=30, step=20, recurrence_rate=0.1, embed=EmbeddingConfig(m=3, tau=1), mode="joint"),
    "large_window":  RQAConfig(window=120, step=20, recurrence_rate=0.1, embed=EmbeddingConfig(m=4, tau=2), mode="joint"),
    "per_series":    RQAConfig(window=60, step=20, recurrence_rate=0.1, embed=EmbeddingConfig(m=4, tau=2), mode="per_series"),
}

STD_FEATURE_COLS = [
    "ret_abs", "ret_sq", "rv",
    "ret_mean_30", "ret_std_30", "ret_abs_mean_30", "rv_mean_30", "rv_std_30",
    "ret_mean_120", "ret_std_120", "ret_abs_mean_120", "rv_mean_120", "rv_std_120",
    "ret_mean_390", "ret_std_390", "ret_abs_mean_390", "rv_mean_390", "rv_std_390",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_split(name: str) -> pd.DataFrame:
    return pd.read_parquet(repo_root() / "data" / "processed" / f"dataset3_{name}.parquet")


def compute_rqa_per_stock(df: pd.DataFrame, rqa_cfg: RQAConfig, rqa_cols: tuple[str, ...]) -> pd.DataFrame:
    """Compute RQA features per stock, then concatenate."""
    all_rqa = []
    tickers = sorted(df["ticker"].unique())

    sample_tickers = tickers[:min(10, len(tickers))]
    sample_blocks = []
    for t in sample_tickers:
        block = df[df["ticker"] == t][list(rqa_cols)].to_numpy(dtype=float)
        sample_blocks.append(block)
    combined_sample = np.concatenate(sample_blocks, axis=0)
    eps_fixed = estimate_epsilon_from_train(combined_sample, rqa_cfg)

    for ticker in tickers:
        mask = df["ticker"] == ticker
        df_ticker = df.loc[mask].reset_index(drop=True)
        rqa_feats = rqa_features_rolling(df_ticker, rqa_cols, rqa_cfg, prefix="rqa", eps_fixed=eps_fixed)
        rqa_feats.index = df.index[mask]
        all_rqa.append(rqa_feats)

    return pd.concat(all_rqa).sort_index()


def build_xy(df: pd.DataFrame, X_df: pd.DataFrame, label_col: str = "regime") -> tuple[np.ndarray, np.ndarray]:
    """Build X, y arrays — predict next-bar regime."""
    y_next = df[label_col].shift(-1).to_numpy()
    ok = pd.notna(y_next)
    X = X_df.to_numpy(dtype=float)[ok]
    y = y_next[ok].astype(int)
    X = np.where(np.isfinite(X), X, np.nan)
    keep = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[keep], y[keep]


def f1_fast(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    denom = 2 * tp + fp + fn
    return float((2 * tp) / denom) if denom else 0.0


def best_threshold_target_rate(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    prob = np.asarray(prob, dtype=float).reshape(-1)
    if y_true.size == 0:
        return 0.5, 0.0

    target = float(np.clip(np.mean(y_true), 0.05, 0.95))

    if prob.size > 100000:
        grid = np.unique(np.quantile(prob, np.linspace(0.01, 0.99, 499)))
    else:
        grid = np.unique(prob)

    best_t, best_f1 = None, -1.0
    for t in grid:
        y_pred = (prob >= float(t)).astype(int)
        pos = float(np.mean(y_pred))
        if not (target * 0.5 <= pos <= min(0.99, target * 1.5)):
            continue
        f1 = f1_fast(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    if best_t is None:
        t = float(np.quantile(prob, 1.0 - target))
        y_pred = (prob >= t).astype(int)
        return float(t), float(f1_fast(y_true, y_pred))

    return float(best_t), float(best_f1)


def fit_thresholded(name: str, model, Xtr, ytr, Xva, yva, Xte, yte, logger) -> None:
    t0 = time.time()
    model.fit(Xtr, ytr)
    fit_time = time.time() - t0

    p_val = model.predict_proba(Xva)[:, 1]
    thr, val_f1 = best_threshold_target_rate(yva, p_val)

    p_test = model.predict_proba(Xte)[:, 1]
    pred = (p_test >= thr).astype(int)

    metrics = classification_metrics(yte, pred, p_test)
    cm = confusion_matrix(yte, pred)
    logger.info({
        f"{name}_threshold": float(thr),
        f"{name}_val_f1": float(val_f1),
        f"{name}_metrics": metrics,
        f"{name}_cm": cm.tolist(),
        f"{name}_fit_time_s": round(fit_time, 1),
    })


def main() -> None:
    logger = get_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument("--rqa_config", type=str, default="default", choices=list(RQA_CONFIGS.keys()))
    parser.add_argument("--skip_rqa", action="store_true", help="Skip RQA computation (run std-only baselines)")
    args = parser.parse_args()

    rqa_cfg = RQA_CONFIGS[args.rqa_config]
    rqa_cols = ("log_return", "rv")

    logger.info({"rqa_config": args.rqa_config, "rqa_window": rqa_cfg.window,
                 "rqa_step": rqa_cfg.step, "rqa_m": rqa_cfg.embed.m,
                 "rqa_tau": rqa_cfg.embed.tau, "rqa_mode": rqa_cfg.mode})

    logger.info("Loading dataset3 splits...")
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")

    logger.info(f"Train: {len(train):,}, Val: {len(val):,}, Test: {len(test):,}")

    X_std_train = train[STD_FEATURE_COLS].copy()
    X_std_val = val[STD_FEATURE_COLS].copy()
    X_std_test = test[STD_FEATURE_COLS].copy()

    def build_xy_per_stock(df, X_df):
        all_X, all_y = [], []
        for ticker in df["ticker"].unique():
            mask = df["ticker"] == ticker
            df_t = df.loc[mask].reset_index(drop=True)
            X_t = X_df.loc[mask].reset_index(drop=True)
            X_arr, y_arr = build_xy(df_t, X_t)
            all_X.append(X_arr)
            all_y.append(y_arr)
        return np.concatenate(all_X), np.concatenate(all_y)

    logger.info("Building X/y arrays (per-stock shift)...")
    X_std_tr, y_tr = build_xy_per_stock(train, X_std_train)
    X_std_va, y_va = build_xy_per_stock(val, X_std_val)
    X_std_te, y_te = build_xy_per_stock(test, X_std_test)

    logger.info({
        "std_features": X_std_tr.shape[1],
        "train_samples": X_std_tr.shape[0],
        "val_samples": X_std_va.shape[0],
        "test_samples": X_std_te.shape[0],
        "train_prevalence": float(np.mean(y_tr)),
        "val_prevalence": float(np.mean(y_va)),
        "test_prevalence": float(np.mean(y_te)),
    })

    logreg = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", n_jobs=-1))])
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=-1,
                                class_weight="balanced_subsample", random_state=42)

    logger.info("=== Standard-only baselines ===")
    fit_thresholded("logreg_std", logreg, X_std_tr, y_tr, X_std_va, y_va, X_std_te, y_te, logger)
    fit_thresholded("rf_std", rf, X_std_tr, y_tr, X_std_va, y_va, X_std_te, y_te, logger)

    if args.skip_rqa:
        logger.info("Skipping RQA (--skip_rqa flag set)")
        return

    logger.info(f"Computing RQA features ({args.rqa_config})...")
    full = pd.concat([train, val, test], ignore_index=True)
    n_tr, n_va = len(train), len(val)

    t0 = time.time()
    X_rqa_all = compute_rqa_per_stock(full, rqa_cfg, rqa_cols)
    rqa_time = time.time() - t0
    logger.info(f"RQA computation: {rqa_time:.1f}s, {X_rqa_all.shape[1]} features")

    X_rqa_train = X_rqa_all.iloc[:n_tr].reset_index(drop=True)
    X_rqa_val = X_rqa_all.iloc[n_tr:n_tr + n_va].reset_index(drop=True)
    X_rqa_test = X_rqa_all.iloc[n_tr + n_va:].reset_index(drop=True)

    def build_rqa_xy_per_stock(df, X_rqa_df):
        all_X, all_y = [], []
        for ticker in df["ticker"].unique():
            mask = df["ticker"] == ticker
            df_t = df.loc[mask].reset_index(drop=True)
            X_t = X_rqa_df.loc[mask].reset_index(drop=True)
            X_arr, y_arr = build_xy(df_t, X_t)
            all_X.append(X_arr)
            all_y.append(y_arr)
        return np.concatenate(all_X), np.concatenate(all_y)

    X_rqa_tr, y_tr2 = build_rqa_xy_per_stock(train, X_rqa_train)
    X_rqa_va, y_va2 = build_rqa_xy_per_stock(val, X_rqa_val)
    X_rqa_te, y_te2 = build_rqa_xy_per_stock(test, X_rqa_test)

    ntr = min(len(y_tr), len(y_tr2))
    nva = min(len(y_va), len(y_va2))
    nte = min(len(y_te), len(y_te2))

    X_std_tr2, y_tr = X_std_tr[:ntr], y_tr[:ntr]
    X_rqa_tr = X_rqa_tr[:ntr]
    X_std_va2, y_va = X_std_va[:nva], y_va[:nva]
    X_rqa_va = X_rqa_va[:nva]
    X_std_te2, y_te = X_std_te[:nte], y_te[:nte]
    X_rqa_te = X_rqa_te[:nte]

    X_comb_tr = np.concatenate([X_std_tr2, X_rqa_tr], axis=1)
    X_comb_va = np.concatenate([X_std_va2, X_rqa_va], axis=1)
    X_comb_te = np.concatenate([X_std_te2, X_rqa_te], axis=1)

    logger.info({"rqa_features": X_rqa_tr.shape[1], "combined_features": X_comb_tr.shape[1]})

    logger.info("=== RQA-only baselines ===")
    fit_thresholded("logreg_rqa", logreg, X_rqa_tr, y_tr, X_rqa_va, y_va, X_rqa_te, y_te, logger)
    fit_thresholded("rf_rqa", rf, X_rqa_tr, y_tr, X_rqa_va, y_va, X_rqa_te, y_te, logger)

    logger.info("=== Standard+RQA combined baselines ===")
    fit_thresholded("logreg_comb", logreg, X_comb_tr, y_tr, X_comb_va, y_va, X_comb_te, y_te, logger)
    fit_thresholded("rf_comb", rf, X_comb_tr, y_tr, X_comb_va, y_va, X_comb_te, y_te, logger)


if __name__ == "__main__":
    main()