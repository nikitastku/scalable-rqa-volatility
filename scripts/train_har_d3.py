"""
train_har_d3.py — HAR-RV for Dataset 3 (intraday multi-stock).

Adapts HAR-RV for intraday bars: uses bar-level RV lags equivalent to
  daily = 60 bars (~2h), weekly = 300 bars (~10h), monthly = 1950 bars (~2weeks)
Computes HAR features per stock, then fits a single pooled model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix

from scalable_rqa_volatility.evaluation.metrics import classification_metrics
from scalable_rqa_volatility.logging_utils import get_logger


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_split(name: str) -> pd.DataFrame:
    return pd.read_parquet(repo_root() / "data" / "processed" / f"dataset3_{name}.parquet")


def rolling_threshold_from_history(rv: np.ndarray, lookback: int = 975, q: float = 0.7) -> np.ndarray:
    s = pd.Series(rv)
    return s.rolling(lookback, min_periods=lookback).quantile(q).shift(1).to_numpy(dtype=float)


def build_har_features_per_stock(df: pd.DataFrame) -> pd.DataFrame:
    """Build HAR features per stock: daily(60), weekly(300), monthly(1950) bar lags."""
    har_frames = []
    for ticker, group in df.groupby("ticker"):
        g = group.sort_values("timestamp").reset_index(drop=True)
        rv = g["rv"].astype(float)

        out = pd.DataFrame(index=g.index)
        out["rv_daily"] = rv.shift(1)
        out["rv_weekly"] = rv.rolling(300, min_periods=300).mean().shift(1)
        # Monthly: use 1950 if available, else skip
        out["rv_monthly"] = rv.rolling(1950, min_periods=1950).mean().shift(1)

        out["rv_next"] = rv.shift(-1)
        out["regime_next"] = g["regime"].shift(-1).astype("Int64")
        out["ticker"] = ticker
        out["global_idx"] = g.index  # for threshold alignment

        har_frames.append(out.dropna(subset=["rv_daily", "rv_weekly", "rv_next", "regime_next"]))

    return pd.concat(har_frames, ignore_index=True)


def calibrate_b_quantile(score_val: np.ndarray, y_val: np.ndarray) -> float:
    score_val = np.asarray(score_val, dtype=float).reshape(-1)
    y_val = np.asarray(y_val, dtype=int).reshape(-1)
    m = np.isfinite(score_val) & np.isfinite(y_val)
    score_val, y_val = score_val[m], y_val[m]
    if score_val.size == 0:
        return 0.0
    p = float(np.clip(np.mean(y_val), 0.01, 0.99))
    return float(np.quantile(score_val, 1.0 - p))


def main() -> None:
    logger = get_logger()

    train = load_split("train")
    val = load_split("val")
    test = load_split("test")
    full = pd.concat([train, val, test], axis=0).reset_index(drop=True)

    logger.info(f"Building HAR features for {full['ticker'].nunique()} stocks...")
    har = build_har_features_per_stock(full)

    # Drop rows without monthly lag (stocks too short)
    har_with_monthly = har.dropna(subset=["rv_monthly"]).reset_index(drop=True)
    har_no_monthly = har[har["rv_monthly"].isna()].reset_index(drop=True)

    logger.info(f"HAR rows with monthly: {len(har_with_monthly)}, without: {len(har_no_monthly)}")

    # Use all available rows (fill missing monthly with weekly)
    har["rv_monthly"] = har["rv_monthly"].fillna(har["rv_weekly"])
    har = har.dropna(subset=["rv_daily", "rv_weekly", "rv_monthly", "rv_next", "regime_next"]).reset_index(drop=True)

    # Split back by stock's chronological position
    n_train = len(train)
    n_val = len(val)

    # We need to identify which HAR rows belong to which split
    # Since we built from the full concat, we can use global_idx
    # But simpler: rebuild per split
    tr_har = build_har_features_per_stock(train)
    tr_har["rv_monthly"] = tr_har["rv_monthly"].fillna(tr_har["rv_weekly"])
    tr_har = tr_har.dropna(subset=["rv_daily", "rv_weekly", "rv_monthly", "rv_next", "regime_next"])

    va_har = build_har_features_per_stock(val)
    va_har["rv_monthly"] = va_har["rv_monthly"].fillna(va_har["rv_weekly"])
    va_har = va_har.dropna(subset=["rv_daily", "rv_weekly", "rv_monthly", "rv_next", "regime_next"])

    te_har = build_har_features_per_stock(test)
    te_har["rv_monthly"] = te_har["rv_monthly"].fillna(te_har["rv_weekly"])
    te_har = te_har.dropna(subset=["rv_daily", "rv_weekly", "rv_monthly", "rv_next", "regime_next"])

    logger.info(f"Train: {len(tr_har)}, Val: {len(va_har)}, Test: {len(te_har)}")

    feat_cols = ["rv_daily", "rv_weekly", "rv_monthly"]
    Xtr = tr_har[feat_cols].to_numpy(dtype=float)
    ytr = tr_har["rv_next"].to_numpy(dtype=float)

    model = LinearRegression()
    model.fit(Xtr, ytr)
    logger.info(f"HAR coefficients: {dict(zip(feat_cols, model.coef_))}, intercept={model.intercept_:.6f}")

    # Predict and threshold
    def eval_split(split_har, name):
        X = split_har[feat_cols].to_numpy(dtype=float)
        pred_rv = model.predict(X)
        y_true = split_har["regime_next"].to_numpy(dtype=int)

        # Simple approach: regime = 1 if predicted RV > quantile threshold
        # Use val to calibrate
        return y_true, pred_rv

    y_va, score_va = eval_split(va_har, "val")
    y_te, score_te = eval_split(te_har, "test")

    b = calibrate_b_quantile(score_va, y_va)
    logger.info(f"Calibration b: {b:.6f}")

    y_pred = (score_te >= b).astype(int)
    metrics = classification_metrics(y_te, y_pred, score_te)
    cm = confusion_matrix(y_te, y_pred)

    logger.info({"prevalence": float(y_te.mean()), **metrics})
    logger.info({"confusion_matrix": cm.tolist()})


if __name__ == "__main__":
    main()