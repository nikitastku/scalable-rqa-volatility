from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from scalable_rqa_volatility.evaluation.metrics import classification_metrics
from scalable_rqa_volatility.logging_utils import get_logger
from scalable_rqa_volatility.models.econometric_gjr_garch import GJRGarchModel


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_split(name: str) -> pd.DataFrame:
    return pd.read_parquet(repo_root() / "data" / "processed" / f"dataset1_{name}.parquet")


def rolling_threshold_from_history(rv: np.ndarray, lookback: int = 252, q: float = 0.7) -> np.ndarray:
    s = pd.Series(rv)
    return s.rolling(lookback, min_periods=lookback).quantile(q).shift(1).to_numpy(dtype=float)


def best_offset_f1(scores: np.ndarray, y_true: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    if scores.size == 0:
        return 0.0
    grid = np.unique(scores)
    best_b = float(grid[0])
    best_f1 = -1.0
    for b in grid:
        y_pred = (scores - float(b) >= 0.0).astype(int)
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_b = float(b)
    return float(best_b)


def fit_scale_a(sigma: np.ndarray, rv_next: np.ndarray) -> float:
    s = np.asarray(sigma, dtype=float).reshape(-1)
    y = np.asarray(rv_next, dtype=float).reshape(-1)
    m = np.isfinite(s) & np.isfinite(y)
    if m.sum() == 0:
        return 1.0
    denom = float(np.dot(s[m], s[m]))
    if denom == 0.0:
        return 1.0
    return float(np.dot(s[m], y[m]) / denom)


def build_split_arrays(
    full_rv: np.ndarray,
    full_regime: np.ndarray,
    thr_hist: np.ndarray,
    sigma_full: np.ndarray,
    a: float,
    split_start: int,
    split_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    anchors = np.arange(split_start, split_end, dtype=int)
    sigma = sigma_full[: anchors.size]
    pred_rv_next = a * sigma
    y_true = full_regime[anchors + 1].astype(int)
    thr_next = thr_hist[anchors + 1]
    m = np.isfinite(pred_rv_next) & np.isfinite(thr_next) & np.isfinite(y_true)
    y = y_true[m]
    score = (pred_rv_next[m] - thr_next[m]).astype(float)
    return y, score


def main() -> None:
    logger = get_logger()

    train = load_split("train")
    val = load_split("val")
    test = load_split("test")
    full = pd.concat([train, val, test], axis=0).reset_index(drop=True)

    n_train = len(train)
    n_val = len(val)

    train_start, train_end = 0, n_train - 1
    val_start, val_end = n_train, n_train + n_val - 1
    test_start, test_end = n_train + n_val, len(full) - 1

    rv = full["rv"].astype(float).to_numpy()
    regime = full["regime"].astype(float).to_numpy()
    thr_hist = rolling_threshold_from_history(rv, lookback=252, q=0.7)

    model = GJRGarchModel()
    model.fit(full["log_return"], train_last_obs=train_end)

    sigma_tr = model.forecast_sigma_one_step(start=train_start, end=train_end - 1)
    rv_next_tr = rv[train_start + 1 : train_end + 1]
    a = fit_scale_a(sigma_tr, rv_next_tr)

    sigma_val = model.forecast_sigma_one_step(start=val_start, end=val_end - 1)
    sigma_te = model.forecast_sigma_one_step(start=test_start, end=test_end - 1)

    y_val, score_val = build_split_arrays(rv, regime, thr_hist, sigma_val, a, val_start, val_end)
    y_te, score_te = build_split_arrays(rv, regime, thr_hist, sigma_te, a, test_start, test_end)

    b = best_offset_f1(score_val, y_val)
    logger.info({"calibration_offset_b": float(b), "a_sigma_to_rv": float(a)})

    y_pred = (score_te - b >= 0.0).astype(int)
    metrics = classification_metrics(y_te, y_pred, score_te)
    cm = confusion_matrix(y_te, y_pred)

    logger.info({"prevalence": float(y_te.mean()), **metrics})
    logger.info({"confusion_matrix": cm.tolist()})


if __name__ == "__main__":
    main()