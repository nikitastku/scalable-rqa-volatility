"""
GJR-GARCH for Dataset 3 (intraday multi-stock).

Fits GJR-GARCH per stock on the training portion, then evaluates on the test split.
Pools predictions across all stocks for the final metric computation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.metrics import confusion_matrix, f1_score

from scalable_rqa_volatility.evaluation.metrics import classification_metrics
from scalable_rqa_volatility.logging_utils import get_logger


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_split(name: str) -> pd.DataFrame:
    return pd.read_parquet(repo_root() / "data" / "processed" / f"dataset3_{name}.parquet")


def best_offset_f1(scores, y_true):
    scores = np.asarray(scores, dtype=float).reshape(-1)
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    if scores.size == 0: return 0.0
    grid = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 199)))
    best_b, best_f1 = float(grid[0]), -1.0
    for b in grid:
        y_pred = (scores - float(b) >= 0.0).astype(int)
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        if f1 > best_f1: best_f1, best_b = f1, float(b)
    return float(best_b)


def main() -> None:
    logger = get_logger()

    train = load_split("train")
    val = load_split("val")
    test = load_split("test")
    full = pd.concat([train, val, test], ignore_index=True)

    logger.info(f"Train: {len(train):,}, Val: {len(val):,}, Test: {len(test):,}")

    scale = 100.0
    n_train = len(train)
    n_val = len(val)

    all_val_scores, all_val_y = [], []
    all_test_scores, all_test_y = [], []
    fitted, failed = 0, 0

    tickers = sorted(full["ticker"].unique())
    logger.info(f"Fitting GJR-GARCH for {len(tickers)} stocks...")

    for i, ticker in enumerate(tickers):
        mask = full["ticker"] == ticker
        df_t = full.loc[mask].reset_index(drop=True)

        returns = df_t["log_return"].astype(float).to_numpy() * scale
        rv = df_t["rv"].astype(float).to_numpy()
        regime = df_t["regime"].astype(float).to_numpy()

        n = len(df_t)
        n_tr = int(n * 0.7)
        n_va = int(n * 0.15)

        try:
            am = arch_model(returns, mean="zero", vol="GARCH", p=1, o=1, q=1, dist="t", rescale=False)
            res = am.fit(disp="off", last_obs=n_tr)
            fc = res.forecast(horizon=1, start=0, reindex=False)
            var = fc.variance.iloc[:, 0].to_numpy()
            sigma = np.sqrt(var) / scale

            sigma_tr = sigma[:n_tr - 1]
            rv_next_tr = rv[1:n_tr]
            m = np.isfinite(sigma_tr) & np.isfinite(rv_next_tr) & (sigma_tr > 0)
            if m.sum() > 0:
                a = float(np.dot(sigma_tr[m], rv_next_tr[m]) / np.dot(sigma_tr[m], sigma_tr[m]))
            else:
                a = 1.0

            val_start, val_end = n_tr, n_tr + n_va
            for t in range(val_start, min(val_end, n - 1)):
                pred_rv = a * sigma[t] if t < len(sigma) else np.nan
                if np.isfinite(pred_rv) and np.isfinite(regime[t + 1]):
                    all_val_scores.append(pred_rv)
                    all_val_y.append(int(regime[t + 1]))

            test_start = n_tr + n_va
            for t in range(test_start, n - 1):
                pred_rv = a * sigma[t] if t < len(sigma) else np.nan
                if np.isfinite(pred_rv) and np.isfinite(regime[t + 1]):
                    all_test_scores.append(pred_rv)
                    all_test_y.append(int(regime[t + 1]))

            fitted += 1
        except Exception:
            failed += 1

        if (i + 1) % 100 == 0:
            logger.info(f"  {i+1}/{len(tickers)} stocks (fitted={fitted}, failed={failed})")

    logger.info(f"GARCH fitting complete: {fitted} fitted, {failed} failed")

    score_va = np.array(all_val_scores)
    y_va = np.array(all_val_y)
    score_te = np.array(all_test_scores)
    y_te = np.array(all_test_y)

    logger.info(f"Val: {len(y_va):,} predictions, Test: {len(y_te):,} predictions")

    b = best_offset_f1(score_va, y_va)
    logger.info({"calibration_offset_b": float(b)})

    y_pred = (score_te - b >= 0.0).astype(int)
    metrics = classification_metrics(y_te, y_pred, score_te)
    cm = confusion_matrix(y_te, y_pred)

    logger.info({"prevalence": float(y_te.mean()), **metrics})
    logger.info({"confusion_matrix": cm.tolist()})


if __name__ == "__main__":
    main()