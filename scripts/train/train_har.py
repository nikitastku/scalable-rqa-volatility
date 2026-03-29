from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix

from scalable_rqa_volatility.evaluation.metrics import classification_metrics
from scalable_rqa_volatility.logging_utils import get_logger


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_split(name: str) -> pd.DataFrame:
    return pd.read_parquet(repo_root() / "data" / "processed" / f"dataset2_{name}.parquet")


def rolling_threshold_from_history(rv: np.ndarray, lookback: int = 252, q: float = 0.7) -> np.ndarray:
    s = pd.Series(rv)
    return s.rolling(lookback, min_periods=lookback).quantile(q).shift(1).to_numpy(dtype=float)


def build_full_har_frame(full: pd.DataFrame) -> pd.DataFrame:
    rv = full["rv"].astype(float)
    out = pd.DataFrame({"t": np.arange(len(full), dtype=int)})

    out["rv_daily"] = rv.shift(1)
    out["rv_weekly"] = rv.rolling(5, min_periods=5).mean().shift(1)
    out["rv_monthly"] = rv.rolling(22, min_periods=22).mean().shift(1)

    out["rv_next"] = rv.shift(-1)
    out["regime_next"] = full["regime"].shift(-1).astype("Int64")

    out = out.dropna(subset=["rv_daily", "rv_weekly", "rv_monthly", "rv_next", "regime_next"]).reset_index(drop=True)
    out["regime_next"] = out["regime_next"].astype(int)
    return out


def select_split(df_har: pd.DataFrame, split_start: int, split_end: int) -> pd.DataFrame:
    t = df_har["t"].to_numpy(dtype=int)
    m = (t >= split_start) & (t <= split_end - 1)
    return df_har.loc[m].reset_index(drop=True)


def calibrate_b_quantile(score_val: np.ndarray, y_val: np.ndarray) -> float:
    score_val = np.asarray(score_val, dtype=float).reshape(-1)
    y_val = np.asarray(y_val, dtype=int).reshape(-1)
    m = np.isfinite(score_val) & np.isfinite(y_val)
    score_val = score_val[m]
    y_val = y_val[m]
    if score_val.size == 0:
        return 0.0
    p = float(np.mean(y_val))
    p = float(np.clip(p, 0.01, 0.99))
    return float(np.quantile(score_val, 1.0 - p))


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
    thr_hist = rolling_threshold_from_history(rv, lookback=252, q=0.7)

    har = build_full_har_frame(full)
    tr = select_split(har, train_start, train_end)
    va = select_split(har, val_start, val_end)
    te = select_split(har, test_start, test_end)

    Xtr = tr[["rv_daily", "rv_weekly", "rv_monthly"]].to_numpy(dtype=float)
    ytr = tr["rv_next"].to_numpy(dtype=float)

    model = LinearRegression()
    model.fit(Xtr, ytr)

    def split_scores(split_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = split_df[["rv_daily", "rv_weekly", "rv_monthly"]].to_numpy(dtype=float)
        pred_rv_next = model.predict(X).astype(float)

        t = split_df["t"].to_numpy(dtype=int)
        y_true = split_df["regime_next"].to_numpy(dtype=int)
        thr_next = thr_hist[t + 1].astype(float)

        m = np.isfinite(pred_rv_next) & np.isfinite(thr_next) & np.isfinite(y_true)
        return y_true[m], (pred_rv_next[m] - thr_next[m]).astype(float)

    y_va, score_va = split_scores(va)
    y_te, score_te = split_scores(te)

    b = calibrate_b_quantile(score_va, y_va)

    val_pos = float(np.mean((score_va - b >= 0.0).astype(int))) if score_va.size else float("nan")
    test_pos = float(np.mean((score_te - b >= 0.0).astype(int))) if score_te.size else float("nan")
    logger.info({"calibration_b": float(b), "val_pos_rate": val_pos, "test_pos_rate": test_pos})

    y_pred = (score_te - b >= 0.0).astype(int)
    metrics = classification_metrics(y_te, y_pred, score_te)
    cm = confusion_matrix(y_te, y_pred)

    logger.info({"prevalence": float(y_te.mean()), **metrics})
    logger.info({"confusion_matrix": cm.tolist()})


if __name__ == "__main__":
    main()