from __future__ import annotations

import argparse
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
from scalable_rqa_volatility.volatility.features_standard import StandardFeatureConfig, standard_features


# ──────────────────────────────────────────────────────────────
# EXPERIMENT GRID: RQA hyperparameter configurations to try
# ──────────────────────────────────────────────────────────────
RQA_CONFIGS = {
    "default": RQAConfig(window=90, step=5, recurrence_rate=0.1, embed=EmbeddingConfig(m=4, tau=2), mode="joint"),
    "small_window": RQAConfig(window=30, step=5, recurrence_rate=0.1, embed=EmbeddingConfig(m=3, tau=1), mode="joint"),
    "medium_window": RQAConfig(window=60, step=5, recurrence_rate=0.1, embed=EmbeddingConfig(m=4, tau=2), mode="joint"),
    "per_series": RQAConfig(window=90, step=5, recurrence_rate=0.1, embed=EmbeddingConfig(m=4, tau=2), mode="per_series"),
    "high_m": RQAConfig(window=90, step=5, recurrence_rate=0.1, embed=EmbeddingConfig(m=6, tau=3), mode="joint"),
}


@dataclass(frozen=True)
class BaselineConfig:
    std: StandardFeatureConfig = StandardFeatureConfig(windows=(5, 22, 60))
    rqa_cols: tuple[str, ...] = ("log_return", "rv")
    label_col: str = "regime"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_split(name: str) -> pd.DataFrame:
    return pd.read_parquet(repo_root() / "data" / "processed" / f"dataset2_{name}.parquet")


def split_slices(n_train: int, n_val: int, n_test: int) -> dict[str, slice]:
    return {
        "train": slice(0, n_train),
        "val": slice(n_train, n_train + n_val),
        "test": slice(n_train + n_val, n_train + n_val + n_test),
    }


def build_xy_within_split(df_split: pd.DataFrame, X_split: pd.DataFrame, label_col: str) -> tuple[np.ndarray, np.ndarray]:
    y_next = df_split[label_col].shift(-1).to_numpy()
    ok_y = pd.notna(y_next)
    X = X_split.to_numpy(dtype=float, copy=False)[ok_y]
    y = y_next[ok_y].astype(int)
    X = np.where(np.isfinite(X), X, np.nan)
    keep = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[keep], y[keep]


def f1_fast(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.int8, copy=False)
    y_pred = y_pred.astype(np.int8, copy=False)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    denom = 2 * tp + fp + fn
    return float((2 * tp) / denom) if denom else 0.0


def best_threshold_target_rate(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    """
    UNIFIED threshold tuning: F1 maximization within prevalence bounds.
    This is now used consistently across ALL feature baseline experiments.
    Previously, train_beta_features_baselines.py used a different function.
    """
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    prob = np.asarray(prob, dtype=float).reshape(-1)
    if y_true.size == 0:
        return 0.5, 0.0

    target = float(np.mean(y_true))
    target = float(np.clip(target, 0.05, 0.95))

    grid = np.unique(prob)
    if grid.size == 0:
        return 0.5, 0.0

    best_t = None
    best_f1 = -1.0

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
        if y_pred.min() == y_pred.max():
            t = float(np.median(prob))
            y_pred = (prob >= t).astype(int)
        return float(t), float(f1_fast(y_true, y_pred))

    y_pred = (prob >= best_t).astype(int)
    return float(best_t), float(best_f1)


def fit_thresholded(name: str, model, Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray, Xte: np.ndarray, yte: np.ndarray, logger) -> None:
    model.fit(Xtr, ytr)
    p_val = model.predict_proba(Xva)[:, 1]
    thr, val_f1 = best_threshold_target_rate(yva, p_val)

    p_test = model.predict_proba(Xte)[:, 1]
    pred = (p_test >= thr).astype(int)

    metrics = classification_metrics(yte, pred, p_test)
    cm = confusion_matrix(yte, pred)
    logger.info({f"{name}_threshold": float(thr), f"{name}_val_f1_at_threshold": float(val_f1), f"{name}_metrics": metrics, f"{name}_cm": cm.tolist()})


def main() -> None:
    logger = get_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument("--rqa_config", type=str, default="default",
                        choices=list(RQA_CONFIGS.keys()),
                        help="RQA hyperparameter configuration to use")
    args = parser.parse_args()

    rqa_cfg = RQA_CONFIGS[args.rqa_config]
    base_cfg = BaselineConfig()

    logger.info({"rqa_config_name": args.rqa_config,
                 "rqa_window": rqa_cfg.window, "rqa_step": rqa_cfg.step,
                 "rqa_m": rqa_cfg.embed.m, "rqa_tau": rqa_cfg.embed.tau,
                 "rqa_mode": rqa_cfg.mode, "rqa_rr": rqa_cfg.recurrence_rate})

    train = load_split("train").reset_index(drop=True)
    val = load_split("val").reset_index(drop=True)
    test = load_split("test").reset_index(drop=True)

    full_all = pd.concat([train, val, test], axis=0, ignore_index=True)
    n_train, n_val, n_test = len(train), len(val), len(test)
    sl = split_slices(n_train, n_val, n_test)

    train_block = train[list(base_cfg.rqa_cols)].to_numpy(dtype=float, copy=False)
    eps_fixed = estimate_epsilon_from_train(train_block, rqa_cfg)

    X_std_all = standard_features(full_all, base_cfg.std)
    X_rqa_all = rqa_features_rolling(full_all, base_cfg.rqa_cols, rqa_cfg, prefix="rqa", eps_fixed=eps_fixed)

    X_std_train = X_std_all.iloc[sl["train"]].reset_index(drop=True)
    X_std_val = X_std_all.iloc[sl["val"]].reset_index(drop=True)
    X_std_test = X_std_all.iloc[sl["test"]].reset_index(drop=True)

    X_rqa_train = X_rqa_all.iloc[sl["train"]].reset_index(drop=True)
    X_rqa_val = X_rqa_all.iloc[sl["val"]].reset_index(drop=True)
    X_rqa_test = X_rqa_all.iloc[sl["test"]].reset_index(drop=True)

    X_std_train, y_train = build_xy_within_split(train, X_std_train, base_cfg.label_col)
    X_std_val, y_val = build_xy_within_split(val, X_std_val, base_cfg.label_col)
    X_std_test, y_test = build_xy_within_split(test, X_std_test, base_cfg.label_col)

    X_rqa_train, y_train2 = build_xy_within_split(train, X_rqa_train, base_cfg.label_col)
    X_rqa_val, y_val2 = build_xy_within_split(val, X_rqa_val, base_cfg.label_col)
    X_rqa_test, y_test2 = build_xy_within_split(test, X_rqa_test, base_cfg.label_col)

    ntr = min(len(y_train), len(y_train2))
    nva = min(len(y_val), len(y_val2))
    nte = min(len(y_test), len(y_test2))

    X_std_train, y_train = X_std_train[:ntr], y_train[:ntr]
    X_rqa_train = X_rqa_train[:ntr]
    X_std_val, y_val = X_std_val[:nva], y_val[:nva]
    X_rqa_val = X_rqa_val[:nva]
    X_std_test, y_test = X_std_test[:nte], y_test[:nte]
    X_rqa_test = X_rqa_test[:nte]

    X_comb_train = np.concatenate([X_std_train, X_rqa_train], axis=1)
    X_comb_val = np.concatenate([X_std_val, X_rqa_val], axis=1)
    X_comb_test = np.concatenate([X_std_test, X_rqa_test], axis=1)

    logreg = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=-1, class_weight="balanced_subsample", random_state=42)

    logger.info(
        {
            "train_rows": int(len(train)),
            "val_rows": int(len(val)),
            "test_rows": int(len(test)),
            "std_features": int(X_std_train.shape[1]),
            "rqa_features": int(X_rqa_train.shape[1]),
            "combined_features": int(X_comb_train.shape[1]),
            "train_prevalence": float(np.mean(y_train)) if len(y_train) else float("nan"),
            "val_prevalence": float(np.mean(y_val)) if len(y_val) else float("nan"),
            "test_prevalence": float(np.mean(y_test)) if len(y_test) else float("nan"),
        }
    )

    fit_thresholded("logreg_std", logreg, X_std_train, y_train, X_std_val, y_val, X_std_test, y_test, logger)
    fit_thresholded("logreg_rqa", logreg, X_rqa_train, y_train, X_rqa_val, y_val, X_rqa_test, y_test, logger)
    fit_thresholded("logreg_comb", logreg, X_comb_train, y_train, X_comb_val, y_val, X_comb_test, y_test, logger)

    fit_thresholded("rf_std", rf, X_std_train, y_train, X_std_val, y_val, X_std_test, y_test, logger)
    fit_thresholded("rf_rqa", rf, X_rqa_train, y_train, X_rqa_val, y_val, X_rqa_test, y_test, logger)
    fit_thresholded("rf_comb", rf, X_comb_train, y_train, X_comb_val, y_val, X_comb_test, y_test, logger)


if __name__ == "__main__":
    main()