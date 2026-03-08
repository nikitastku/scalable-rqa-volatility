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
from scalable_rqa_volatility.recurrence.beta_rqa import BetaRQAConfig, beta_rqa_features_rolling
from scalable_rqa_volatility.recurrence.embeddings import EmbeddingConfig
from scalable_rqa_volatility.recurrence.rqa import RQAConfig, rqa_features_rolling
from scalable_rqa_volatility.volatility.features_standard import StandardFeatureConfig, standard_features


# ──────────────────────────────────────────────────────────────
# EXPERIMENT VARIABLE: β values to sweep
#   β = 0   → IS-divergence (scale-invariant, emphasizes ratios)
#   β = 0.5 → intermediate
#   β = 1   → KL-divergence (emphasizes small-value differences)
#   β = 1.5 → intermediate
#   β = 2   → Euclidean / least-squares (symmetric, same as standard RQA)
# ──────────────────────────────────────────────────────────────
BETA_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0]


@dataclass(frozen=True)
class BetaBaselineConfig:
    std: StandardFeatureConfig = StandardFeatureConfig(windows=(5, 22, 60))
    rqa_cols: tuple[str, ...] = ("log_return",)
    label_col: str = "regime"
    rqa: RQAConfig = RQAConfig(window=90, step=5, recurrence_rate=0.1, embed=EmbeddingConfig(m=4, tau=2))


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


def build_xy_within_split(df_split: pd.DataFrame, X_split: pd.DataFrame, label_col: str) -> tuple[pd.DataFrame, np.ndarray]:
    y = df_split[label_col].shift(-1)
    mask = y.notna().to_numpy()
    X2 = X_split.iloc[: len(mask)].copy()
    X2 = X2.iloc[mask].replace([np.inf, -np.inf], np.nan)
    y2 = y.iloc[: len(mask)].iloc[mask].astype(int)

    keep = X2.notna().all(axis=1).to_numpy()
    X3 = X2.iloc[keep].copy()
    y3 = y2.iloc[keep].to_numpy()
    return X3, y3


def align_two(
    X_a: pd.DataFrame,
    X_b: pd.DataFrame,
    y: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    n = min(len(X_a), len(X_b), len(y))
    return X_a.iloc[:n].copy(), X_b.iloc[:n].copy(), y[:n]


def f1_fast(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.int8, copy=False)
    y_pred = y_pred.astype(np.int8, copy=False)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    denom = 2 * tp + fp + fn
    return float((2 * tp) / denom) if denom else 0.0


def best_threshold_target_rate(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    """UNIFIED: same function as in train_features_baselines.py"""
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

    return float(best_t), float(best_f1)


def fit_thresholded(
    name: str,
    model,
    Xtr: pd.DataFrame,
    ytr: np.ndarray,
    Xva: pd.DataFrame,
    yva: np.ndarray,
    Xte: pd.DataFrame,
    yte: np.ndarray,
    logger,
) -> None:
    model.fit(Xtr, ytr)

    p_val = model.predict_proba(Xva)[:, 1]
    thr, val_f1 = best_threshold_target_rate(yva, p_val)

    p_test = model.predict_proba(Xte)[:, 1]
    pred = (p_test >= thr).astype(int)

    metrics = classification_metrics(yte, pred, p_test)
    cm = confusion_matrix(yte, pred)

    logger.info(
        {
            f"{name}_threshold": float(thr),
            f"{name}_val_f1_at_threshold": float(val_f1),
            f"{name}_metrics": metrics,
            f"{name}_cm": cm.tolist(),
        }
    )


def main() -> None:
    logger = get_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=1.0,
                        help=f"β-divergence parameter. Suggested sweep: {BETA_VALUES}")
    args = parser.parse_args()

    cfg = BetaBaselineConfig()

    beta_rqa_cfg = BetaRQAConfig(
        window=cfg.rqa.window,
        step=cfg.rqa.step,
        recurrence_rate=cfg.rqa.recurrence_rate,
        embed=cfg.rqa.embed,
        beta=args.beta,
        transform="minmax",
    )

    train = load_split("train").reset_index(drop=True)
    val = load_split("val").reset_index(drop=True)
    test = load_split("test").reset_index(drop=True)

    full_all = pd.concat([train, val, test], axis=0, ignore_index=True)

    n_train, n_val, n_test = len(train), len(val), len(test)
    sl = split_slices(n_train, n_val, n_test)

    X_std_all = standard_features(full_all, cfg.std)
    X_beta_all = beta_rqa_features_rolling(full_all, cfg.rqa_cols, beta_rqa_cfg, prefix="beta_rqa")

    X_std_train = X_std_all.iloc[sl["train"]].reset_index(drop=True)
    X_std_val = X_std_all.iloc[sl["val"]].reset_index(drop=True)
    X_std_test = X_std_all.iloc[sl["test"]].reset_index(drop=True)

    X_beta_train = X_beta_all.iloc[sl["train"]].reset_index(drop=True)
    X_beta_val = X_beta_all.iloc[sl["val"]].reset_index(drop=True)
    X_beta_test = X_beta_all.iloc[sl["test"]].reset_index(drop=True)

    X_std_train, y_train = build_xy_within_split(train, X_std_train, cfg.label_col)
    X_std_val, y_val = build_xy_within_split(val, X_std_val, cfg.label_col)
    X_std_test, y_test = build_xy_within_split(test, X_std_test, cfg.label_col)

    X_beta_train, y_train_beta = build_xy_within_split(train, X_beta_train, cfg.label_col)
    X_beta_val, y_val_beta = build_xy_within_split(val, X_beta_val, cfg.label_col)
    X_beta_test, y_test_beta = build_xy_within_split(test, X_beta_test, cfg.label_col)

    # Align
    X_std_train, X_beta_train, y_train = align_two(X_std_train, X_beta_train, y_train)
    X_std_val, X_beta_val, y_val = align_two(X_std_val, X_beta_val, y_val)
    X_std_test, X_beta_test, y_test = align_two(X_std_test, X_beta_test, y_test)

    X_comb_train = pd.concat([X_std_train.reset_index(drop=True), X_beta_train.reset_index(drop=True)], axis=1)
    X_comb_val = pd.concat([X_std_val.reset_index(drop=True), X_beta_val.reset_index(drop=True)], axis=1)
    X_comb_test = pd.concat([X_std_test.reset_index(drop=True), X_beta_test.reset_index(drop=True)], axis=1)

    logreg = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(max_iter=2000, n_jobs=None, class_weight="balanced")),
        ]
    )

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=42,
    )

    logger.info(
        {
            "beta": float(args.beta),
            "beta_transform": str(beta_rqa_cfg.transform),
            "beta_window": int(beta_rqa_cfg.window),
            "beta_step": int(beta_rqa_cfg.step),
            "beta_embed_m": int(beta_rqa_cfg.embed.m),
            "beta_embed_tau": int(beta_rqa_cfg.embed.tau),
            "std_features": int(X_std_train.shape[1]),
            "beta_rqa_features": int(X_beta_train.shape[1]),
            "combined_features": int(X_comb_train.shape[1]),
            "train_prevalence": float(np.mean(y_train)) if len(y_train) else float("nan"),
            "val_prevalence": float(np.mean(y_val)) if len(y_val) else float("nan"),
            "test_prevalence": float(np.mean(y_test)) if len(y_test) else float("nan"),
        }
    )

    # β-RQA only
    fit_thresholded("logreg_beta_rqa", logreg, X_beta_train, y_train, X_beta_val, y_val, X_beta_test, y_test, logger)
    fit_thresholded("rf_beta_rqa", rf, X_beta_train, y_train, X_beta_val, y_val, X_beta_test, y_test, logger)

    # Standard + β-RQA combined
    fit_thresholded("logreg_std_beta", logreg, X_comb_train, y_train, X_comb_val, y_val, X_comb_test, y_test, logger)
    fit_thresholded("rf_std_beta", rf, X_comb_train, y_train, X_comb_val, y_val, X_comb_test, y_test, logger)


if __name__ == "__main__":
    main()