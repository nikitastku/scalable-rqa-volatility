"""
run_pipeline_dataset2.py — Process the S&P 500 Macro-Financial Volatility Dataset.

This dataset contains real S&P 500 daily data (2010–2019) with 31 columns
including cross-market indices, options data, macro indicators, and commodities.

We process it to match Dataset 1's format:
  - log_return: from SP500 close prices (or use the provided column)
  - rv: 20-day rolling std of log_return (computed fresh, NOT the provided 30-day vol)
  - regime: rolling quantile labeling (lookback=252, q=0.7, no-leak)
  - Chronological 70/15/15 split

Key difference from Dataset 1: test/train RV ratio ≈ 0.96x (no distribution shift).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scalable_rqa_volatility.data.splits import SplitConfig, time_series_split
from scalable_rqa_volatility.logging_utils import get_logger
from scalable_rqa_volatility.utils.io import DataPaths, ensure_dirs, write_table
from scalable_rqa_volatility.utils.seed import set_global_seed
from scalable_rqa_volatility.volatility.labeling import RollingQuantileConfig, label_regimes_rolling_quantile


@dataclass(frozen=True)
class PipelineConfig:
    repo_root: Path
    dataset_path: Path
    seed: int = 42
    rv_window: int = 20


def load_sp500_macro(path: Path) -> pd.DataFrame:
    """
    Load the S&P 500 Macro-Financial Volatility Dataset.
    Parses dates (dd/mm/yyyy format), sorts chronologically,
    and renames columns to match our internal schema.
    """
    df = pd.read_csv(path)

    # Parse date — the dataset uses dd/mm/yyyy format
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="raise")
    df = df.sort_values("Date").reset_index(drop=True)

    if df["Date"].isna().any():
        raise ValueError("Found NaT values in Date column after parsing.")

    return df


def add_returns_and_rv(df: pd.DataFrame, rv_window: int = 20) -> pd.DataFrame:
    """
    Compute log_return and rv from SP500 close prices.

    We compute our OWN log_return and rv to be fully consistent with Dataset 1,
    rather than using the provided 'SP500 Log Returns' and 'SP500 30 Day Volatility'
    columns (which use a different scale/estimator).

    However, we validate against the provided log returns to ensure consistency.
    """
    out = df.copy()

    # Compute log return from SP500 close
    close = out["SP500"].astype(float)
    computed_log_ret = np.log(close).diff()

    # Validate against provided log returns
    provided_log_ret = out["SP500 Log Returns"].astype(float)
    valid_mask = computed_log_ret.notna() & provided_log_ret.notna()
    if valid_mask.sum() > 0:
        corr = computed_log_ret[valid_mask].corr(provided_log_ret[valid_mask])
        max_diff = (computed_log_ret[valid_mask] - provided_log_ret[valid_mask]).abs().max()
        if corr < 0.999 or max_diff > 0.001:
            print(f"  WARNING: Computed vs provided log returns differ (corr={corr:.4f}, max_diff={max_diff:.6f})")
            print(f"  Using provided log returns for consistency with dataset.")
            out["log_return"] = provided_log_ret
        else:
            out["log_return"] = computed_log_ret
    else:
        out["log_return"] = computed_log_ret

    # Compute 20-day rolling RV (same as Dataset 1)
    out["rv"] = (
        out["log_return"]
        .rolling(rv_window, min_periods=rv_window)
        .std(ddof=0)
        .astype(float)
    )

    return out


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = PipelineConfig(
        repo_root=repo_root,
        dataset_path=repo_root / "data" / "raw" / "S_P_500_Macro-Financial_Volatility_Dataset.csv",
    )

    logger = get_logger()
    set_global_seed(cfg.seed)

    paths = DataPaths.from_repo_root(cfg.repo_root)
    ensure_dirs(paths.raw, paths.interim, paths.processed)

    # ── Load ──
    logger.info(f"Loading dataset: {cfg.dataset_path}")
    df = load_sp500_macro(cfg.dataset_path)
    logger.info(f"Rows={len(df)} Cols={len(df.columns)} DateRange={df['Date'].min().date()}..{df['Date'].max().date()}")

    # ── Compute returns and RV ──
    logger.info(f"Computing log_return and rv (window={cfg.rv_window})")
    df = add_returns_and_rv(df, rv_window=cfg.rv_window)
    df = df.dropna(subset=["log_return", "rv"]).reset_index(drop=True)
    logger.info(f"After dropna: {len(df)} rows")

    # ── Log dataset statistics ──
    log_ret = df["log_return"]
    rv = df["rv"]
    logger.info({
        "log_return_mean": float(log_ret.mean()),
        "log_return_std": float(log_ret.std()),
        "log_return_kurtosis": float(log_ret.kurtosis()),
        "log_return_skew": float(log_ret.skew()),
        "rv_mean": float(rv.mean()),
        "rv_std": float(rv.std()),
        "rv_min": float(rv.min()),
        "rv_max": float(rv.max()),
        "rv_autocorr_lag1": float(rv.autocorr(1)),
        "rv_autocorr_lag5": float(rv.autocorr(5)),
        "rv_autocorr_lag22": float(rv.autocorr(22)),
    })

    # ── Label regimes (same methodology as Dataset 1) ──
    logger.info("Labeling regimes (rolling quantile, lookback=252, q=0.7, no-leak)")
    df = label_regimes_rolling_quantile(
        df,
        RollingQuantileConfig(
            target_vol_col="rv",
            label_col="regime",
            lookback=252,
            quantile=0.7,
            min_periods=252,
        ),
    )
    df = df.dropna(subset=["regime"]).reset_index(drop=True)
    logger.info(f"After regime labeling dropna: {len(df)} rows, regime_mean={df['regime'].mean():.3f}")

    # ── Split ──
    splits = time_series_split(df, SplitConfig(train_frac=0.7, val_frac=0.15))

    # ── Log split statistics ──
    for name, part in splits.items():
        rv_split = part["rv"].astype(float)
        regime_split = part["regime"].astype(float)
        logger.info({
            f"{name}_rows": len(part),
            f"{name}_date_range": f"{part['Date'].min().date()}..{part['Date'].max().date()}",
            f"{name}_rv_mean": float(rv_split.mean()),
            f"{name}_rv_std": float(rv_split.std()),
            f"{name}_regime_mean": float(regime_split.mean()),
        })

    # Distribution shift check
    train_rv = splits["train"]["rv"].astype(float).mean()
    test_rv = splits["test"]["rv"].astype(float).mean()
    logger.info({"test_train_rv_ratio": float(test_rv / train_rv)})

    # ── Save ──
    for name, part in splits.items():
        out_path = paths.processed / f"dataset2_{name}.parquet"
        write_table(part, out_path, fmt="parquet")
        logger.info(f"Saved {name}: {out_path} rows={len(part)}")

    summary = pd.DataFrame(
        {
            "split": list(splits.keys()),
            "rows": [len(splits[k]) for k in splits.keys()],
            "regime_mean": [float(splits[k]["regime"].mean()) for k in splits.keys()],
        }
    )
    write_table(summary, paths.processed / "dataset2_summary.csv", fmt="csv")
    logger.info("Saved summary: data/processed/dataset2_summary.csv")


if __name__ == "__main__":
    main()