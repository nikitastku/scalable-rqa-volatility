"""
run_pipeline_dataset3.py — Process Yahoo Finance intraday S&P 500 data.

Input:  Parquet file with ~3M rows of 2-minute bars for 503 S&P 500 stocks.
Output: dataset3_train/val/test.parquet with features ready for model training.

Design decisions:
  - Each stock is an independent time series (~6,000 intraday bars per stock)
  - Rolling RV computed at intraday level (window=60 bars ≈ 2 hours)
  - Regime labeling: rolling quantile (lookback=975 bars ≈ 5 trading days, q=0.7)
  - Chronological split per stock (70/15/15), then pool across stocks
  - Standard features computed per stock before pooling
  - RQA/β-RQA features computed later in training scripts (too slow for preprocessing)

Estimated output: ~503 stocks × ~5,000 valid rows each ≈ 2.5M pooled rows.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scalable_rqa_volatility.logging_utils import get_logger
from scalable_rqa_volatility.utils.io import DataPaths, ensure_dirs, write_table
from scalable_rqa_volatility.utils.seed import set_global_seed


@dataclass(frozen=True)
class IntraDayConfig:
    rv_window: int = 60           # 60 bars × 2 min = 2 hours
    regime_lookback: int = 975    # 975 bars ≈ 5 trading days (195 bars/day × 5)
    regime_quantile: float = 0.7
    train_frac: float = 0.70
    val_frac: float = 0.15
    min_bars_per_stock: int = 1000  # skip stocks with too few bars
    std_windows: tuple[int, ...] = (30, 120, 390)  # ~1h, ~4h, ~2days in bars


def compute_features_for_ticker(
    df_ticker: pd.DataFrame,
    cfg: IntraDayConfig,
) -> pd.DataFrame | None:
    """
    Process a single ticker's intraday data:
      1. Compute log returns from close prices
      2. Compute rolling RV
      3. Label regimes
      4. Compute standard rolling features
      5. Return the processed DataFrame (or None if too few bars)
    """
    df = df_ticker.sort_values("timestamp").reset_index(drop=True)

    if len(df) < cfg.min_bars_per_stock:
        return None

    close = df["close"].astype(float)
    ticker = df["ticker"].iloc[0]

    # ── Log returns ──
    log_ret = np.log(close).diff()
    df["log_return"] = log_ret

    # ── Rolling RV (intraday) ──
    df["rv"] = log_ret.rolling(cfg.rv_window, min_periods=cfg.rv_window).std(ddof=0)

    # ── Regime labeling (rolling quantile, no-leak) ──
    rv = df["rv"].astype(float)
    thr = rv.rolling(cfg.regime_lookback, min_periods=cfg.regime_lookback).quantile(cfg.regime_quantile).shift(1)
    df["regime"] = (rv >= thr).astype("Int64")

    # ── Standard features (return-based + RV-based) ──
    r = df["log_return"].astype(float)
    rv_s = df["rv"].astype(float)

    df["ret_abs"] = r.abs()
    df["ret_sq"] = r.pow(2)

    for w in cfg.std_windows:
        df[f"ret_mean_{w}"] = r.rolling(w, min_periods=w).mean()
        df[f"ret_std_{w}"] = r.rolling(w, min_periods=w).std(ddof=0)
        df[f"ret_abs_mean_{w}"] = r.abs().rolling(w, min_periods=w).mean()
        df[f"rv_mean_{w}"] = rv_s.rolling(w, min_periods=w).mean()
        df[f"rv_std_{w}"] = rv_s.rolling(w, min_periods=w).std(ddof=0)

    # ── Drop NaN rows ──
    required_cols = ["log_return", "rv", "regime"]
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    if len(df) < 100:
        return None

    # ── Add ticker column ──
    df["ticker"] = ticker

    return df


def chronological_split_per_stock(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split each ticker's data chronologically, then concatenate across tickers.
    This ensures no future leakage within any stock.
    """
    trains, vals, tests = [], [], []

    for ticker, group in df.groupby("ticker"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        n = len(group)
        n_tr = int(n * train_frac)
        n_va = int(n * val_frac)

        trains.append(group.iloc[:n_tr])
        vals.append(group.iloc[n_tr:n_tr + n_va])
        tests.append(group.iloc[n_tr + n_va:])

    train = pd.concat(trains, ignore_index=True)
    val = pd.concat(vals, ignore_index=True)
    test = pd.concat(tests, ignore_index=True)

    return train, val, test


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Preprocess Yahoo Finance intraday data")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to input parquet file. Default: data/raw/sp500_intraday.parquet")
    parser.add_argument("--max_tickers", type=int, default=None,
                        help="Process only first N tickers (for testing). Default: all")
    args = parser.parse_args()

    logger = get_logger()
    set_global_seed(42)
    cfg = IntraDayConfig()

    paths = DataPaths.from_repo_root(repo_root)
    ensure_dirs(paths.raw, paths.interim, paths.processed)

    # ── Load ──
    input_path = Path(args.input) if args.input else (paths.raw / "sp500_intraday.parquet")
    logger.info(f"Loading intraday data from {input_path}")
    t0 = time.time()
    raw = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(raw):,} rows, {raw['ticker'].nunique()} tickers in {time.time()-t0:.1f}s")

    # ── Ensure timestamp is datetime ──
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)

    # ── Optional ticker limit (for testing) ──
    tickers = sorted(raw["ticker"].unique())
    if args.max_tickers:
        tickers = tickers[:args.max_tickers]
        raw = raw[raw["ticker"].isin(tickers)]
        logger.info(f"Limited to {len(tickers)} tickers for testing")

    # ── Process each ticker ──
    logger.info(f"Processing {len(tickers)} tickers (rv_window={cfg.rv_window}, "
                f"regime_lookback={cfg.regime_lookback}, q={cfg.regime_quantile})")

    processed_dfs = []
    skipped = 0
    t0 = time.time()

    for i, ticker in enumerate(tickers):
        df_ticker = raw[raw["ticker"] == ticker].copy()
        result = compute_features_for_ticker(df_ticker, cfg)

        if result is None:
            skipped += 1
            continue

        processed_dfs.append(result)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            logger.info(f"  Processed {i+1}/{len(tickers)} tickers ({elapsed:.1f}s, "
                        f"{skipped} skipped, {sum(len(d) for d in processed_dfs):,} total rows)")

    if not processed_dfs:
        raise RuntimeError("No tickers produced valid data!")

    full = pd.concat(processed_dfs, ignore_index=True)
    elapsed = time.time() - t0
    logger.info(f"All tickers processed in {elapsed:.1f}s: {len(full):,} total rows, "
                f"{len(tickers)-skipped} valid tickers, {skipped} skipped")

    # ── Dataset statistics ──
    logger.info({
        "total_rows": len(full),
        "valid_tickers": len(tickers) - skipped,
        "skipped_tickers": skipped,
        "log_return_mean": float(full["log_return"].mean()),
        "log_return_std": float(full["log_return"].std()),
        "rv_mean": float(full["rv"].mean()),
        "rv_std": float(full["rv"].std()),
        "regime_mean": float(full["regime"].astype(float).mean()),
        "rows_per_ticker_mean": float(full.groupby("ticker").size().mean()),
        "rows_per_ticker_min": int(full.groupby("ticker").size().min()),
        "rows_per_ticker_max": int(full.groupby("ticker").size().max()),
    })

    # ── Split per stock, then pool ──
    logger.info(f"Splitting chronologically per stock ({cfg.train_frac}/{cfg.val_frac}/"
                f"{1-cfg.train_frac-cfg.val_frac:.2f})")
    train, val, test = chronological_split_per_stock(full, cfg.train_frac, cfg.val_frac)

    for name, part in [("train", train), ("val", val), ("test", test)]:
        rv_split = part["rv"].astype(float)
        regime_split = part["regime"].astype(float)
        logger.info({
            f"{name}_rows": len(part),
            f"{name}_tickers": part["ticker"].nunique(),
            f"{name}_rv_mean": float(rv_split.mean()),
            f"{name}_regime_mean": float(regime_split.mean()),
        })

    # Distribution shift check
    train_rv = train["rv"].astype(float).mean()
    test_rv = test["rv"].astype(float).mean()
    logger.info({"test_train_rv_ratio": float(test_rv / train_rv)})

    # ── Save ──
    for name, part in [("train", train), ("val", val), ("test", test)]:
        out_path = paths.processed / f"dataset3_{name}.parquet"
        write_table(part, out_path, fmt="parquet")
        logger.info(f"Saved {name}: {out_path} rows={len(part):,}")

    summary = pd.DataFrame({
        "split": ["train", "val", "test"],
        "rows": [len(train), len(val), len(test)],
        "tickers": [train["ticker"].nunique(), val["ticker"].nunique(), test["ticker"].nunique()],
        "regime_mean": [
            float(train["regime"].astype(float).mean()),
            float(val["regime"].astype(float).mean()),
            float(test["regime"].astype(float).mean()),
        ],
    })
    write_table(summary, paths.processed / "dataset3_summary.csv", fmt="csv")
    logger.info("Done! Saved dataset3_train/val/test.parquet and dataset3_summary.csv")


if __name__ == "__main__":
    main()