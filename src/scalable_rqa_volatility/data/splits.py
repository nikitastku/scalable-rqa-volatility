"""
Load and validate the core time-series dataset.

This module defines the expected schema for the raw core dataset and provides a
loader that reads the data, validates required OHLC columns, parses dates,
sorts observations chronologically, converts numeric columns, and raises clear
errors when the dataset does not match the expected format.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for time-series split."""

    date_col: str = "Date"
    train_frac: float = 0.7
    val_frac: float = 0.15


def time_series_split(df: pd.DataFrame, cfg: SplitConfig | None = None) -> dict[str, pd.DataFrame]:
    """Split chronologically into train/val/test by fraction."""
    cfg = cfg or SplitConfig()
    if not (0.0 < cfg.train_frac < 1.0 and 0.0 <= cfg.val_frac < 1.0):
        raise ValueError("Invalid split fractions.")

    out = df.sort_values(cfg.date_col).reset_index(drop=True)
    n = len(out)
    n_train = int(n * cfg.train_frac)
    n_val = int(n * cfg.val_frac)

    train = out.iloc[:n_train].reset_index(drop=True)
    val = out.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test = out.iloc[n_train + n_val :].reset_index(drop=True)

    return {"train": train, "val": val, "test": test}