"""
Compute return and realized-volatility features.

This module adds log returns, high-low price range, and a rolling realized-
volatility proxy to OHLC price data. The output columns and rolling volatility
window are controlled through a small configuration dataclass.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VolatilityConfig:
    """Configuration for volatility/return computations."""

    date_col: str = "Date"
    close_col: str = "Close_Price"
    high_col: str = "High_Price"
    low_col: str = "Low_Price"
    return_col: str = "log_return"
    rv_col: str = "rv"
    range_col: str = "hl_range"
    rv_window: int = 20


def add_returns_and_volatility(df: pd.DataFrame, cfg: VolatilityConfig | None = None) -> pd.DataFrame:
    """Add log returns, HL range and rolling realized volatility proxy."""
    cfg = cfg or VolatilityConfig()
    out = df.copy()

    close = out[cfg.close_col].astype(float)
    out[cfg.return_col] = np.log(close).diff()

    out[cfg.range_col] = (out[cfg.high_col].astype(float) - out[cfg.low_col].astype(float)).abs()

    out[cfg.rv_col] = (
        out[cfg.return_col]
        .rolling(cfg.rv_window, min_periods=cfg.rv_window)
        .std(ddof=0)
        .astype(float)
    )

    return out