from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

LabelMethod = Literal["median_threshold", "quantile_threshold"]


@dataclass(frozen=True)
class LabelingConfig:
    target_vol_col: str = "rv"
    label_col: str = "regime"


@dataclass(frozen=True)
class StaticThresholdConfig(LabelingConfig):
    method: LabelMethod = "median_threshold"
    quantile: float = 0.7


@dataclass
class ThresholdLabeler:
    cfg: StaticThresholdConfig
    threshold_: float | None = None

    def fit(self, df: pd.DataFrame) -> "ThresholdLabeler":
        v = df[self.cfg.target_vol_col].astype(float).to_numpy()
        if self.cfg.method == "median_threshold":
            thr = float(np.nanmedian(v))
        elif self.cfg.method == "quantile_threshold":
            if not (0.0 < self.cfg.quantile < 1.0):
                raise ValueError("quantile must be in (0, 1)")
            thr = float(np.nanquantile(v, self.cfg.quantile))
        else:
            raise ValueError(f"Unknown method: {self.cfg.method}")
        self.threshold_ = thr
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.threshold_ is None:
            raise RuntimeError("ThresholdLabeler is not fitted.")
        out = df.copy()
        v = out[self.cfg.target_vol_col].astype(float)
        out[self.cfg.label_col] = (v >= float(self.threshold_)).astype("Int64")
        return out


@dataclass(frozen=True)
class RollingQuantileConfig(LabelingConfig):
    lookback: int = 252
    quantile: float = 0.7
    min_periods: int | None = None


def label_regimes_rolling_quantile(df: pd.DataFrame, cfg: RollingQuantileConfig) -> pd.DataFrame:
    """
    NO-LEAK labeling:
    regime[t] uses threshold computed from history up to t-1.
    """
    v = df[cfg.target_vol_col].astype(float)
    minp = cfg.min_periods if cfg.min_periods is not None else cfg.lookback
    thr = v.rolling(cfg.lookback, min_periods=minp).quantile(cfg.quantile).shift(1)
    out = df.copy()
    out[cfg.label_col] = (v >= thr).astype("Int64")
    return out