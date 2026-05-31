"""
Compute standard rolling volatility features.

This module builds fast baseline features from log returns and realized
volatility. It includes absolute returns, squared returns, current realized
volatility, and rolling return and volatility summaries over configurable
window lengths.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StandardFeatureConfig:
    """Standard, fast rolling features for volatility/regime prediction."""

    return_col: str = "log_return"
    rv_col: str = "rv"
    windows: tuple[int, ...] = (5, 22, 60)


def standard_features(df: pd.DataFrame, cfg: StandardFeatureConfig | None = None) -> pd.DataFrame:
    cfg = cfg or StandardFeatureConfig()
    r = df[cfg.return_col].astype(float)
    rv = df[cfg.rv_col].astype(float)

    out = pd.DataFrame(index=df.index)

    out["ret_abs"] = r.abs()
    out["ret_sq"] = r.pow(2)
    out["rv"] = rv

    for w in cfg.windows:
        out[f"ret_mean_{w}"] = r.rolling(w, min_periods=w).mean()
        out[f"ret_std_{w}"] = r.rolling(w, min_periods=w).std(ddof=0)
        out[f"ret_abs_mean_{w}"] = r.abs().rolling(w, min_periods=w).mean()
        out[f"rv_mean_{w}"] = rv.rolling(w, min_periods=w).mean()
        out[f"rv_std_{w}"] = rv.rolling(w, min_periods=w).std(ddof=0)

    return out