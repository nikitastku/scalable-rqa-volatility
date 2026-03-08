from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class HARConfig:
    rv_col: str = "rv"
    regime_col: str = "regime"


def build_har_dataset(df: pd.DataFrame, rv_col: str = "rv", regime_col: str = "regime") -> pd.DataFrame:
    out = df.copy()

    rv = out[rv_col].astype(float)

    out["rv_daily"] = rv.shift(1)
    out["rv_weekly"] = rv.rolling(5).mean().shift(1)
    out["rv_monthly"] = rv.rolling(22).mean().shift(1)

    out["rv_next"] = rv.shift(-1)
    out["regime_next"] = out[regime_col].shift(-1).astype("Int64")

    out = out.dropna(subset=["rv_daily", "rv_weekly", "rv_monthly", "rv_next", "regime_next"]).reset_index(drop=True)
    out["regime_next"] = out["regime_next"].astype(int)
    return out


class HARRVModel:
    def __init__(self, cfg: HARConfig | None = None):
        self.cfg = cfg or HARConfig()
        self.model = LinearRegression()

    def fit(self, df: pd.DataFrame) -> None:
        X = df[["rv_daily", "rv_weekly", "rv_monthly"]].to_numpy()
        y = df["rv_next"].to_numpy()
        self.model.fit(X, y)

    def predict_rv_next(self, df: pd.DataFrame) -> np.ndarray:
        X = df[["rv_daily", "rv_weekly", "rv_monthly"]].to_numpy()
        return self.model.predict(X)