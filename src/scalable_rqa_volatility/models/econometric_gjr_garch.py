from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model


@dataclass(frozen=True)
class GJRGarchConfig:
    """Configuration for fitting and forecasting a GJR-GARCH(1,1) model."""

    return_col: str = "log_return"
    scale: float = 100.0
    p: int = 1
    o: int = 1
    q: int = 1
    dist: str = "t"


class GJRGarchModel:
    """GJR-GARCH volatility forecaster using fixed parameters estimated on a training window."""

    def __init__(self, cfg: GJRGarchConfig | None = None):
        self.cfg = cfg or GJRGarchConfig()
        self._result = None
        self._train_last_obs = None

    @property
    def is_fitted(self) -> bool:
        return self._result is not None and self._train_last_obs is not None

    def fit(self, returns_full: pd.Series, train_last_obs: int) -> None:
        """
        Fit GJR-GARCH parameters using observations up to train_last_obs (inclusive),
        while keeping the full series available for out-of-sample forecasting.
        """
        y = pd.to_numeric(returns_full, errors="raise").astype(float) * self.cfg.scale
        if train_last_obs < 5:
            raise ValueError("train_last_obs too small for GARCH estimation.")

        am = arch_model(
            y,
            mean="zero",
            vol="GARCH",
            p=self.cfg.p,
            o=self.cfg.o,
            q=self.cfg.q,
            dist=self.cfg.dist,
            rescale=False,
        )
        self._result = am.fit(disp="off", last_obs=train_last_obs)
        self._train_last_obs = int(train_last_obs)

    def forecast_sigma_one_step(self, start: int, end: int) -> np.ndarray:
        """
        Return one-step-ahead conditional sigma forecasts for indices [start, end] (inclusive),
        aligned so that sigma[t] forecasts volatility for t+1.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        fcst = self._result.forecast(horizon=1, start=start, reindex=False)
        var = fcst.variance.iloc[:, 0].to_numpy()
        sigma = np.sqrt(var)
        sigma = sigma[: (end - start + 1)]
        return sigma