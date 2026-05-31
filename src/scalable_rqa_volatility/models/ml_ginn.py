"""
Define the GARCH-informed neural network components.

This module provides the configuration, neural network architecture, GJR-GARCH
teacher-volatility fitting, volatility-scale mapping, and sequence-building
utilities used by the GINN training pipeline. The GINN model predicts realized
volatility from sequential features while optionally using a GJR-GARCH teacher
signal as an auxiliary guide during training.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from arch import arch_model
from torch import nn


@dataclass(frozen=True)
class GINNConfig:
    seq_len: int = 60
    feature_cols: tuple[str, ...] = ("log_return", "rv")
    return_col: str = "log_return"
    rv_col: str = "rv"
    garch_scale: float = 100.0
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    lambda_garch: float = 0.3


class GINNNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        y = self.head(last).squeeze(-1)
        return y


def fit_gjr_garch_teacher_sigma(
    returns_full: pd.Series,
    train_last_obs: int,
    scale: float = 100.0,
) -> np.ndarray:
    y = pd.to_numeric(returns_full, errors="raise").astype(float).to_numpy() * float(scale)
    am = arch_model(y, mean="zero", vol="GARCH", p=1, o=1, q=1, dist="t", rescale=False)
    res = am.fit(disp="off", last_obs=int(train_last_obs))
    fc = res.forecast(horizon=1, start=0, reindex=False)
    var = fc.variance.iloc[:, 0].to_numpy()
    sigma = np.sqrt(var)
    return sigma


def map_sigma_to_rv(
    sigma: np.ndarray,
    rv_next: np.ndarray,
) -> float:
    s = sigma.reshape(-1)
    y = rv_next.reshape(-1)
    m = np.isfinite(s) & np.isfinite(y)
    if m.sum() == 0:
        return 1.0
    denom = float(np.dot(s[m], s[m]))
    if denom == 0.0:
        return 1.0
    a = float(np.dot(s[m], y[m]) / denom)
    return a


def build_sequences_with_targets(
    df: pd.DataFrame,
    cfg: GINNConfig,
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feats = df.loc[:, cfg.feature_cols].astype(float).to_numpy()
    rv = df.loc[:, cfg.rv_col].astype(float).to_numpy()

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    t_list: list[int] = []

    first_t = start + cfg.seq_len - 1
    for t in range(first_t, end):
        x = feats[t - cfg.seq_len + 1 : t + 1]
        y = rv[t + 1]
        if np.isfinite(x).all() and np.isfinite(y):
            X_list.append(x)
            y_list.append(float(y))
            t_list.append(t)

    X = np.stack(X_list, axis=0) if X_list else np.empty((0, cfg.seq_len, len(cfg.feature_cols)), dtype=float)
    y = np.asarray(y_list, dtype=float)
    t_idx = np.asarray(t_list, dtype=int)
    return X, y, t_idx, feats