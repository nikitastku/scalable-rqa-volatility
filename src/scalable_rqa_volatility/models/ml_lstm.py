from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset


@dataclass(frozen=True)
class LSTMConfig:
    seq_len: int = 60
    feature_cols: tuple[str, ...] = ("log_return",)
    target_col: str = "rv"
    device: str = "cpu"
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def build_lstm_supervised(
    df: pd.DataFrame,
    cfg: LSTMConfig,
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build supervised sequences from df indices [start, end] (inclusive).
    For each t, X uses rows [t-seq_len+1 .. t], y is rv at t+1.
    Returns X, y, t_index (the anchor t indices in the original df).
    """
    if start < 0 or end >= len(df):
        raise ValueError("start/end out of bounds.")
    if (end - start + 1) <= cfg.seq_len:
        raise ValueError("Segment too short for seq_len.")

    feats = df.loc[:, cfg.feature_cols].astype(float).to_numpy()
    target = df.loc[:, cfg.target_col].astype(float).to_numpy()

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    t_list: list[int] = []

    first_t = start + cfg.seq_len - 1
    for t in range(first_t, end):
        x = feats[t - cfg.seq_len + 1 : t + 1]
        y = target[t + 1]
        if np.isfinite(x).all() and np.isfinite(y):
            X_list.append(x)
            y_list.append(float(y))
            t_list.append(t)

    X = np.stack(X_list, axis=0) if X_list else np.empty((0, cfg.seq_len, len(cfg.feature_cols)), dtype=float)
    y = np.asarray(y_list, dtype=float)
    t_index = np.asarray(t_list, dtype=int)
    return X, y, t_index


class VolatilityLSTM(nn.Module):
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