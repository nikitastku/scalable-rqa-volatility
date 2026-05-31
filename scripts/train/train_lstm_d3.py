"""
LSTM for Dataset 3 (intraday multi-stock).

Key adaptations:
  - Standard features are precomputed (uses STD_FEATURE_COLS from parquet)
  - Sequences built per-stock (no cross-stock leakage)
  - Larger dataset → smaller batch size won't help, but seq_len=60 still works
  - Same improvements as D1/D2: RV features, LR scheduler, gradient clipping
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset

from scalable_rqa_volatility.evaluation.metrics import classification_metrics
from scalable_rqa_volatility.logging_utils import get_logger
from scalable_rqa_volatility.models.ml_lstm import VolatilityLSTM
from scalable_rqa_volatility.utils.seed import set_global_seed


STD_FEATURE_COLS = [
    "ret_abs", "ret_sq", "rv",
    "ret_mean_30", "ret_std_30", "ret_abs_mean_30", "rv_mean_30", "rv_std_30",
    "ret_mean_120", "ret_std_120", "ret_abs_mean_120", "rv_mean_120", "rv_std_120",
    "ret_mean_390", "ret_std_390", "ret_abs_mean_390", "rv_mean_390", "rv_std_390",
]

FEATURE_COLS = ["log_return"] + STD_FEATURE_COLS 


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 30
    batch_size: int = 512       
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 8
    num_workers: int = 0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    max_grad_norm: float = 1.0


@dataclass(frozen=True)
class ModelConfig:
    seq_len: int = 60
    hidden_size: int = 96
    num_layers: int = 2
    dropout: float = 0.1
    device: str = "cuda"
    eps: float = 1e-12
    thr_lookback: int = 975     
    thr_q: float = 0.7


class NumpySeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
    def __len__(self): return int(self.X.shape[0])
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_split(name: str) -> pd.DataFrame:
    return pd.read_parquet(repo_root() / "data" / "processed" / f"dataset3_{name}.parquet")


def as_device(device: str) -> torch.device:
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_supervised_per_stock(
    df: pd.DataFrame,
    feat_cols: list[str],
    seq_len: int,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build supervised sequences per stock using vectorized sliding windows."""
    all_X, all_y = [], []
    n_feat = len(feat_cols)

    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker
        df_t = df.loc[mask].reset_index(drop=True)
        n = len(df_t)

        if n <= seq_len:
            continue

        feats = df_t[feat_cols].to_numpy(dtype=np.float32)
        rv = df_t["rv"].astype(np.float32).to_numpy()

        shape = (n - seq_len, seq_len, n_feat)
        strides = (feats.strides[0], feats.strides[0], feats.strides[1])
        windows = np.lib.stride_tricks.as_strided(feats, shape=shape, strides=strides).copy()

        y_rv = rv[seq_len:]  
        y_log = np.log(y_rv.astype(np.float32) + eps)

        valid = (np.isfinite(windows).reshape(len(windows), -1).all(axis=1)
                 & np.isfinite(y_log))

        if valid.sum() > 0:
            all_X.append(windows[valid])
            all_y.append(y_log[valid])

    if not all_X:
        return np.empty((0, seq_len, n_feat), dtype=np.float32), np.empty((0,), dtype=np.float32)

    return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)


def rolling_threshold_from_history(rv: np.ndarray, lookback: int, q: float) -> np.ndarray:
    s = pd.Series(rv)
    return s.rolling(lookback, min_periods=lookback).quantile(q).shift(1).to_numpy(dtype=float)


def fit_model(model, tr_loader, va_loader, device, cfg):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=cfg.scheduler_factor, patience=cfg.scheduler_patience)

    best_val, best_state, bad = float("inf"), None, 0
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(cfg.epochs):
        model.train()
        for X, y in tr_loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(X)
                loss = loss_fn(pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_grad_norm)
            scaler.step(opt)
            scaler.update()

        model.eval()
        vals = []
        with torch.no_grad():
            for X, y in va_loader:
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    v = loss_fn(model(X), y).item()
                vals.append(v)
        val_loss = float(np.mean(vals)) if vals else float("inf")
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return {"best_val_mse": float(best_val), "stopped_epoch": epoch + 1}


def predict(model, loader, device):
    model.eval()
    out = []
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for X, _ in loader:
            X = X.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                y = model(X).detach().cpu().numpy()
            out.append(y)
    return np.concatenate(out) if out else np.empty((0,))


def best_offset_target_rate(scores, y_true):
    scores = np.asarray(scores, dtype=float).reshape(-1)
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    if scores.size == 0:
        return 0.0
    target = float(np.clip(np.mean(y_true), 0.05, 0.95))
    qs = np.linspace(0.01, 0.99, 199)
    grid = np.unique(np.quantile(scores, qs))
    best_b, best_f1 = 0.0, -1.0
    for b in grid:
        y_pred = (scores - float(b) >= 0.0).astype(int)
        pos = float(np.mean(y_pred))
        if not (target * 0.5 <= pos <= min(0.99, target * 1.5)):
            continue
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        denom = 2 * tp + fp + fn
        f1 = float((2 * tp) / denom) if denom else 0.0
        if f1 > best_f1:
            best_f1, best_b = f1, float(b)
    if best_f1 < 0.0:
        best_b = float(np.quantile(scores, 1.0 - target))
    return float(best_b)


def main() -> None:
    logger = get_logger()
    set_global_seed(42)
    torch.manual_seed(42)

    cfg = ModelConfig()
    train_cfg = TrainConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_tickers", type=int, default=100,
                        help="Max tickers for LSTM (memory constraint). Default=100")
    args = parser.parse_args()

    train = load_split("train")
    val = load_split("val")
    test = load_split("test")

    all_tickers = sorted(set(train["ticker"].unique()) & set(val["ticker"].unique()) & set(test["ticker"].unique()))
    if args.max_tickers and len(all_tickers) > args.max_tickers:
        np.random.seed(42)
        selected = sorted(np.random.choice(all_tickers, args.max_tickers, replace=False))
        train = train[train["ticker"].isin(selected)].reset_index(drop=True)
        val = val[val["ticker"].isin(selected)].reset_index(drop=True)
        test = test[test["ticker"].isin(selected)].reset_index(drop=True)
        logger.info(f"Subsampled to {args.max_tickers} tickers for memory")

    logger.info(f"Train: {len(train):,}, Val: {len(val):,}, Test: {len(test):,}")
    logger.info(f"Building sequences per stock (seq_len={cfg.seq_len}, {len(FEATURE_COLS)} features)...")

    Xtr, ytr = build_supervised_per_stock(train, FEATURE_COLS, cfg.seq_len, cfg.eps)
    Xva, yva = build_supervised_per_stock(val, FEATURE_COLS, cfg.seq_len, cfg.eps)
    Xte, yte = build_supervised_per_stock(test, FEATURE_COLS, cfg.seq_len, cfg.eps)

    logger.info(f"Sequences — Train: {len(Xtr):,}, Val: {len(Xva):,}, Test: {len(Xte):,}")

    if len(Xtr) == 0 or len(Xva) == 0 or len(Xte) == 0:
        raise RuntimeError("No sequences produced.")

    logger.info("Standardizing features...")
    n_feat = Xtr.shape[2]
    for f in range(n_feat):
        col = Xtr[:, :, f].ravel()
        m = float(np.nanmean(col[:min(500000, len(col))]))
        s = float(np.nanstd(col[:min(500000, len(col))]))
        if s == 0.0:
            s = 1.0
        Xtr[:, :, f] = (Xtr[:, :, f] - m) / s
        Xva[:, :, f] = (Xva[:, :, f] - m) / s
        Xte[:, :, f] = (Xte[:, :, f] - m) / s
    logger.info("Standardization done.")

    y_mu, y_sd = float(np.mean(ytr)), float(np.std(ytr))
    if y_sd == 0.0: y_sd = 1.0
    ytr_s = (ytr - y_mu) / y_sd
    yva_s = (yva - y_mu) / y_sd
    yte_s = (yte - y_mu) / y_sd

    device = as_device(cfg.device)
    pin = device.type == "cuda"
    torch.backends.cudnn.benchmark = bool(pin)

    tr_loader = DataLoader(NumpySeqDataset(Xtr, ytr_s), batch_size=train_cfg.batch_size, shuffle=True, pin_memory=pin)
    va_loader = DataLoader(NumpySeqDataset(Xva, yva_s), batch_size=train_cfg.batch_size, shuffle=False, pin_memory=pin)
    te_loader = DataLoader(NumpySeqDataset(Xte, yte_s), batch_size=train_cfg.batch_size, shuffle=False, pin_memory=pin)

    model = VolatilityLSTM(
        input_size=len(FEATURE_COLS), hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers, dropout=cfg.dropout).to(device)

    fit_info = fit_model(model, tr_loader, va_loader, device, train_cfg)
    logger.info(fit_info)

    log_pred_te = predict(model, te_loader, device) * y_sd + y_mu
    pred_rv_te = np.exp(log_pred_te)

    log_pred_va = predict(model, va_loader, device) * y_sd + y_mu

    def get_regime_labels(split_df, seq_len):
        labels = []
        for ticker in split_df["ticker"].unique():
            mask = split_df["ticker"] == ticker
            regime = split_df.loc[mask, "regime"].astype(int).to_numpy()
            if len(regime) > seq_len:
                labels.append(regime[seq_len:])  
        return np.concatenate(labels) if labels else np.array([], dtype=int)

    y_va_regime = get_regime_labels(val, cfg.seq_len)[:len(log_pred_va)]
    y_te_regime = get_regime_labels(test, cfg.seq_len)[:len(log_pred_te)]

    b = best_offset_target_rate(log_pred_va, y_va_regime)
    logger.info({"calibration_offset_b": float(b)})

    y_pred = (log_pred_te >= b).astype(int)
    metrics = classification_metrics(y_te_regime, y_pred, log_pred_te)
    cm = confusion_matrix(y_te_regime, y_pred)

    logger.info({"feature_cols": FEATURE_COLS, "n_features": len(FEATURE_COLS)})
    logger.info({"prevalence": float(y_te_regime.mean()), **metrics})
    logger.info({"confusion_matrix": cm.tolist()})


if __name__ == "__main__":
    main()