from __future__ import annotations

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


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50                # increased from 30 — scheduler needs room
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 10              # increased from 6 — scheduler reduces LR before stopping
    num_workers: int = 0
    # --- NEW: scheduler & clipping ---
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
    thr_lookback: int = 252
    thr_q: float = 0.7
    windows: tuple[int, ...] = (5, 22, 60)


class NumpySeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_split(name: str) -> pd.DataFrame:
    return pd.read_parquet(repo_root() / "data" / "processed" / f"dataset2_{name}.parquet")


def as_device(device: str) -> torch.device:
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_features(df: pd.DataFrame, windows: tuple[int, ...]) -> pd.DataFrame:
    """
    IMPROVED: includes RV and rolling-RV features alongside return features.
    RV autocorrelation is ~0.997 at lag-1, making lagged RV the single most
    informative predictor for next-day RV (as HAR-RV demonstrates).
    """
    r = df["log_return"].astype(float)
    rv = df["rv"].astype(float)
    out = pd.DataFrame(index=df.index)

    # --- return-based features (unchanged) ---
    out["ret"] = r
    out["ret_abs"] = r.abs()
    out["ret_sq"] = r.pow(2)
    for w in windows:
        out[f"ret_mean_{w}"] = r.rolling(w, min_periods=w).mean()
        out[f"ret_std_{w}"] = r.rolling(w, min_periods=w).std(ddof=0)
        out[f"ret_abs_mean_{w}"] = r.abs().rolling(w, min_periods=w).mean()

    # --- NEW: RV-based features ---
    out["rv"] = rv
    for w in windows:
        out[f"rv_mean_{w}"] = rv.rolling(w, min_periods=w).mean()
        out[f"rv_std_{w}"] = rv.rolling(w, min_periods=w).std(ddof=0)

    return out


def build_supervised(
    feats: np.ndarray,
    rv: np.ndarray,
    y_regime: np.ndarray,
    seq_len: int,
    start: int,
    end: int,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    t_list: list[int] = []

    first_t = start + seq_len - 1
    for t in range(first_t, end):
        x = feats[t - seq_len + 1 : t + 1]
        y_next = rv[t + 1]
        reg_next = y_regime[t + 1]
        if np.isfinite(x).all() and np.isfinite(y_next) and np.isfinite(reg_next):
            X_list.append(x)
            y_list.append(float(np.log(float(y_next) + eps)))
            t_list.append(t)

    X = np.stack(X_list, axis=0) if X_list else np.empty((0, seq_len, feats.shape[1]), dtype=float)
    y = np.asarray(y_list, dtype=float)
    t_idx = np.asarray(t_list, dtype=int)
    return X, y, t_idx


def rolling_threshold_from_history(rv: np.ndarray, lookback: int, q: float) -> np.ndarray:
    s = pd.Series(rv)
    return s.rolling(lookback, min_periods=lookback).quantile(q).shift(1).to_numpy(dtype=float)


def fit_model(
    model: torch.nn.Module,
    tr_loader: DataLoader,
    va_loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
) -> dict[str, float]:
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = torch.nn.MSELoss()

    # --- NEW: learning rate scheduler ---
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
    )

    best_val = float("inf")
    best_state = None
    bad = 0

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(cfg.epochs):
        model.train()
        for X, y in tr_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(X)
                loss = loss_fn(pred, y)
            scaler.scale(loss).backward()

            # --- NEW: gradient clipping ---
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_grad_norm)

            scaler.step(opt)
            scaler.update()

        model.eval()
        vals: list[float] = []
        with torch.no_grad():
            for X, y in va_loader:
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model(X)
                    v = loss_fn(pred, y).item()
                vals.append(float(v))

        val_loss = float(np.mean(vals)) if vals else float("inf")

        # --- NEW: step scheduler ---
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"best_val_mse": float(best_val), "stopped_epoch": epoch + 1}


def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for X, _ in loader:
            X = X.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                y = model(X).detach().cpu().numpy()
            out.append(y)
    return np.concatenate(out, axis=0) if out else np.empty((0,), dtype=float)


def best_offset_target_rate(scores: np.ndarray, y_true: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    if scores.size == 0:
        return 0.0

    target = float(np.mean(y_true))
    target = float(np.clip(target, 0.05, 0.95))

    qs = np.linspace(0.01, 0.99, 199)
    grid = np.unique(np.quantile(scores, qs))

    best_b = 0.0
    best_f1 = -1.0

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
            best_f1 = f1
            best_b = float(b)

    if best_f1 < 0.0:
        best_b = float(np.quantile(scores, 1.0 - target))

    return float(best_b)


def build_eval_arrays(
    full_rv: np.ndarray,
    full_regime: np.ndarray,
    split_start: int,
    split_end: int,
    t_idx: np.ndarray,
    log_rv_pred_next: np.ndarray,
    eps: float,
    lookback: int,
    q: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seg_rv = full_rv[split_start : split_end + 1]
    seg_reg = full_regime[split_start : split_end + 1]

    thr = rolling_threshold_from_history(seg_rv, lookback=lookback, q=q)
    thr_next = np.roll(thr, -1)

    idx = (t_idx - split_start).astype(int)
    y_next = np.roll(seg_reg, -1)[idx].astype(int)
    thr_log = np.log(np.maximum(thr_next[idx], eps))
    pred_log = np.asarray(log_rv_pred_next, dtype=float).reshape(-1)

    m = np.isfinite(y_next) & np.isfinite(thr_log) & np.isfinite(pred_log)
    return y_next[m], thr_log[m], pred_log[m]


def main() -> None:
    logger = get_logger()
    set_global_seed(42)
    torch.manual_seed(42)

    cfg = ModelConfig()
    train_cfg = TrainConfig()

    train = load_split("train")
    val = load_split("val")
    test = load_split("test")
    full = pd.concat([train, val, test], axis=0).reset_index(drop=True)

    n_train = len(train)
    n_val = len(val)

    train_start, train_end = 0, n_train - 1
    val_start, val_end = n_train, n_train + n_val - 1
    test_start, test_end = n_train + n_val, len(full) - 1

    # --- CHANGED: build_features instead of return_only_features ---
    feat_df = build_features(full, cfg.windows).replace([np.inf, -np.inf], np.nan).astype(float)
    feat_cols = list(feat_df.columns)

    feats = feat_df.to_numpy(dtype=float, copy=False)
    rv = full["rv"].astype(float).to_numpy()
    regime = full["regime"].astype(float).to_numpy()

    Xtr, ytr, _ = build_supervised(feats, rv, regime, cfg.seq_len, train_start, train_end, cfg.eps)
    Xva, yva, t_va = build_supervised(feats, rv, regime, cfg.seq_len, val_start, val_end, cfg.eps)
    Xte, yte, t_te = build_supervised(feats, rv, regime, cfg.seq_len, test_start, test_end, cfg.eps)

    if len(Xtr) == 0 or len(Xva) == 0 or len(Xte) == 0:
        raise RuntimeError("No sequences produced.")

    mu = Xtr.mean(axis=(0, 1), keepdims=True)
    sd = Xtr.std(axis=(0, 1), keepdims=True)
    sd = np.where(sd == 0.0, 1.0, sd)

    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd
    Xte = (Xte - mu) / sd

    y_mu = float(np.mean(ytr))
    y_sd = float(np.std(ytr))
    if y_sd == 0.0:
        y_sd = 1.0

    ytr_s = (ytr - y_mu) / y_sd
    yva_s = (yva - y_mu) / y_sd
    yte_s = (yte - y_mu) / y_sd

    tr_ds = NumpySeqDataset(Xtr, ytr_s)
    va_ds = NumpySeqDataset(Xva, yva_s)
    te_ds = NumpySeqDataset(Xte, yte_s)

    device = as_device(cfg.device)
    pin = device.type == "cuda"
    torch.backends.cudnn.benchmark = bool(pin)

    tr_loader = DataLoader(tr_ds, batch_size=train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers, pin_memory=pin)
    va_loader = DataLoader(va_ds, batch_size=train_cfg.batch_size, shuffle=False, num_workers=train_cfg.num_workers, pin_memory=pin)
    te_loader = DataLoader(te_ds, batch_size=train_cfg.batch_size, shuffle=False, num_workers=train_cfg.num_workers, pin_memory=pin)

    model = VolatilityLSTM(
        input_size=len(feat_cols),
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

    fit_info = fit_model(model, tr_loader, va_loader, device, train_cfg)
    logger.info(fit_info)

    log_pred_va = predict(model, va_loader, device) * y_sd + y_mu
    log_pred_te = predict(model, te_loader, device) * y_sd + y_mu

    y_va2, thr_log_va, pred_log_va = build_eval_arrays(rv, regime, val_start, val_end, t_va, log_pred_va, cfg.eps, cfg.thr_lookback, cfg.thr_q)
    y_te2, thr_log_te, pred_log_te = build_eval_arrays(rv, regime, test_start, test_end, t_te, log_pred_te, cfg.eps, cfg.thr_lookback, cfg.thr_q)

    score_va = pred_log_va - thr_log_va
    score_te = pred_log_te - thr_log_te

    b = best_offset_target_rate(score_va, y_va2)
    logger.info({"calibration_offset_b": float(b)})

    y_pred_te = (score_te - b >= 0.0).astype(int)

    metrics = classification_metrics(y_te2, y_pred_te, score_te)
    cm = confusion_matrix(y_te2, y_pred_te)

    logger.info({"feature_cols": feat_cols, "n_features": len(feat_cols)})
    logger.info({"prevalence": float(y_te2.mean()), **metrics})
    logger.info({"confusion_matrix": cm.tolist()})


if __name__ == "__main__":
    main()