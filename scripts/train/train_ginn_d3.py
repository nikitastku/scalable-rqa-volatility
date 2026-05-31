"""
GINN for Dataset 3 (intraday multi-stock).

Same as train_lstm_d3.py but with GJR-GARCH teacher distillation.
For multi-stock, we fit one GJR-GARCH per stock and concatenate teacher signals.
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
from scalable_rqa_volatility.models.ml_ginn import GINNNet
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
    garch_scale: float = 100.0
    hidden_size: int = 96
    num_layers: int = 2
    dropout: float = 0.1
    eps: float = 1e-12


class NumpyDataset(Dataset):
    def __init__(self, X, y, y_teacher):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.y_teacher = torch.as_tensor(y_teacher, dtype=torch.float32)
    def __len__(self): return int(self.X.shape[0])
    def __getitem__(self, idx): return self.X[idx], self.y[idx], self.y_teacher[idx]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_split(name: str) -> pd.DataFrame:
    return pd.read_parquet(repo_root() / "data" / "processed" / f"dataset3_{name}.parquet")


def as_device(device: str) -> torch.device:
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def fit_garch_per_stock(df: pd.DataFrame, train_frac: float = 0.7, scale: float = 100.0) -> pd.Series:
    """
    Fit GJR-GARCH per stock on training portion, forecast sigma for full series.
    Returns a Series of sigma values aligned with df index.
    """
    from arch import arch_model

    all_sigma = pd.Series(np.nan, index=df.index, dtype=float)

    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker
        df_t = df.loc[mask].reset_index(drop=True)
        returns = df_t["log_return"].astype(float).to_numpy() * scale

        n = len(df_t)
        n_train = int(n * train_frac)

        try:
            am = arch_model(returns, mean="zero", vol="GARCH", p=1, o=1, q=1, dist="t", rescale=False)
            res = am.fit(disp="off", last_obs=n_train)
            fc = res.forecast(horizon=1, start=0, reindex=False)
            var = fc.variance.iloc[:, 0].to_numpy()
            sigma = np.sqrt(var) / scale  
            idx = df.index[mask]
            all_sigma.iloc[idx[:len(sigma)]] = sigma[:len(idx)]
        except Exception:
            pass  

    return all_sigma


def build_supervised_with_teacher_per_stock(
    df: pd.DataFrame, sigma_arr: np.ndarray, feat_cols: list[str],
    seq_len: int, eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sequences with teacher signal per stock. Vectorized version."""
    all_X, all_y, all_teacher = [], [], []
    n_feat = len(feat_cols)

    pos = 0
    for ticker in df["ticker"].unique():
        mask = (df["ticker"] == ticker).to_numpy()
        df_t = df.loc[mask].reset_index(drop=True)
        n_t = len(df_t)
        sig_vals = sigma_arr[pos:pos + n_t].astype(np.float32)
        pos += n_t

        if n_t <= seq_len:
            continue

        feats = df_t[feat_cols].to_numpy(dtype=np.float32)
        rv = df_t["rv"].astype(np.float32).to_numpy()

        shape = (n_t - seq_len, seq_len, n_feat)
        strides = (feats.strides[0], feats.strides[0], feats.strides[1])
        windows = np.lib.stride_tricks.as_strided(feats, shape=shape, strides=strides).copy()

        y_rv = rv[seq_len:]
        y_log = np.log(y_rv + eps)
        teacher_rv = sig_vals[seq_len:]
        teacher_log = np.log(np.maximum(teacher_rv, eps) + eps)

        valid = (np.isfinite(windows).reshape(len(windows), -1).all(axis=1)
                 & np.isfinite(y_log)
                 & np.isfinite(teacher_log))

        if valid.sum() > 0:
            all_X.append(windows[valid])
            all_y.append(y_log[valid])
            all_teacher.append(teacher_log[valid])

    if not all_X:
        return (np.empty((0, seq_len, n_feat), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32))

    return (np.concatenate(all_X, axis=0),
            np.concatenate(all_y, axis=0),
            np.concatenate(all_teacher, axis=0))


def fit_model(model, train_loader, val_loader, device, cfg, lambda_garch):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=cfg.scheduler_factor, patience=cfg.scheduler_patience)

    best_val, best_state, bad = float("inf"), None, 0
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(cfg.epochs):
        model.train()
        for X, y, y_teacher in train_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_teacher = y_teacher.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(X)
                loss = (1 - lambda_garch) * loss_fn(pred, y) + lambda_garch * loss_fn(pred, y_teacher)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_grad_norm)
            scaler.step(opt)
            scaler.update()

        model.eval()
        vals = []
        with torch.no_grad():
            for X, y, y_teacher in val_loader:
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                y_teacher = y_teacher.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model(X)
                    v = ((1 - lambda_garch) * loss_fn(pred, y) + lambda_garch * loss_fn(pred, y_teacher)).item()
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
    return {"best_val_loss": float(best_val), "stopped_epoch": epoch + 1}


def predict(model, loader, device):
    model.eval()
    out = []
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for X, _, _ in loader:
            X = X.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                y = model(X).detach().cpu().numpy()
            out.append(y)
    return np.concatenate(out) if out else np.empty((0,))


def best_offset_target_rate(scores, y_true):
    scores = np.asarray(scores, dtype=float).reshape(-1)
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    if scores.size == 0: return 0.0
    target = float(np.clip(np.mean(y_true), 0.05, 0.95))
    grid = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 199)))
    best_b, best_f1 = 0.0, -1.0
    for b in grid:
        y_pred = (scores - float(b) >= 0.0).astype(int)
        pos = float(np.mean(y_pred))
        if not (target * 0.5 <= pos <= min(0.99, target * 1.5)): continue
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        denom = 2 * tp + fp + fn
        f1 = float((2 * tp) / denom) if denom else 0.0
        if f1 > best_f1: best_f1, best_b = f1, float(b)
    if best_f1 < 0.0: best_b = float(np.quantile(scores, 1.0 - target))
    return float(best_b)


def main() -> None:
    logger = get_logger()
    set_global_seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_garch", type=float, default=0.3)
    parser.add_argument("--max_tickers", type=int, default=100,
                        help="Max tickers for GINN (memory constraint). Default=100")
    args = parser.parse_args()

    cfg = ModelConfig()
    train_cfg = TrainConfig()

    train = load_split("train")
    val = load_split("val")
    test = load_split("test")
    full = pd.concat([train, val, test], ignore_index=True)

    all_tickers = sorted(set(train["ticker"].unique()) & set(val["ticker"].unique()) & set(test["ticker"].unique()))
    if args.max_tickers and len(all_tickers) > args.max_tickers:
        np.random.seed(42)
        selected = sorted(np.random.choice(all_tickers, args.max_tickers, replace=False))
        train = train[train["ticker"].isin(selected)].reset_index(drop=True)
        val = val[val["ticker"].isin(selected)].reset_index(drop=True)
        test = test[test["ticker"].isin(selected)].reset_index(drop=True)
        full = pd.concat([train, val, test], ignore_index=True)
        logger.info(f"Subsampled to {args.max_tickers} tickers for memory")

    logger.info(f"Train: {len(train):,}, Val: {len(val):,}, Test: {len(test):,}")

    logger.info("Fitting GJR-GARCH teacher per stock...")
    sigma = fit_garch_per_stock(full, train_frac=0.7, scale=cfg.garch_scale)
    valid_sigma = sigma.notna().sum()
    logger.info(f"GARCH teacher: {valid_sigma:,}/{len(full):,} valid sigma values")

    logger.info("Building sequences with teacher signal...")
    sigma_np = sigma.to_numpy(dtype=float)
    sigma_tr = sigma_np[:len(train)]
    sigma_va = sigma_np[len(train):len(train)+len(val)]
    sigma_te = sigma_np[len(train)+len(val):]

    Xtr, ytr, ytr_teacher = build_supervised_with_teacher_per_stock(train, sigma_tr, FEATURE_COLS, cfg.seq_len, cfg.eps)
    Xva, yva, yva_teacher = build_supervised_with_teacher_per_stock(val, sigma_va, FEATURE_COLS, cfg.seq_len, cfg.eps)
    Xte, yte, yte_teacher = build_supervised_with_teacher_per_stock(test, sigma_te, FEATURE_COLS, cfg.seq_len, cfg.eps)

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
    ytr_teacher_s = (ytr_teacher - y_mu) / y_sd
    yva_teacher_s = (yva_teacher - y_mu) / y_sd
    yte_teacher_s = (yte_teacher - y_mu) / y_sd

    device = as_device("cuda")
    pin = device.type == "cuda"
    torch.backends.cudnn.benchmark = bool(pin)

    tr_loader = DataLoader(NumpyDataset(Xtr, ytr_s, ytr_teacher_s), batch_size=train_cfg.batch_size, shuffle=True, pin_memory=pin)
    va_loader = DataLoader(NumpyDataset(Xva, yva_s, yva_teacher_s), batch_size=train_cfg.batch_size, shuffle=False, pin_memory=pin)
    te_loader = DataLoader(NumpyDataset(Xte, (yte - y_mu) / y_sd, yte_teacher_s), batch_size=train_cfg.batch_size, shuffle=False, pin_memory=pin)

    model = GINNNet(input_size=len(FEATURE_COLS), hidden_size=cfg.hidden_size,
                    num_layers=cfg.num_layers, dropout=cfg.dropout).to(device)

    fit_info = fit_model(model, tr_loader, va_loader, device, train_cfg, lambda_garch=args.lambda_garch)
    logger.info(fit_info)

    log_pred_va = predict(model, va_loader, device) * y_sd + y_mu
    log_pred_te = predict(model, te_loader, device) * y_sd + y_mu

    y_regime_va, y_regime_te = [], []

    pos = 0
    for ticker in val["ticker"].unique():
        mask = (val["ticker"] == ticker).to_numpy()
        df_t = val.loc[mask].reset_index(drop=True)
        regime = df_t["regime"].astype(float).to_numpy()
        n_t = len(df_t)
        sig_chunk = sigma_va[pos:pos + n_t]
        pos += n_t
        for t in range(cfg.seq_len - 1, n_t - 1):
            if t + 1 < len(sig_chunk) and np.isfinite(sig_chunk[t + 1]):
                y_regime_va.append(int(regime[t + 1]))

    pos = 0
    for ticker in test["ticker"].unique():
        mask = (test["ticker"] == ticker).to_numpy()
        df_t = test.loc[mask].reset_index(drop=True)
        regime = df_t["regime"].astype(float).to_numpy()
        n_t = len(df_t)
        sig_chunk = sigma_te[pos:pos + n_t]
        pos += n_t
        for t in range(cfg.seq_len - 1, n_t - 1):
            if t + 1 < len(sig_chunk) and np.isfinite(sig_chunk[t + 1]):
                y_regime_te.append(int(regime[t + 1]))

    y_va_reg = np.array(y_regime_va[:len(log_pred_va)])
    y_te_reg = np.array(y_regime_te[:len(log_pred_te)])

    b = best_offset_target_rate(log_pred_va, y_va_reg)
    logger.info({"calibration_offset_b": float(b), "lambda_garch": float(args.lambda_garch)})

    y_pred = (log_pred_te >= b).astype(int)
    metrics = classification_metrics(y_te_reg, y_pred, log_pred_te)
    cm = confusion_matrix(y_te_reg, y_pred)

    logger.info({"prevalence": float(y_te_reg.mean()), **metrics})
    logger.info({"confusion_matrix": cm.tolist()})


if __name__ == "__main__":
    main()