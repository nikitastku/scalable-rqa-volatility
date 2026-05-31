"""
Non-chronological-split robustness for the headline D3 result.

Addresses question: does the RQA classification gain on D3 survive under a CV protocol
that's more defensible than a single 70/15/15 chronological split?

Protocol: purged walk-forward CV with embargo
  - 5 forecast origins, evenly spaced through the second half of each ticker's
    series. At each origin t_i, train on [0, t_i) and test on (t_i+embargo, t_i+embargo+test_len].
  - Purge: drop training samples whose feature window (≤ 390 bars) extends past t_i.
  - Embargo: 60 bars (one RV window) between train end and test start.
  - Per-ticker walk-forward, then pool across tickers; one ΔAUC per fold.
  - Report mean ± SE across folds, plus a Wilcoxon signed-rank test on per-fold ΔAUC.
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score

from scalable_rqa_volatility.recurrence.embeddings import EmbeddingConfig
from scalable_rqa_volatility.recurrence.rqa import (
    RQAConfig, rqa_features_rolling, estimate_epsilon_from_train,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


RQA_CFG = RQAConfig(
    window=60, step=20, recurrence_rate=0.1,
    embed=EmbeddingConfig(m=4, tau=2), mode="joint",
)
RQA_COLS = ("log_return", "rv")
STD_COLS = [
    "ret_abs", "ret_sq", "rv",
    "ret_mean_30", "ret_std_30", "ret_abs_mean_30", "rv_mean_30", "rv_std_30",
    "ret_mean_120", "ret_std_120", "ret_abs_mean_120", "rv_mean_120", "rv_std_120",
    "ret_mean_390", "ret_std_390", "ret_abs_mean_390", "rv_mean_390", "rv_std_390",
]
MAX_FEATURE_WINDOW = 390  
EMBARGO_BARS = 60          
N_FOLDS = 5
SEED = 42


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_d3_full() -> pd.DataFrame:
    base = repo_root() / "data" / "processed"
    parts = [pd.read_parquet(base / f"dataset3_{s}.parquet")
             for s in ("train", "val", "test")]
    full = pd.concat(parts, ignore_index=True)
    if "timestamp" in full.columns:
        full = full.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    else:
        full = full.sort_values(["ticker"]).reset_index(drop=True)
    return full


def make_fold_indices(n: int, origins: list[float], test_frac: float,
                      embargo: int, purge: int) -> list[dict]:
    """For one ticker of length n, build (train_idx, test_idx) per origin."""
    folds = []
    test_len = int(n * test_frac)
    for f_origin in origins:
        t_i = int(n * f_origin)
        test_start = t_i + embargo
        test_end = test_start + test_len
        if test_end > n or test_start >= n:
            continue
        train_end = t_i - purge
        if train_end < MAX_FEATURE_WINDOW + 100:
            continue
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        folds.append({"origin": f_origin, "train": train_idx, "test": test_idx})
    return folds


def build_xy(df: pd.DataFrame, X_df: pd.DataFrame,
             label_col: str = "regime") -> tuple[np.ndarray, np.ndarray]:
    y_next = df[label_col].shift(-1).to_numpy()
    ok = pd.notna(y_next)
    X = X_df.to_numpy(dtype=float)[ok]
    y = y_next[ok].astype(int)
    keep = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[keep], y[keep]


def best_threshold_target_rate(y_true: np.ndarray, prob: np.ndarray) -> float:
    target = float(np.clip(np.mean(y_true), 0.05, 0.95))
    if prob.size > 5000:
        grid = np.unique(np.quantile(prob, np.linspace(0.01, 0.99, 199)))
    else:
        grid = np.unique(prob)
    if grid.size == 0:
        return 0.5
    best_t, best_f1 = float(np.median(prob)), -1.0
    for t in grid:
        y_pred = (prob >= t).astype(int)
        pos = float(np.mean(y_pred))
        if not (target * 0.5 <= pos <= min(0.99, target * 1.5)):
            continue
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def run_one_fold(full: pd.DataFrame, tickers: list[str], origin: float,
                 test_frac: float, eps: float, fold_id: int) -> dict | None:
    """One walk-forward origin: build pooled X/y, fit both models, return metrics."""
    X_std_tr, y_tr, X_rqa_tr = [], [], []
    X_std_te, y_te, X_rqa_te = [], [], []

    for tk in tickers:
        sub = full[full["ticker"] == tk].reset_index(drop=True)
        if len(sub) < MAX_FEATURE_WINDOW * 2:
            continue
        folds = make_fold_indices(
            n=len(sub), origins=[origin], test_frac=test_frac,
            embargo=EMBARGO_BARS, purge=MAX_FEATURE_WINDOW,
        )
        if not folds:
            continue
        f = folds[0]

        X_std_full = sub[STD_COLS]
        X_rqa_full = rqa_features_rolling(
            sub, RQA_COLS, RQA_CFG, prefix="rqa", eps_fixed=eps,
        )

        sub_tr = sub.iloc[f["train"]].reset_index(drop=True)
        sub_te = sub.iloc[f["test"]].reset_index(drop=True)
        Xs_tr = X_std_full.iloc[f["train"]].reset_index(drop=True)
        Xs_te = X_std_full.iloc[f["test"]].reset_index(drop=True)
        Xr_tr = X_rqa_full.iloc[f["train"]].reset_index(drop=True)
        Xr_te = X_rqa_full.iloc[f["test"]].reset_index(drop=True)

        Xs_tr_arr, ys_tr = build_xy(sub_tr, Xs_tr)
        Xs_te_arr, ys_te = build_xy(sub_te, Xs_te)
        Xr_tr_arr, yr_tr = build_xy(sub_tr, Xr_tr)
        Xr_te_arr, yr_te = build_xy(sub_te, Xr_te)

        ntr = min(len(ys_tr), len(yr_tr))
        nte = min(len(ys_te), len(yr_te))
        if ntr < 50 or nte < 50:
            continue

        X_std_tr.append(Xs_tr_arr[:ntr])
        y_tr.append(ys_tr[:ntr])
        X_rqa_tr.append(Xr_tr_arr[:ntr])

        X_std_te.append(Xs_te_arr[:nte])
        y_te.append(ys_te[:nte])
        X_rqa_te.append(Xr_te_arr[:nte])

    if not X_std_tr:
        return None

    X_std_tr = np.concatenate(X_std_tr)
    y_tr_arr = np.concatenate(y_tr)
    X_rqa_tr = np.concatenate(X_rqa_tr)
    X_std_te = np.concatenate(X_std_te)
    y_te_arr = np.concatenate(y_te)
    X_rqa_te = np.concatenate(X_rqa_te)

    X_comb_tr = np.concatenate([X_std_tr, X_rqa_tr], axis=1)
    X_comb_te = np.concatenate([X_std_te, X_rqa_te], axis=1)

    rf1 = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5, n_jobs=-1,
        class_weight="balanced_subsample", random_state=SEED,
    )
    rf1.fit(X_std_tr, y_tr_arr)
    p_te1 = rf1.predict_proba(X_std_te)[:, 1]
    auc_std = roc_auc_score(y_te_arr, p_te1)

    rf2 = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5, n_jobs=-1,
        class_weight="balanced_subsample", random_state=SEED,
    )
    rf2.fit(X_comb_tr, y_tr_arr)
    p_te2 = rf2.predict_proba(X_comb_te)[:, 1]
    auc_rqa = roc_auc_score(y_te_arr, p_te2)

    return {
        "fold": fold_id,
        "origin": origin,
        "n_train_pooled": int(len(y_tr_arr)),
        "n_test_pooled": int(len(y_te_arr)),
        "auc_std": float(auc_std),
        "auc_rqa": float(auc_rqa),
        "delta_auc": float(auc_rqa - auc_std),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_tickers", type=int, default=50,
                    help="Tickers to sample (default 50; pass 503 for full panel).")
    ap.add_argument("--test_frac", type=float, default=0.10,
                    help="Fraction of each ticker's bars to allocate to each test fold.")
    args = ap.parse_args()

    print("=" * 78)
    print("WALK-FORWARD CV ROBUSTNESS — D3")
    print("Protocol: purged walk-forward (López de Prado 2018, ch. 7)")
    print("=" * 78)

    full = load_d3_full()
    all_tickers = sorted(full["ticker"].unique())
    n_pick = min(args.n_tickers, len(all_tickers))
    rng = np.random.default_rng(SEED)
    chosen = sorted(rng.choice(all_tickers, n_pick, replace=False).tolist())
    full = full[full["ticker"].isin(chosen)].reset_index(drop=True)
    print(f"Using {len(chosen)} of {len(all_tickers)} tickers (seed {SEED})")

    eps_sample_blocks = []
    for tk in chosen[:10]:
        sub = full[full["ticker"] == tk].reset_index(drop=True)
        early = sub.iloc[: int(len(sub) * 0.3)][["log_return", "rv"]].to_numpy(dtype=float)
        if len(early) > 0:
            eps_sample_blocks.append(early)
    eps = estimate_epsilon_from_train(np.concatenate(eps_sample_blocks), RQA_CFG)
    print(f"epsilon estimated from early-period pooled sample: {eps:.6g}")

    origins = np.linspace(0.60, 0.88, N_FOLDS).tolist()

    rows: list[dict] = []
    for k, o in enumerate(origins, start=1):
        t0 = time.time()
        print(f"\nFold {k}/{N_FOLDS}  ·  origin = {o:.2f} of each ticker's series")
        r = run_one_fold(full, chosen, origin=o, test_frac=args.test_frac,
                         eps=eps, fold_id=k)
        if r is None:
            print(f"  fold {k}: not enough samples — skipped")
            continue
        elapsed = time.time() - t0
        print(f"  n_train_pooled={r['n_train_pooled']:,}  "
              f"n_test_pooled={r['n_test_pooled']:,}")
        print(f"  AUC Std    = {r['auc_std']:.4f}")
        print(f"  AUC +RQA   = {r['auc_rqa']:.4f}")
        print(f"  ΔAUC       = {r['delta_auc']:+.5f}   ({elapsed:.0f}s)")
        rows.append(r)

    df = pd.DataFrame(rows)
    deltas = df["delta_auc"].to_numpy()
    mean_d, se_d = float(deltas.mean()), float(deltas.std(ddof=1) / np.sqrt(len(deltas)))

    try:
        stat, p = wilcoxon(deltas, alternative="greater")
        p_str = f"{p:.4g}"
    except Exception:
        stat, p = float("nan"), float("nan")
        p_str = "n/a"

    print("\n" + "=" * 78)
    print("AGGREGATED ACROSS FOLDS")
    print("=" * 78)
    print(f"Mean ΔAUC  = {mean_d:+.5f}   SE = {se_d:.5f}   n_folds = {len(deltas)}")
    print(f"Wilcoxon signed-rank (H1: ΔAUC > 0):  p = {p_str}")
    print()
    print("Reference: headline single-split (chronological) ΔAUC was +0.00057 (p<0.001 per-stock Wilcoxon).")

    out_txt = repo_root() / "results" / "walk_forward_cv_d3.txt"
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as fp:
        fp.write("=" * 78 + "\n")
        fp.write("WALK-FORWARD CV ROBUSTNESS — D3\n")
        fp.write("=" * 78 + "\n\n")
        fp.write(f"Tickers used: {len(chosen)} (seed {SEED})\n")
        fp.write(f"Folds: {len(deltas)} expanding-window origins in [0.60, 0.88]\n")
        fp.write(f"Embargo: {EMBARGO_BARS} bars between train end and test start\n")
        fp.write(f"Purge horizon: {MAX_FEATURE_WINDOW} bars\n")
        fp.write(f"epsilon: {eps:.6g} (estimated from early-period pooled sample)\n\n")

        fp.write(f"{'Fold':>5}  {'Origin':>7}  {'N_train':>10}  {'N_test':>9}  "
                 f"{'AUC Std':>9}  {'AUC +RQA':>10}  {'ΔAUC':>9}\n")
        fp.write("-" * 75 + "\n")
        for r in rows:
            fp.write(f"{r['fold']:>5}  {r['origin']:>7.2f}  "
                     f"{r['n_train_pooled']:>10,}  {r['n_test_pooled']:>9,}  "
                     f"{r['auc_std']:>9.4f}  {r['auc_rqa']:>10.4f}  "
                     f"{r['delta_auc']:>+9.5f}\n")

        fp.write("\n" + "=" * 78 + "\n")
        fp.write("AGGREGATED\n")
        fp.write("=" * 78 + "\n")
        fp.write(f"Mean ΔAUC = {mean_d:+.5f}\n")
        fp.write(f"SE        = {se_d:.5f}\n")
        fp.write(f"n_folds   = {len(deltas)}\n")
        fp.write(f"Wilcoxon (H1: ΔAUC > 0):  p = {p_str}\n\n")

        fp.write("Reference: headline single-chronological-split ΔAUC was +0.00057\n")
        fp.write("           (per-stock Wilcoxon p < 0.001, full 503-ticker panel).\n\n")

        fp.write("INTERPRETATION\n")
        fp.write("-" * 78 + "\n")
        fp.write("If mean ΔAUC > 0 and Wilcoxon p < 0.05: the RQA gain is robust to\n")
        fp.write("split protocol; it reproduces under purged walk-forward CV (López\n")
        fp.write("de Prado 2018), addressing Philippe's concern about reliance on a\n")
        fp.write("single chronological split.\n")

    print(f"\nSaved text report -> {out_txt}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    folds = df["fold"].to_numpy()
    ax1.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax1.axhline(0.00057, color="#1E2761", linewidth=1.4, linestyle="--", alpha=0.7,
                label="headline single-split ΔAUC (+0.00057)")
    colors = ["#10B981" if d > 0 else "#EF4444" for d in deltas]
    ax1.bar(folds, deltas, color=colors, edgecolor="black", linewidth=0.6, alpha=0.85)
    for f, d in zip(folds, deltas):
        ax1.text(f, d, f"{d:+.4f}", ha="center",
                 va="bottom" if d > 0 else "top", fontsize=9)
    ax1.set_xlabel("Walk-forward fold")
    ax1.set_ylabel("ΔAUC  (RQA − Std)")
    ax1.set_title("Per-fold ΔAUC")
    ax1.set_xticks(folds)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(axis="y", alpha=0.25)

    ax2.axis("off")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    sig = "robust" if (mean_d > 0 and (p == p) and p < 0.05) else "noisy"
    ax2.text(0.05, 0.92,
             "Aggregated result", fontsize=14, fontweight="bold", color="#1E2761")
    ax2.text(0.05, 0.80, f"Mean ΔAUC", fontsize=11, color="#6B7280")
    ax2.text(0.05, 0.66, f"{mean_d:+.5f}", fontsize=28, fontweight="bold",
             color=("#10B981" if mean_d > 0 else "#EF4444"))
    ax2.text(0.05, 0.55, f"SE = {se_d:.5f}   ·   n = {len(deltas)} folds",
             fontsize=10, color="#374151")
    ax2.text(0.05, 0.45,
             f"Wilcoxon  (H1: ΔAUC > 0)\np = {p_str}",
             fontsize=10, color="#374151")
    ax2.text(0.05, 0.20,
             "Headline single-split  ΔAUC = +0.00057",
             fontsize=10, color="#6B7280", style="italic")
    ax2.text(0.05, 0.10,
             f"→ result {'survives' if mean_d > 0 else 'does not survive'} "
             f"purged walk-forward CV",
             fontsize=11,
             color=("#10B981" if mean_d > 0 else "#EF4444"),
             fontweight="bold")

    fig.suptitle(
        "Walk-forward CV robustness — D3 (purged + embargoed, López de Prado 2018)",
        fontsize=12, y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_fig = repo_root() / "figures" / "checkpoint_followup" / "fig_f_walk_forward_cv_d3.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure   -> {out_fig}")


if __name__ == "__main__":
    main()