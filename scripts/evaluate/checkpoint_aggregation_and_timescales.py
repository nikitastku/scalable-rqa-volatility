"""
Addresses two outstanding items from the checkpoint feedback:

  (A) D3 aggregation experiment
      Aggregates the 2-min D3 panel to coarser bars (10-min, 30-min, daily-eq.)
      and reruns the same RF Std vs RF Std+RQA comparison the main pipeline does.
      Tests whether the small-but-significant RQA gain on D3 is genuinely
      about intraday dynamics or just about sample size.

  (B) Embedding-timescale analysis
      Plots the autocorrelation function of |log returns| for each dataset on
      a shared lag-axis, with vertical markers showing where the embedding
      span (m-1)*tau falls. Lets you defend / revise the choice of m=4, tau=2.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score

from scalable_rqa_volatility.recurrence.embeddings import EmbeddingConfig
from scalable_rqa_volatility.recurrence.rqa import (
    RQAConfig, rqa_features_rolling, estimate_epsilon_from_train,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

N_TICKERS_FOR_AGG = 50          
SEED = 42

AGG_FACTORS = [1, 5, 15, 60]
AGG_LABELS = {1: "2-min (base)", 5: "10-min", 15: "30-min", 60: "2-hour"}

D3_NATIVE_STD_COLS = [
    "ret_abs", "ret_sq", "rv",
    "ret_mean_30", "ret_std_30", "ret_abs_mean_30", "rv_mean_30", "rv_std_30",
    "ret_mean_120", "ret_std_120", "ret_abs_mean_120", "rv_mean_120", "rv_std_120",
    "ret_mean_390", "ret_std_390", "ret_abs_mean_390", "rv_mean_390", "rv_std_390",
]

def rqa_cfg_for(factor: int) -> RQAConfig:
    base_window = 60
    base_step = 20
    return RQAConfig(
        window=max(20, base_window // max(factor // 5, 1)) if factor > 1 else base_window,
        step=max(5, base_step // max(factor // 5, 1)) if factor > 1 else base_step,
        recurrence_rate=0.1,
        embed=EmbeddingConfig(m=4, tau=2),
        mode="joint",
    )


def aggregate_ticker(df_t: pd.DataFrame, factor: int) -> pd.DataFrame:
    """
    Aggregate a single ticker's 2-min bars to coarser bars by averaging.

    For our purposes:
      - close: take last value within each block (proxy for OHLC close)
      - log_return: sum within each block (log returns are additive)
      - rv: recompute as rolling-std-60 of the new log_return series
      - regime: re-derive from the new rv at 0.7 quantile, lookback=975/factor
    """
    if factor == 1:
        out = df_t.copy()
        out.attrs["std_feature_cols"] = list(D3_NATIVE_STD_COLS)
        return out

    n = len(df_t)
    block = np.arange(n) // factor
    n_full = (n // factor) * factor
    df_full = df_t.iloc[:n_full].copy()
    block = block[:n_full]

    df_full["_block"] = block

    agg_dict: dict[str, tuple[str, str]] = {"log_return": ("log_return", "sum")}
    if "close" in df_full.columns:
        agg_dict["close"] = ("close", "last")
    if "timestamp" in df_full.columns:
        agg_dict["timestamp"] = ("timestamp", "last")
    agg = df_full.groupby("_block").agg(**agg_dict).reset_index(drop=True)
    agg["ticker"] = df_t["ticker"].iloc[0]

    rv_window_eff = max(10, 60 // factor)
    agg["rv"] = (
        agg["log_return"].rolling(rv_window_eff, min_periods=rv_window_eff)
        .std(ddof=0).astype(float)
    )

    lookback_eff = max(50, 975 // factor)
    rv = agg["rv"].astype(float)
    thr = (rv.rolling(lookback_eff, min_periods=lookback_eff)
              .quantile(0.7).shift(1))
    agg["regime"] = (rv >= thr).astype("Int64")

    win_set = sorted({max(5,  30  // factor),
                      max(10, 120 // factor),
                      max(20, 390 // factor)})
    r = agg["log_return"].astype(float)
    rv_s = agg["rv"].astype(float)
    agg["ret_abs"] = r.abs()
    agg["ret_sq"] = r.pow(2)
    for w in win_set:
        agg[f"ret_mean_{w}"] = r.rolling(w, min_periods=w).mean()
        agg[f"ret_std_{w}"] = r.rolling(w, min_periods=w).std(ddof=0)
        agg[f"ret_abs_mean_{w}"] = r.abs().rolling(w, min_periods=w).mean()
        agg[f"rv_mean_{w}"] = rv_s.rolling(w, min_periods=w).mean()
        agg[f"rv_std_{w}"] = rv_s.rolling(w, min_periods=w).std(ddof=0)

    feat_cols = ["ret_abs", "ret_sq"] + [
        f"{p}_{w}" for p in ("ret_mean", "ret_std", "ret_abs_mean", "rv_mean", "rv_std")
        for w in win_set
    ]
    agg = agg.dropna(subset=["log_return", "rv", "regime"] + feat_cols).reset_index(drop=True)
    agg.attrs["std_feature_cols"] = ["ret_abs", "ret_sq", "rv"] + feat_cols[2:]
    return agg


def split_chronologically(df_t: pd.DataFrame, train_frac=0.7, val_frac=0.15):
    n = len(df_t)
    n_tr = int(n * train_frac)
    n_va = int(n * val_frac)
    return (
        df_t.iloc[:n_tr].reset_index(drop=True),
        df_t.iloc[n_tr:n_tr + n_va].reset_index(drop=True),
        df_t.iloc[n_tr + n_va:].reset_index(drop=True),
    )


def build_xy(df: pd.DataFrame, X_df: pd.DataFrame, label_col="regime"):
    y_next = df[label_col].shift(-1).to_numpy()
    ok = pd.notna(y_next)
    X = X_df.to_numpy(dtype=float)[ok]
    y = y_next[ok].astype(int)
    keep = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[keep], y[keep]


def best_threshold_target_rate(y_true, prob):
    target = float(np.clip(np.mean(y_true), 0.05, 0.95))
    grid = np.unique(np.quantile(prob, np.linspace(0.01, 0.99, 199))) \
        if prob.size > 5000 else np.unique(prob)
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


def run_aggregation_experiment(out_txt: Path) -> dict[int, dict[str, float]]:
    """For each aggregation factor, fit RF Std and RF Std+RQA, return AUC/F1."""
    np.random.seed(SEED)
    print("\n" + "=" * 78)
    print("(A) D3 AGGREGATION EXPERIMENT")
    print("=" * 78)

    full = pd.read_parquet(repo_root() / "data" / "processed" / "dataset3_train.parquet")
    val = pd.read_parquet(repo_root() / "data" / "processed" / "dataset3_val.parquet")
    test = pd.read_parquet(repo_root() / "data" / "processed" / "dataset3_test.parquet")
    full = pd.concat([full, val, test], ignore_index=True)

    all_tickers = sorted(full["ticker"].unique())
    chosen = sorted(np.random.choice(all_tickers, N_TICKERS_FOR_AGG, replace=False).tolist())
    print(f"Using {len(chosen)} of {len(all_tickers)} tickers (random seed {SEED})")

    full = full[full["ticker"].isin(chosen)].reset_index(drop=True)

    results = {}

    with open(out_txt, "w", encoding="utf-8") as fp:
        fp.write("=" * 78 + "\n")
        fp.write("D3 AGGREGATION EXPERIMENT — checkpoint follow-up (item 2)\n")
        fp.write("=" * 78 + "\n\n")
        fp.write(f"Random subset: {N_TICKERS_FOR_AGG} tickers (seed {SEED})\n\n")

        fp.write(f"{'Factor':>8}  {'Bar size':>12}  {'Pooled rows':>12}  "
                 f"{'AUC Std':>10}  {'AUC +RQA':>10}  {'ΔAUC':>9}  "
                 f"{'F1 Std':>9}  {'F1 +RQA':>9}\n")
        fp.write("-" * 95 + "\n")

        for factor in AGG_FACTORS:
            t0 = time.time()
            cfg = rqa_cfg_for(factor)

            train_blocks, val_blocks, test_blocks = [], [], []
            std_cols: list[str] | None = None
            for ticker in chosen:
                sub = full[full["ticker"] == ticker]
                if "timestamp" in sub.columns:
                    sub = sub.sort_values("timestamp")
                df_t = sub.reset_index(drop=True)
                df_a = aggregate_ticker(df_t, factor)
                if len(df_a) < cfg.window * 4:
                    continue
                if std_cols is None:
                    candidate = df_a.attrs.get("std_feature_cols") or list(D3_NATIVE_STD_COLS)
                    std_cols = [c for c in candidate if c in df_a.columns]
                tr, va, te = split_chronologically(df_a)
                if len(tr) < cfg.window * 2 or len(va) < cfg.window or len(te) < cfg.window:
                    continue
                train_blocks.append(tr)
                val_blocks.append(va)
                test_blocks.append(te)

            if not train_blocks or not std_cols:
                fp.write(f"  factor={factor}: insufficient data after aggregation\n")
                print(f"  factor={factor}: insufficient data after aggregation")
                continue

            train = pd.concat(train_blocks, ignore_index=True)
            val_df = pd.concat(val_blocks, ignore_index=True)
            test_df = pd.concat(test_blocks, ignore_index=True)

            std_cols = [c for c in std_cols if c in train.columns and c in val_df.columns and c in test_df.columns]

            X_std_tr, y_tr = build_xy(train, train[std_cols])
            X_std_va, y_va = build_xy(val_df, val_df[std_cols])
            X_std_te, y_te = build_xy(test_df, test_df[std_cols])

            rf = RandomForestClassifier(
                n_estimators=200, min_samples_leaf=5, n_jobs=-1,
                class_weight="balanced_subsample", random_state=SEED,
            )
            rf.fit(X_std_tr, y_tr)
            p_va = rf.predict_proba(X_std_va)[:, 1]
            p_te = rf.predict_proba(X_std_te)[:, 1]
            thr = best_threshold_target_rate(y_va, p_va)
            auc_std = roc_auc_score(y_te, p_te)
            f1_std = f1_score(y_te, (p_te >= thr).astype(int), zero_division=0)

            sample_blocks = train_blocks[:5]
            sample = np.concatenate(
                [b[["log_return", "rv"]].to_numpy(dtype=float) for b in sample_blocks],
                axis=0,
            )
            eps = estimate_epsilon_from_train(sample, cfg)

            def rqa_block(df_b):
                feats_list = []
                for tk in df_b["ticker"].unique():
                    sub = df_b[df_b["ticker"] == tk].reset_index(drop=True)
                    f = rqa_features_rolling(sub, ("log_return", "rv"), cfg,
                                             prefix="rqa", eps_fixed=eps)
                    f.index = df_b.index[df_b["ticker"] == tk]
                    feats_list.append(f)
                return pd.concat(feats_list).sort_index()

            X_rqa_tr_df = rqa_block(train)
            X_rqa_va_df = rqa_block(val_df)
            X_rqa_te_df = rqa_block(test_df)

            X_comb_tr_df = pd.concat([train[std_cols].reset_index(drop=True),
                                      X_rqa_tr_df.reset_index(drop=True)], axis=1)
            X_comb_va_df = pd.concat([val_df[std_cols].reset_index(drop=True),
                                      X_rqa_va_df.reset_index(drop=True)], axis=1)
            X_comb_te_df = pd.concat([test_df[std_cols].reset_index(drop=True),
                                      X_rqa_te_df.reset_index(drop=True)], axis=1)

            X_comb_tr, y_tr2 = build_xy(train, X_comb_tr_df)
            X_comb_va, y_va2 = build_xy(val_df, X_comb_va_df)
            X_comb_te, y_te2 = build_xy(test_df, X_comb_te_df)

            rf2 = RandomForestClassifier(
                n_estimators=200, min_samples_leaf=5, n_jobs=-1,
                class_weight="balanced_subsample", random_state=SEED,
            )
            rf2.fit(X_comb_tr, y_tr2)
            p_va2 = rf2.predict_proba(X_comb_va)[:, 1]
            p_te2 = rf2.predict_proba(X_comb_te)[:, 1]
            thr2 = best_threshold_target_rate(y_va2, p_va2)
            auc_rqa = roc_auc_score(y_te2, p_te2)
            f1_rqa = f1_score(y_te2, (p_te2 >= thr2).astype(int), zero_division=0)

            elapsed = time.time() - t0
            line = (f"{factor:>8}  {AGG_LABELS[factor]:>12}  {len(train):>12,}  "
                    f"{auc_std:>10.4f}  {auc_rqa:>10.4f}  {auc_rqa - auc_std:>+9.5f}  "
                    f"{f1_std:>9.4f}  {f1_rqa:>9.4f}")
            print(line + f"  ({elapsed:.0f}s)")
            fp.write(line + "\n")

            results[factor] = dict(
                auc_std=auc_std, auc_rqa=auc_rqa,
                f1_std=f1_std, f1_rqa=f1_rqa,
                pooled_rows=len(train),
            )

        fp.write("\n" + "=" * 78 + "\n")
        fp.write("INTERPRETATION\n")
        fp.write("=" * 78 + "\n")
        fp.write("- If ΔAUC stays positive at all aggregation levels: the RQA gain is\n")
        fp.write("  genuinely about dynamics and survives bar-size changes.\n")
        fp.write("- If ΔAUC trends to zero as bar size grows (sample size shrinks):\n")
        fp.write("  the gain on the full 2-min panel was largely a power effect.\n")

    print(f"\nSaved text report -> {out_txt}")
    return results


def plot_aggregation_results(results: dict, out_path: Path) -> None:
    factors = sorted(results.keys())
    auc_std = [results[f]["auc_std"] for f in factors]
    auc_rqa = [results[f]["auc_rqa"] for f in factors]
    delta = [r - s for r, s in zip(auc_rqa, auc_std)]
    rows = [results[f]["pooled_rows"] for f in factors]
    labels = [AGG_LABELS[f] for f in factors]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    x = np.arange(len(factors))
    width = 0.35
    ax1.bar(x - width / 2, auc_std, width, label="RF Std", color="#3B82F6")
    ax1.bar(x + width / 2, auc_rqa, width, label="RF Std + RQA", color="#1E2761")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Test ROC-AUC")
    ax1.set_title("D3 aggregation — AUC by bar size")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    auc_min = min(auc_std + auc_rqa) - 0.005
    ax1.set_ylim(max(0.5, auc_min), 1.0)
    for xi, (s, r) in enumerate(zip(auc_std, auc_rqa)):
        ax1.text(xi - width / 2, s, f"{s:.3f}", ha="center", va="bottom", fontsize=8)
        ax1.text(xi + width / 2, r, f"{r:.3f}", ha="center", va="bottom", fontsize=8)

    color = ["#10B981" if d > 0 else "#EF4444" for d in delta]
    ax2.bar(x, delta, color=color, edgecolor="black", linewidth=0.8)
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("ΔAUC  (RQA − Std)")
    ax2.set_title("Effect of RQA across aggregation levels")
    ax2.grid(axis="y", alpha=0.3)
    for xi, (d, n) in enumerate(zip(delta, rows)):
        ax2.text(xi, d, f"{d:+.4f}\n(n={n:,})", ha="center",
                 va="bottom" if d > 0 else "top", fontsize=8)

    fig.suptitle("D3 aggregation experiment — does RQA gain survive coarsening?",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {out_path}")


def acf_abs_returns(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = np.abs(x - x.mean())
    n = x.size
    var = (x * x).sum()
    if var == 0:
        return np.zeros(max_lag + 1)
    out = np.empty(max_lag + 1)
    for k in range(max_lag + 1):
        if k == 0:
            out[k] = 1.0
        else:
            out[k] = float((x[:-k] * x[k:]).sum() / var)
    return out


def plot_acf_timescales(out_path: Path) -> None:
    print("\n" + "=" * 78)
    print("(B) EMBEDDING-TIMESCALE ANALYSIS")
    print("=" * 78)

    paths_by_ds = {
        "D1 (synthetic, daily)":
            repo_root() / "data" / "processed" / "dataset1_train.parquet",
        "D2 (S&P 500, daily)":
            repo_root() / "data" / "processed" / "dataset2_train.parquet",
        "D3 (intraday, 2-min)":
            repo_root() / "data" / "processed" / "dataset3_train.parquet",
    }

    max_lag = 60
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = {"D1 (synthetic, daily)": "#1E2761",
            "D2 (S&P 500, daily)": "#3B82F6",
            "D3 (intraday, 2-min)": "#EF4444"}

    for label, path in paths_by_ds.items():
        df = pd.read_parquet(path)
        if "ticker" in df.columns:
            tk = df["ticker"].iloc[0]
            x = df.loc[df["ticker"] == tk, "log_return"].astype(float).to_numpy()
        else:
            x = df["log_return"].astype(float).to_numpy()
        ac = acf_abs_returns(x, max_lag=max_lag)
        ax.plot(np.arange(max_lag + 1), ac, "-", color=cmap[label],
                label=label, linewidth=1.8)
        def _first_lag_below(ac: np.ndarray, thr: float) -> str:
            below = ac < thr
            if not below.any():
                return f">{len(ac) - 1}"
            return str(int(np.argmax(below)))

        tau_e = _first_lag_below(ac, 1.0 / np.e)
        tau_05 = _first_lag_below(ac, 0.05)
        print(f"  {label:30s}  lag where ACF<1/e: {tau_e:>6}   lag where ACF<0.05: {tau_05:>6}")

    ax.axvline(6, color="black", linestyle="--", linewidth=1.0, alpha=0.7,
               label="(m−1)·τ = 6  (current embedding span)")
    ax.axhline(1.0 / np.e, color="grey", linestyle=":", linewidth=0.8)
    ax.text(max_lag, 1.0 / np.e, " 1/e", color="grey", va="center", fontsize=8)
    ax.set_xlabel("lag  (in bars at each dataset's native resolution)")
    ax.set_ylabel("ACF of |log returns|")
    ax.set_title("Decorrelation timescale of |log returns|, by dataset\n"
                 "Same lag axis = same number of native bars (different physical time)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, max_lag)
    ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {out_path}")


def main() -> None:
    out_dir = repo_root() / "figures" / "checkpoint_followup"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_dir = repo_root() / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    agg_txt = results_dir / "checkpoint_aggregation_d3.txt"
    results = run_aggregation_experiment(agg_txt)
    if results:
        plot_aggregation_results(results, out_dir / "fig_d_d3_aggregation_auc.png")

    plot_acf_timescales(out_dir / "fig_e_acf_timescales.png")

    print("\nDone.")


if __name__ == "__main__":
    main()