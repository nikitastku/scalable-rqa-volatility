from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, f1_score

from scalable_rqa_volatility.logging_utils import get_logger
from scalable_rqa_volatility.recurrence.beta_rqa import BetaRQAConfig, beta_rqa_features_rolling
from scalable_rqa_volatility.recurrence.embeddings import EmbeddingConfig

STD_FEATURE_COLS = [
    "ret_abs", "ret_sq", "rv",
    "ret_mean_30", "ret_std_30", "ret_abs_mean_30", "rv_mean_30", "rv_std_30",
    "ret_mean_120", "ret_std_120", "ret_abs_mean_120", "rv_mean_120", "rv_std_120",
    "ret_mean_390", "ret_std_390", "ret_abs_mean_390", "rv_mean_390", "rv_std_390",
]

BETA_FEATURE_NAMES = ["rr", "det", "lam", "lmax", "tt", "entr",
                       "lam_h", "tt_h", "delta_lam", "delta_tt"]

HORIZONTAL_SUFFIXES = ["lam_h", "tt_h", "delta_lam", "delta_tt"]
STANDARD_RQA_SUFFIXES = ["rr", "det", "lam", "lmax", "tt", "entr"]

COLORS = {
    "primary": "#1E2761", "accent": "#3B82F6", "green": "#10B981",
    "red": "#EF4444", "orange": "#F59E0B", "purple": "#8B5CF6",
    "gray": "#94A3B8",
}
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "legend.fontsize": 9, "figure.figsize": (12, 7),
})


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_split(name: str) -> pd.DataFrame:
    return pd.read_parquet(repo_root() / "data" / "processed" / f"dataset3_{name}.parquet")


def savefig(fig, name: str):
    out_dir = repo_root() / "figures"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def compute_beta_rqa_per_stock(df, beta_cfg, cols):
    all_feats = []
    for ticker in sorted(df["ticker"].unique()):
        mask = df["ticker"] == ticker
        df_ticker = df.loc[mask].reset_index(drop=True)
        feats = beta_rqa_features_rolling(df_ticker, cols, beta_cfg, prefix="beta_rqa")
        feats.index = df.index[mask]
        all_feats.append(feats)
    return pd.concat(all_feats).sort_index()


def build_xy_per_stock(df, X_df):
    all_X, all_y = [], []
    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker
        df_t = df.loc[mask].reset_index(drop=True)
        X_t = X_df.loc[mask].reset_index(drop=True)
        y_next = df_t["regime"].shift(-1).to_numpy()
        ok = pd.notna(y_next)
        X_arr = X_t.to_numpy(dtype=float)[ok]
        y_arr = y_next[ok].astype(int)
        keep = np.isfinite(X_arr).all(axis=1) & np.isfinite(y_arr)
        if keep.sum() > 0:
            all_X.append(X_arr[keep])
            all_y.append(y_arr[keep])
    return np.concatenate(all_X), np.concatenate(all_y)


def f1_fast(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    denom = 2 * tp + fp + fn
    return float((2 * tp) / denom) if denom else 0.0


def best_threshold(y_true, prob):
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    prob = np.asarray(prob, dtype=float).reshape(-1)
    if y_true.size == 0:
        return 0.5
    target = float(np.clip(np.mean(y_true), 0.05, 0.95))
    if prob.size > 100000:
        grid = np.unique(np.quantile(prob, np.linspace(0.01, 0.99, 499)))
    else:
        grid = np.unique(prob)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        y_pred = (prob >= float(t)).astype(int)
        pos = float(np.mean(y_pred))
        if not (target * 0.5 <= pos <= min(0.99, target * 1.5)):
            continue
        f1 = f1_fast(y_true, y_pred)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    if best_f1 < 0:
        best_t = float(np.quantile(prob, 1.0 - target))
    return best_t


def analyse_beta(
    beta: float,
    train, val, test, full, n_tr, n_va,
    X_std_tr, y_tr, X_std_va, y_va, X_std_te, y_te,
    logger,
) -> dict:
    """Run full analysis for one β value."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  β = {beta}")
    logger.info(f"{'='*60}")

    beta_cfg = BetaRQAConfig(
        window=60, step=20, recurrence_rate=0.1,
        embed=EmbeddingConfig(m=4, tau=2),
        beta=beta, transform="minmax",
    )
    rqa_cols = ("log_return",)

    logger.info(f"  Computing β-RQA features (β={beta})...")
    t0 = time.time()
    X_beta_all = compute_beta_rqa_per_stock(full, beta_cfg, rqa_cols)
    comp_time = time.time() - t0
    logger.info(f"  β-RQA computation: {comp_time:.1f}s, {X_beta_all.shape[1]} features")

    beta_col_names = list(X_beta_all.columns)

    X_beta_tr_df = X_beta_all.iloc[:n_tr].reset_index(drop=True)
    X_beta_va_df = X_beta_all.iloc[n_tr:n_tr + n_va].reset_index(drop=True)
    X_beta_te_df = X_beta_all.iloc[n_tr + n_va:].reset_index(drop=True)

    X_beta_tr, y_tr2 = build_xy_per_stock(train, X_beta_tr_df)
    X_beta_va, y_va2 = build_xy_per_stock(val, X_beta_va_df)
    X_beta_te, y_te2 = build_xy_per_stock(test, X_beta_te_df)

    ntr = min(len(y_tr), len(y_tr2))
    nva = min(len(y_va), len(y_va2))
    nte = min(len(y_te), len(y_te2))

    X_comb_tr = np.concatenate([X_std_tr[:ntr], X_beta_tr[:ntr]], axis=1)
    X_comb_va = np.concatenate([X_std_va[:nva], X_beta_va[:nva]], axis=1)
    X_comb_te = np.concatenate([X_std_te[:nte], X_beta_te[:nte]], axis=1)
    y_tr_a, y_va_a, y_te_a = y_tr[:ntr], y_va[:nva], y_te[:nte]

    comb_names = STD_FEATURE_COLS + beta_col_names

    logger.info(f"  Training RF Std+β-RQA ({len(comb_names)} features)...")
    rf = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5, n_jobs=-1,
        class_weight="balanced_subsample", random_state=42,
    )
    rf.fit(X_comb_tr, y_tr_a)

    p_va = rf.predict_proba(X_comb_va)[:, 1]
    thr = best_threshold(y_va_a, p_va)
    p_te = rf.predict_proba(X_comb_te)[:, 1]
    y_pred = (p_te >= thr).astype(int)
    auc_full = roc_auc_score(y_te_a, p_te)
    f1_full = f1_score(y_te_a, y_pred, zero_division=0)
    logger.info(f"  RF Std+β-RQA (β={beta}): AUC={auc_full:.6f}, F1={f1_full:.4f}")

    logger.info(f"  Computing permutation importances...")
    t0 = time.time()
    perm = permutation_importance(
        rf, X_comb_te, y_te_a, n_repeats=10, random_state=42,
        scoring="roc_auc", n_jobs=-1,
    )
    perm_time = time.time() - t0
    logger.info(f"  Permutation importance: {perm_time:.1f}s")

    perm_dict = {}
    perm_std_dict = {}
    for i, name in enumerate(comb_names):
        perm_dict[name] = perm.importances_mean[i]
        perm_std_dict[name] = perm.importances_std[i]

    beta_perm = {name: perm_dict[name] for name in beta_col_names}
    beta_perm_std = {name: perm_std_dict[name] for name in beta_col_names}

    horiz_features = [n for n in beta_col_names if any(n.endswith(s) for s in HORIZONTAL_SUFFIXES)]
    std_rqa_features = [n for n in beta_col_names if any(n.endswith(s) for s in STANDARD_RQA_SUFFIXES)]

    horiz_perm_sum = sum(perm_dict[n] for n in horiz_features)
    std_rqa_perm_sum = sum(perm_dict[n] for n in std_rqa_features)

    logger.info(f"  Horizontal measures total perm importance: {horiz_perm_sum:.6f}")
    logger.info(f"  Standard RQA measures total perm importance: {std_rqa_perm_sum:.6f}")

    horiz_indices = [comb_names.index(n) for n in horiz_features]
    keep_cols = [i for i in range(X_comb_tr.shape[1]) if i not in horiz_indices]

    rf_no_horiz = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5, n_jobs=-1,
        class_weight="balanced_subsample", random_state=42,
    )
    rf_no_horiz.fit(X_comb_tr[:, keep_cols], y_tr_a)
    p_va_nh = rf_no_horiz.predict_proba(X_comb_va[:, keep_cols])[:, 1]
    thr_nh = best_threshold(y_va_a, p_va_nh)
    p_te_nh = rf_no_horiz.predict_proba(X_comb_te[:, keep_cols])[:, 1]
    auc_no_horiz = roc_auc_score(y_te_a, p_te_nh)
    delta_horiz = auc_no_horiz - auc_full

    logger.info(f"  Ablation (drop 4 horiz): AUC={auc_no_horiz:.6f} (Δ={delta_horiz:+.6f})")

    std_only_cols = list(range(len(STD_FEATURE_COLS)))
    rf_std = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5, n_jobs=-1,
        class_weight="balanced_subsample", random_state=42,
    )
    rf_std.fit(X_comb_tr[:, std_only_cols], y_tr_a)
    p_va_std = rf_std.predict_proba(X_comb_va[:, std_only_cols])[:, 1]
    thr_std = best_threshold(y_va_a, p_va_std)
    p_te_std = rf_std.predict_proba(X_comb_te[:, std_only_cols])[:, 1]
    auc_std_only = roc_auc_score(y_te_a, p_te_std)
    delta_all = auc_std_only - auc_full

    logger.info(f"  Ablation (drop all 10 β-RQA): AUC={auc_std_only:.6f} (Δ={delta_all:+.6f})")

    lam_col = [n for n in beta_col_names if n.endswith("_lam") and not n.endswith("delta_lam")]
    lam_h_col = [n for n in beta_col_names if n.endswith("_lam_h")]
    if lam_col and lam_h_col:
        lam_vals = X_beta_te_df[lam_col[0]].dropna()
        lam_h_vals = X_beta_te_df[lam_h_col[0]].dropna()
        n_common = min(len(lam_vals), len(lam_h_vals))
        if n_common > 100:
            lam_lam_h_corr = lam_vals.iloc[:n_common].corr(lam_h_vals.iloc[:n_common])
            lam_lam_h_mean_diff = float((lam_vals.iloc[:n_common] - lam_h_vals.iloc[:n_common]).abs().mean())
        else:
            lam_lam_h_corr = float("nan")
            lam_lam_h_mean_diff = float("nan")
    else:
        lam_lam_h_corr = float("nan")
        lam_lam_h_mean_diff = float("nan")

    logger.info(f"  LAM vs LAM_h correlation: {lam_lam_h_corr:.4f}")
    logger.info(f"  LAM vs LAM_h mean |diff|: {lam_lam_h_mean_diff:.6f}")

    return {
        "beta": beta,
        "auc_full": auc_full,
        "f1_full": f1_full,
        "auc_no_horiz": auc_no_horiz,
        "delta_horiz": delta_horiz,
        "auc_std_only": auc_std_only,
        "delta_all": delta_all,
        "beta_perm": beta_perm,
        "beta_perm_std": beta_perm_std,
        "horiz_perm_sum": horiz_perm_sum,
        "std_rqa_perm_sum": std_rqa_perm_sum,
        "horiz_features": horiz_features,
        "std_rqa_features": std_rqa_features,
        "beta_col_names": beta_col_names,
        "lam_lam_h_corr": lam_lam_h_corr,
        "lam_lam_h_mean_diff": lam_lam_h_mean_diff,
        "comp_time": comp_time,
    }


def plot_beta_sweep_horizontal(results_list, fig_name):
    """Main figure: horizontal measure importance across β values."""
    betas = [r["beta"] for r in results_list]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    horiz_sums = [r["horiz_perm_sum"] for r in results_list]
    std_sums = [r["std_rqa_perm_sum"] for r in results_list]
    x = np.arange(len(betas))
    w = 0.35
    ax.bar(x - w/2, std_sums, w, color=COLORS["accent"], label="Standard 6 (RR,DET,...)")
    ax.bar(x + w/2, horiz_sums, w, color=COLORS["purple"], label="Horizontal 4 (LAM_h,...)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"β={b}" for b in betas])
    ax.set_ylabel("Total Permutation Importance (ΔAUC)")
    ax.set_title("A. Horizontal vs Standard RQA Contribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)

    ax = axes[0, 1]
    deltas = [r["delta_horiz"] for r in results_list]
    colors_bar = [COLORS["green"] if d > 0 else COLORS["red"] for d in deltas]
    ax.bar(x, deltas, 0.5, color=colors_bar)
    ax.set_xticks(x)
    ax.set_xticklabels([f"β={b}" for b in betas])
    ax.set_ylabel("ΔAUC when horizontal measures removed")
    ax.set_title("B. Ablation: Drop 4 Horizontal Measures")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)
    for i, d in enumerate(deltas):
        ax.text(i, d + (0.00002 if d >= 0 else -0.00005), f"{d:+.5f}",
                ha="center", fontsize=9, color=COLORS["primary"])

    ax = axes[1, 0]
    corrs = [r["lam_lam_h_corr"] for r in results_list]
    ax.bar(x, corrs, 0.5, color=COLORS["orange"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"β={b}" for b in betas])
    ax.set_ylabel("Correlation(LAM, LAM_h)")
    ax.set_title("C. RP Asymmetry: LAM vs LAM_h Correlation")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color=COLORS["red"], linestyle="--", alpha=0.5, label="Perfect symmetry")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, c in enumerate(corrs):
        ax.text(i, c + 0.02, f"{c:.3f}", ha="center", fontsize=9)

    ax = axes[1, 1]
    horiz_names_short = ["LAM_h", "TT_h", "ΔLAM", "ΔTT"]
    horiz_data = np.zeros((len(horiz_names_short), len(betas)))
    for j, r in enumerate(results_list):
        for i, suffix in enumerate(HORIZONTAL_SUFFIXES):
            matching = [n for n in r["horiz_features"] if n.endswith(suffix)]
            if matching:
                horiz_data[i, j] = r["beta_perm"][matching[0]]

    im = ax.imshow(horiz_data, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(betas)))
    ax.set_xticklabels([f"β={b}" for b in betas])
    ax.set_yticks(range(len(horiz_names_short)))
    ax.set_yticklabels(horiz_names_short)
    ax.set_title("D. Horizontal Feature Importance by β")
    fig.colorbar(im, ax=ax, label="Perm ΔAUC", shrink=0.8)

    for i in range(len(horiz_names_short)):
        for j in range(len(betas)):
            val = horiz_data[i, j]
            color = "white" if abs(val) > 0.0001 else "black"
            ax.text(j, i, f"{val:.5f}", ha="center", va="center", fontsize=8, color=color)

    fig.suptitle("β-RQA: Do Horizontal Measures Help When RP Is Asymmetric?",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    savefig(fig, fig_name)


def main():
    logger = get_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--betas", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0],
                        help="β values to test (default: 0.5 1.0 1.5 2.0)")
    args = parser.parse_args()

    betas = sorted(args.betas)

    print("=" * 70)
    print("  β-RQA HORIZONTAL MEASURE ANALYSIS — BETA SWEEP")
    print(f"  Testing β = {betas}")
    print("  Question: Do horizontal measures help when RP is asymmetric (β≠2)?")
    print("=" * 70)

    logger.info("Loading D3 splits...")
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")
    full = pd.concat([train, val, test], ignore_index=True)
    n_tr, n_va = len(train), len(val)
    logger.info(f"Train: {n_tr:,}, Val: {n_va:,}, Test: {len(test):,}")

    logger.info("Building standard feature arrays...")
    X_std_tr, y_tr = build_xy_per_stock(train, train[STD_FEATURE_COLS].copy())
    X_std_va, y_va = build_xy_per_stock(val, val[STD_FEATURE_COLS].copy())
    X_std_te, y_te = build_xy_per_stock(test, test[STD_FEATURE_COLS].copy())

    results_list = []
    for beta in betas:
        result = analyse_beta(
            beta, train, val, test, full, n_tr, n_va,
            X_std_tr, y_tr, X_std_va, y_va, X_std_te, y_te,
            logger,
        )
        results_list.append(result)

    print("\n--- Generating figures ---")
    plot_beta_sweep_horizontal(results_list, "6_4_beta_sweep_horizontal_importance")

    out_dir = repo_root() / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "feature_importance_beta_sweep_d3.txt"

    lines = []
    lines.append("=" * 80)
    lines.append("  β-RQA HORIZONTAL MEASURE ANALYSIS — BETA SWEEP ON DATASET 3")
    lines.append("  Question: Do horizontal measures help when RP is asymmetric (β≠2)?")
    lines.append("=" * 80)

    lines.append(f"\n\nA. SUMMARY ACROSS β VALUES\n")
    lines.append(f"  {'β':>5} {'AUC (full)':>12} {'AUC (no horiz)':>16} {'Δ (drop horiz)':>16} "
                 f"{'Horiz perm Σ':>14} {'Std RQA perm Σ':>16} {'LAM↔LAM_h corr':>16}")
    lines.append("  " + "-" * 102)

    for r in results_list:
        lines.append(
            f"  {r['beta']:>5.1f} {r['auc_full']:>12.6f} {r['auc_no_horiz']:>16.6f} "
            f"{r['delta_horiz']:>+16.6f} {r['horiz_perm_sum']:>14.6f} "
            f"{r['std_rqa_perm_sum']:>16.6f} {r['lam_lam_h_corr']:>16.4f}"
        )

    for r in results_list:
        lines.append(f"\n\n{'='*60}")
        lines.append(f"  β = {r['beta']}")
        lines.append(f"{'='*60}")
        lines.append(f"  AUC (Std+β-RQA):          {r['auc_full']:.6f}")
        lines.append(f"  AUC (drop 4 horiz):        {r['auc_no_horiz']:.6f}  (Δ = {r['delta_horiz']:+.6f})")
        lines.append(f"  AUC (std-only):            {r['auc_std_only']:.6f}  (Δ = {r['delta_all']:+.6f})")
        lines.append(f"  LAM vs LAM_h correlation:  {r['lam_lam_h_corr']:.4f}")
        lines.append(f"  LAM vs LAM_h mean |diff|:  {r['lam_lam_h_mean_diff']:.6f}")

        lines.append(f"\n  Permutation importance (β-RQA features only):")
        lines.append(f"  {'Feature':<40} {'Perm ΔAUC':>12} {'Std':>10} {'Type':>12}")
        lines.append("  " + "-" * 76)
        sorted_feats = sorted(r["beta_perm"].items(), key=lambda x: x[1], reverse=True)
        for name, val in sorted_feats:
            ftype = "HORIZONTAL" if any(name.endswith(s) for s in HORIZONTAL_SUFFIXES) else "standard"
            std = r["beta_perm_std"][name]
            lines.append(f"  {name:<40} {val:>12.6f} {std:>10.6f} {ftype:>12}")

    lines.append(f"\n\n{'='*80}")
    lines.append(f"  INTERPRETATION")
    lines.append(f"{'='*80}\n")

    any_horiz_helps = any(r["delta_horiz"] < -0.0001 for r in results_list)
    any_asymmetric = any(r["lam_lam_h_corr"] < 0.95 for r in results_list)

    if any_asymmetric:
        lines.append("  The RP becomes measurably asymmetric at β ≠ 2:")
        for r in results_list:
            if r["lam_lam_h_corr"] < 0.99:
                lines.append(f"    β={r['beta']}: LAM↔LAM_h correlation = {r['lam_lam_h_corr']:.4f}")
    else:
        lines.append("  The RP remains approximately symmetric across all tested β values.")

    lines.append("")
    if any_horiz_helps:
        lines.append("  Horizontal measures DO contribute at some β values:")
        for r in results_list:
            if r["delta_horiz"] < -0.0001:
                lines.append(f"    β={r['beta']}: dropping horizontal costs {r['delta_horiz']:+.6f} AUC")
    else:
        lines.append("  Horizontal measures do NOT contribute meaningfully at any tested β.")
        lines.append("  Dropping them either has no effect or slightly improves the model.")

    results_text = "\n".join(lines)
    out_path.write_text(results_text, encoding="utf-8")
    logger.info(f"\nResults saved to {out_path}")
    print(results_text)

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()