"""
plot_rp_with_regimes.py — Recurrence-plot visualizations.

Produces three figures, all from D3 (the dataset where the result actually lives):

  fig_a_rp_calm_vs_volatile_d3.png
      Two RPs from the same ticker, one calm window, one volatile window.
      Each panel shows: time-series strip on top, the RP itself, and a table of
      all six RQA measures computed with the project's own rqa.py.

  fig_b_rp_with_regime_axes_d3.png
      A single long ticker segment with RP-on-axes annotation: the x and y
      borders of the RP are coloured red where regime[t] == 1 (high vol)
      and blue where regime[t] == 0 (low vol). Reader can see directly that
      dense recurrence blocks correspond to one regime class.

  fig_c_rqa_distribution_by_regime_d3.png
      Distribution (boxplots/violins) of each RQA measure within
      low- vs high-vol regimes, pooled across many windows. Makes the
      "RQA values differ between regimes" claim quantitative.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from scalable_rqa_volatility.recurrence.embeddings import EmbeddingConfig
from scalable_rqa_volatility.recurrence.rqa import (
    RQAConfig,
    _delay_embed_joint,
    _epsilon_for_rr,
    _pairwise_distances,
    _recurrence_matrix,
    _rqa_from_embedded,
    _standardize_cols,
    estimate_epsilon_from_train,
)


COLORS = {
    "primary": "#1E2761",
    "calm":    "#3B82F6",   
    "vol":     "#EF4444",  
    "rp_bg":   "#FFFFFF",
    "rp_pt":   "#1E2761",
}

WINDOW = 60         
M = 4
TAU = 2
RR = 0.1

def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_d3_train() -> pd.DataFrame:
    p = repo_root() / "data" / "processed" / "dataset3_train.parquet"
    return pd.read_parquet(p)


def pick_calm_and_vol_windows(
    df_t: pd.DataFrame,
    window: int = WINDOW,
) -> tuple[int, int]:
    """
    Pick a calm window (regime[t..t+W] all == 0) and a volatile window
    (regime[t..t+W] all == 1) from the same ticker. Returns (calm_start, vol_start).
    """
    regime = df_t["regime"].astype(float).to_numpy()
    n = len(regime)

    is_high = (regime == 1).astype(float)
    is_low  = (regime == 0).astype(float)

    cs = np.cumsum(np.r_[0.0, is_high])
    cumlo = np.cumsum(np.r_[0.0, is_low])
    win_high = (cs[window:] - cs[:-window])     
    win_low  = (cumlo[window:] - cumlo[:-window])

    vol_candidates  = np.where(win_high == window)[0]
    calm_candidates = np.where(win_low  == window)[0]

    if vol_candidates.size == 0 or calm_candidates.size == 0:
        vol_start  = int(np.argmax(win_high))
        calm_start = int(np.argmax(win_low))
    else:
        vol_start  = int(vol_candidates[len(vol_candidates) // 2])
        calm_start = int(calm_candidates[len(calm_candidates) // 2])

    return calm_start, vol_start


def compute_rp_and_rqa(
    df_t: pd.DataFrame,
    start: int,
    cfg: RQAConfig,
    eps_fixed: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Build RP exactly the way the project pipeline does:
      - take cols (log_return, rv) over the window
      - standardize columns
      - joint delay-embed with (m, tau)
      - threshold pairwise distances at eps_fixed
    Returns (R matrix, RQA-dict).
    """
    cols = ["log_return", "rv"]
    arr = df_t[cols].to_numpy(dtype=float, copy=False)
    W = arr[start : start + cfg.window, :]
    if cfg.standardize:
        W = _standardize_cols(W)
    X = _delay_embed_joint(W, cfg.embed)

    D = _pairwise_distances(X)
    R = _recurrence_matrix(D, eps_fixed, cfg.exclude_diagonal)
    feats = _rqa_from_embedded(X, cfg, eps_fixed)
    return R, feats


def fig_a_calm_vs_vol(out_dir: Path) -> None:
    train = load_d3_train()
    cfg = RQAConfig(
        window=WINDOW, step=20, recurrence_rate=RR,
        embed=EmbeddingConfig(m=M, tau=TAU), mode="joint",
    )

    candidate_tickers = sorted(train["ticker"].unique())
    chosen = None
    for ticker in candidate_tickers:
        df_t = train[train["ticker"] == ticker].reset_index(drop=True)
        if len(df_t) < 2000:
            continue
        regime = df_t["regime"].astype(float).to_numpy()
        if (regime == 1).sum() < WINDOW * 3 or (regime == 0).sum() < WINDOW * 3:
            continue
        chosen = (ticker, df_t)
        break
    if chosen is None:
        raise RuntimeError("No ticker with both regimes found.")
    ticker, df_t = chosen

    calm_start, vol_start = pick_calm_and_vol_windows(df_t, WINDOW)

    eps = estimate_epsilon_from_train(
        df_t[["log_return", "rv"]].to_numpy(dtype=float), cfg,
    )

    R_calm, feats_calm = compute_rp_and_rqa(df_t, calm_start, cfg, eps)
    R_vol,  feats_vol  = compute_rp_and_rqa(df_t, vol_start,  cfg, eps)

    fig = plt.figure(figsize=(13, 6.5))
    gs  = fig.add_gridspec(2, 2, height_ratios=[1, 4], hspace=0.25, wspace=0.18)

    panels = [
        ("Calm window  (regime = 0)", R_calm, feats_calm, calm_start, COLORS["calm"]),
        ("Volatile window  (regime = 1)", R_vol, feats_vol, vol_start, COLORS["vol"]),
    ]

    for col, (title, R, feats, start, edge) in enumerate(panels):
        ax_ts = fig.add_subplot(gs[0, col])
        ts = df_t["log_return"].iloc[start : start + WINDOW].to_numpy()
        ax_ts.plot(ts, color=edge, linewidth=1.0)
        ax_ts.axhline(0, color="grey", linewidth=0.5, alpha=0.5)
        ax_ts.set_title(title, fontsize=12, fontweight="bold", color=edge)
        ax_ts.set_ylabel("log return", fontsize=9)
        ax_ts.tick_params(labelsize=8)
        ax_ts.spines[["top", "right"]].set_visible(False)

        ax_rp = fig.add_subplot(gs[1, col])
        ax_rp.imshow(R, cmap="binary", origin="lower", aspect="equal", interpolation="nearest")
        ax_rp.set_xlabel("time index  i", fontsize=9)
        ax_rp.set_ylabel("time index  j", fontsize=9)
        ax_rp.tick_params(labelsize=8)
        for spine in ax_rp.spines.values():
            spine.set_edgecolor(edge)
            spine.set_linewidth(2.0)

        cap = (
            f"RR  = {feats['rr']:.3f}\n"
            f"DET = {feats['det']:.3f}\n"
            f"LAM = {feats['lam']:.3f}\n"
            f"L_max = {int(feats['lmax'])}\n"
            f"TT  = {feats['tt']:.2f}\n"
            f"ENTR = {feats['entr']:.3f}"
        )
        ax_rp.text(
            1.03, 0.5, cap, transform=ax_rp.transAxes,
            va="center", ha="left", family="monospace", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor=edge, linewidth=1.5),
        )

    fig.suptitle(
        f"Recurrence plots — Dataset 3 (ticker {ticker})\n"
        f"window = {WINDOW} bars (~{WINDOW * 2}-min span),  m = {M},  τ = {TAU},  RR target = {RR}",
        fontsize=13, y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = out_dir / "fig_a_rp_calm_vs_volatile_d3.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def fig_b_rp_with_regime_axes(out_dir: Path) -> None:
    train = load_d3_train()
    cfg = RQAConfig(
        window=WINDOW, step=20, recurrence_rate=RR,
        embed=EmbeddingConfig(m=M, tau=TAU), mode="joint",
    )

    target_n = 400
    for ticker in sorted(train["ticker"].unique()):
        df_t = train[train["ticker"] == ticker].reset_index(drop=True)
        if len(df_t) < target_n + 200:
            continue
        regime = df_t["regime"].astype(float).to_numpy()

        for s in range(100, len(df_t) - target_n, 50):
            seg = regime[s : s + target_n]
            f_high = (seg == 1).mean()
            if 0.30 <= f_high <= 0.70:
                segment_start = s
                break
        else:
            continue
        break
    else:
        raise RuntimeError("Could not find a transition-containing segment.")

    seg = df_t.iloc[segment_start : segment_start + target_n].reset_index(drop=True)
    arr = seg[["log_return", "rv"]].to_numpy(dtype=float)
    W = _standardize_cols(arr)
    X = _delay_embed_joint(W, cfg.embed)
    D = _pairwise_distances(X)

    eps = estimate_epsilon_from_train(
        df_t[["log_return", "rv"]].to_numpy(dtype=float), cfg,
    )
    R = _recurrence_matrix(D, eps, cfg.exclude_diagonal)

    span = (cfg.embed.m - 1) * cfg.embed.tau
    regime_emb = seg["regime"].astype(int).to_numpy()[span:]
    n = R.shape[0]
    regime_emb = regime_emb[:n]

    fig = plt.figure(figsize=(8.5, 8.5))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1, 24], height_ratios=[24, 1],
        wspace=0.02, hspace=0.02,
    )

    ax_main = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[0, 0], sharey=ax_main)
    ax_bot  = fig.add_subplot(gs[1, 1], sharex=ax_main)

    ax_main.imshow(R, cmap="binary", origin="lower", aspect="equal",
                   interpolation="nearest")
    ax_main.set_xticks([])
    ax_main.set_yticks([])

    cmap_regime = matplotlib.colors.ListedColormap([COLORS["calm"], COLORS["vol"]])
    ax_bot.imshow(regime_emb.reshape(1, -1), aspect="auto", cmap=cmap_regime,
                  origin="lower", interpolation="nearest", vmin=0, vmax=1,
                  extent=(-0.5, n - 0.5, 0, 1))
    ax_bot.set_yticks([])
    ax_bot.set_xlim(-0.5, n - 0.5)
    ax_bot.set_xlabel("time index  i", fontsize=10)

    ax_left.imshow(regime_emb.reshape(-1, 1), aspect="auto", cmap=cmap_regime,
                   origin="lower", interpolation="nearest", vmin=0, vmax=1,
                   extent=(0, 1, -0.5, n - 0.5))
    ax_left.set_xticks([])
    ax_left.set_ylim(-0.5, n - 0.5)
    ax_left.set_ylabel("time index  j", fontsize=10)

    legend_handles = [
        mpatches.Patch(color=COLORS["calm"], label="low-vol  (regime 0)"),
        mpatches.Patch(color=COLORS["vol"],  label="high-vol (regime 1)"),
    ]
    ax_main.legend(handles=legend_handles, loc="upper left", fontsize=9,
                   framealpha=0.9)

    fig.suptitle(
        f"Recurrence plot with regime-shaded axes — Dataset 3 (ticker {ticker})\n"
        f"{n} embedded points;  blocks of high recurrence ↔ within-regime persistence",
        fontsize=12, y=0.96,
    )
    out = out_dir / "fig_b_rp_with_regime_axes_d3.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def fig_c_rqa_distributions(out_dir: Path, n_tickers: int = 30) -> None:
    """
    For a sample of tickers, slide a non-overlapping window across the series,
    compute the 6 RQA measures per window, and label each window by the
    *modal* regime within it (>= 75% one class). Then box-plot each measure
    by regime class.
    """
    train = load_d3_train()
    cfg = RQAConfig(
        window=WINDOW, step=WINDOW, recurrence_rate=RR,  
        embed=EmbeddingConfig(m=M, tau=TAU), mode="joint",
    )

    rows = []
    tickers = sorted(train["ticker"].unique())[:n_tickers]
    for ticker in tickers:
        df_t = train[train["ticker"] == ticker].reset_index(drop=True)
        if len(df_t) < cfg.window * 3:
            continue
        eps = estimate_epsilon_from_train(
            df_t[["log_return", "rv"]].to_numpy(dtype=float), cfg,
        )
        n = len(df_t)
        for start in range(0, n - cfg.window + 1, cfg.window):
            R, feats = compute_rp_and_rqa(df_t, start, cfg, eps)
            seg_regime = df_t["regime"].iloc[start : start + cfg.window].astype(float)
            f_high = float(seg_regime.mean())
            if f_high >= 0.75:
                cls = "high"
            elif f_high <= 0.25:
                cls = "low"
            else:
                continue
            rows.append({"ticker": ticker, "regime": cls, **feats})

    df = pd.DataFrame(rows)
    print(f"Collected {len(df)} pure-regime windows from {df['ticker'].nunique()} tickers")
    print(df.groupby("regime").size())

    measures = ["rr", "det", "lam", "lmax", "tt", "entr"]
    nice = {
        "rr":  "RR (recurrence rate)",
        "det": "DET (determinism)",
        "lam": "LAM (laminarity)",
        "lmax": "L_max",
        "tt":   "TT (trapping time)",
        "entr": "ENTR (diag-entropy)",
    }

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, m in zip(axes.flat, measures):
        low  = df.loc[df["regime"] == "low",  m].to_numpy()
        high = df.loc[df["regime"] == "high", m].to_numpy()
        bp = ax.boxplot(
            [low, high],
            labels=["low\n(0)", "high\n(1)"],
            patch_artist=True, widths=0.55,
            medianprops=dict(color="black", linewidth=1.5),
            flierprops=dict(marker="o", markersize=2, alpha=0.4),
        )
        for patch, c in zip(bp["boxes"], [COLORS["calm"], COLORS["vol"]]):
            patch.set_facecolor(c)
            patch.set_alpha(0.65)
            patch.set_edgecolor("black")
        ax.set_title(nice[m], fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)

        try:
            from scipy.stats import mannwhitneyu
            u, p = mannwhitneyu(low, high, alternative="two-sided")
            ax.text(0.5, 0.97, f"MW p = {p:.1e}",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=8, color="dimgray")
        except Exception:
            pass

    fig.suptitle(
        f"Distribution of RQA measures by regime — D3 ({df['ticker'].nunique()} tickers, "
        f"{len(df)} pure-regime windows of {WINDOW} bars)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = out_dir / "fig_c_rqa_distribution_by_regime_d3.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    out_dir = repo_root() / "figures" / "checkpoint_followup"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_a_calm_vs_vol(out_dir)
    fig_b_rp_with_regime_axes(out_dir)
    fig_c_rqa_distributions(out_dir, n_tickers=30)

    print(f"\nAll three figures saved to {out_dir}/")


if __name__ == "__main__":
    main()