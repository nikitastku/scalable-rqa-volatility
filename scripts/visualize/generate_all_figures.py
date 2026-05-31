"""
Generate thesis figures for dataset overviews and distribution-shift comparison.

This script creates the main preprocessing and dataset-description figures used
to compare the three processed datasets. It visualizes price or volatility
series, regime labels, train/validation/test split boundaries, and train-test
realized-volatility distribution shifts.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.figsize": (10, 6),
})

COLORS = {
    "primary": "#1E2761",
    "accent": "#3B82F6",
    "green": "#10B981",
    "red": "#EF4444",
    "orange": "#F59E0B",
    "purple": "#8B5CF6",
    "gray": "#94A3B8",
    "lightgray": "#F1F5F9",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_split(dataset: int, name: str) -> pd.DataFrame:
    return pd.read_parquet(repo_root() / "data" / "processed" / f"dataset{dataset}_{name}.parquet")


def savefig(fig, name: str):
    out_dir = repo_root() / "figures" / "general_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_dataset1_overview():
    """Fig 1.1: Dataset 1 time series, RV, regime labels, and splits."""
    print("Generating Dataset 1 overview...")
    train = load_split(1, "train")
    val = load_split(1, "val")
    test = load_split(1, "test")
    full = pd.concat([train, val, test], ignore_index=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    ax.plot(full.index, full["Close_Price"], color=COLORS["primary"], linewidth=0.5)
    ax.set_ylabel("Close Price")
    ax.set_title("Dataset 1: Synthetic S&P 500 (1902-2017)")

    n_tr, n_va = len(train), len(val)
    for a in axes:
        a.axvline(n_tr, color=COLORS["orange"], linestyle="--", alpha=0.7, label="Train/Val")
        a.axvline(n_tr + n_va, color=COLORS["red"], linestyle="--", alpha=0.7, label="Val/Test")

    ax = axes[1]
    ax.plot(full.index, full["rv"], color=COLORS["accent"], linewidth=0.5)
    ax.set_ylabel("Realized Volatility")

    ax = axes[2]
    regime = full["regime"].astype(float)
    ax.fill_between(full.index, 0, regime, alpha=0.5, color=COLORS["red"], label="High vol regime")
    ax.set_ylabel("Regime")
    ax.set_xlabel("Time index")
    ax.legend(loc="upper right")

    fig.tight_layout()
    savefig(fig, "1_1_dataset1_overview")


def plot_dataset2_overview():
    """Fig 1.2: Dataset 2 time series with macro features."""
    print("Generating Dataset 2 overview...")
    train = load_split(2, "train")
    val = load_split(2, "val")
    test = load_split(2, "test")
    full = pd.concat([train, val, test], ignore_index=True)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    n_tr, n_va = len(train), len(val)

    ax = axes[0]
    ax.plot(full.index, full["SP500"], color=COLORS["primary"], linewidth=0.8)
    ax.set_ylabel("S&P 500")
    ax.set_title("Dataset 2: S&P 500 Macro-Financial (2010-2019)")

    ax = axes[1]
    ax.plot(full.index, full["VIX"], color=COLORS["orange"], linewidth=0.8)
    ax.set_ylabel("VIX")

    ax = axes[2]
    ax.plot(full.index, full["rv"], color=COLORS["accent"], linewidth=0.8)
    ax.set_ylabel("Realized Volatility")

    ax = axes[3]
    regime = full["regime"].astype(float)
    ax.fill_between(full.index, 0, regime, alpha=0.5, color=COLORS["red"])
    ax.set_ylabel("Regime")
    ax.set_xlabel("Time index")

    for a in axes:
        a.axvline(n_tr, color=COLORS["orange"], linestyle="--", alpha=0.7)
        a.axvline(n_tr + n_va, color=COLORS["red"], linestyle="--", alpha=0.7)

    fig.tight_layout()
    savefig(fig, "1_2_dataset2_overview")


def plot_dataset3_overview():
    """Fig 1.3: Dataset 3 sample stocks."""
    print("Generating Dataset 3 overview...")
    train = load_split(3, "train")

    tickers = sorted(train["ticker"].unique())
    sample_tickers = [tickers[0], tickers[100], tickers[250], tickers[400]]

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=False)
    fig.suptitle("Dataset 3: Intraday S&P 500 (503 Stocks, 2-min bars)", fontsize=14)

    for i, ticker in enumerate(sample_tickers):
        ax = axes[i]
        df_t = train[train["ticker"] == ticker].reset_index(drop=True)
        ax.plot(df_t.index, df_t["rv"], color=COLORS["accent"], linewidth=0.5)
        ax.set_ylabel(f"{ticker}\nRV")
        if i == len(sample_tickers) - 1:
            ax.set_xlabel("Bar index")

    fig.tight_layout()
    savefig(fig, "1_3_dataset3_overview")


def plot_distribution_shift_comparison():
    """Fig 1.4: Distribution shift comparison across datasets."""
    print("Generating distribution shift comparison...")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    datasets = [
        (1, "D1: Synthetic\n(RV ratio: 0.17x)", True),
        (2, "D2: Daily S&P 500\n(RV ratio: 0.97x)", False),
        (3, "D3: Intraday 503 stocks\n(RV ratio: 0.95x)", False),
    ]

    for idx, (d, title, severe) in enumerate(datasets):
        ax = axes[idx]
        train = load_split(d, "train")
        test = load_split(d, "test")

        rv_tr = train["rv"].dropna().astype(float)
        rv_te = test["rv"].dropna().astype(float)

        if len(rv_tr) > 50000:
            rv_tr = rv_tr.sample(50000, random_state=42)
            rv_te = rv_te.sample(min(50000, len(rv_te)), random_state=42)

        ax.hist(rv_tr, bins=50, alpha=0.6, color=COLORS["accent"], label="Train", density=True)
        ax.hist(rv_te, bins=50, alpha=0.6, color=COLORS["red"], label="Test", density=True)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("RV")
        ax.legend()
        if severe:
            ax.annotate("SEVERE SHIFT", xy=(0.5, 0.9), xycoords="axes fraction",
                         ha="center", fontsize=10, color=COLORS["red"], fontweight="bold")

    fig.suptitle("Distribution Shift Comparison: Train vs Test RV", fontsize=13)
    fig.tight_layout()
    savefig(fig, "1_4_distribution_shift_comparison")


def main():
    print("=" * 60)
    print("  GENERATING ALL THESIS FIGURES")
    print("=" * 60)

    print("\n--- Raw Data & Preprocessing ---")
    plot_dataset1_overview()
    plot_dataset2_overview()
    plot_dataset3_overview()
    plot_distribution_shift_comparison()

    print("\n" + "=" * 60)
    print(f"  FIGURES SAVED TO: {repo_root() / 'figures' / 'general_figures'}")
    print("=" * 60)


if __name__ == "__main__":
    main()