from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scalable_rqa_volatility.plots.io import ensure_dir


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_split(name: str) -> pd.DataFrame:
    return pd.read_parquet(repo_root() / "data" / "processed" / f"dataset1_{name}.parquet")


def main() -> None:
    out_dir = ensure_dir(repo_root() / "reports" / "figures")

    train = load_split("train").reset_index(drop=True)
    val = load_split("val").reset_index(drop=True)
    test = load_split("test").reset_index(drop=True)

    full = pd.concat(
        [
            train.assign(split="train"),
            val.assign(split="val"),
            test.assign(split="test"),
        ],
        axis=0,
    ).reset_index(drop=True)

    n_train = len(train)
    n_val = len(val)

    x = np.arange(len(full))
    rv = full["rv"].astype(float).to_numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, rv, linewidth=1)
    ax.axvline(n_train, linewidth=1)
    ax.axvline(n_train + n_val, linewidth=1)
    ax.set_title("Realized Volatility (rv) with Train/Val/Test Splits")
    ax.set_xlabel("Index")
    ax.set_ylabel("rv")
    fig.tight_layout()
    fig.savefig(out_dir / "dataset_rv_splits.png", dpi=200)
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(train["rv"].astype(float).to_numpy(), bins=60, alpha=0.7, label="train")
    ax.hist(val["rv"].astype(float).to_numpy(), bins=60, alpha=0.7, label="val")
    ax.hist(test["rv"].astype(float).to_numpy(), bins=60, alpha=0.7, label="test")
    ax.set_title("rv distribution by split")
    ax.set_xlabel("rv")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "dataset_rv_hist_by_split.png", dpi=200)
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    rv0 = full.loc[full["regime"].astype(int) == 0, "rv"].astype(float).to_numpy()
    rv1 = full.loc[full["regime"].astype(int) == 1, "rv"].astype(float).to_numpy()
    ax.boxplot([rv0, rv1], labels=["regime=0", "regime=1"])
    ax.set_title("rv by regime")
    ax.set_ylabel("rv")
    fig.tight_layout()
    fig.savefig(out_dir / "dataset_rv_by_regime.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()