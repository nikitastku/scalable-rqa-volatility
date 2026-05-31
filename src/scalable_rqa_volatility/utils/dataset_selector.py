"""
Utility for switching between datasets in training scripts.

Usage in any training script:
    from scalable_rqa_volatility.utils.dataset_selector import add_dataset_arg, load_splits

    parser = argparse.ArgumentParser()
    add_dataset_arg(parser)  # adds --dataset {1, 2} flag
    args = parser.parse_args()

    train, val, test = load_splits(args.dataset)

This keeps all training scripts dataset-agnostic.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def repo_root() -> Path:
    """Find the repo root (parent of scripts/)."""
    return Path(__file__).resolve().parents[2]


def add_dataset_arg(parser: argparse.ArgumentParser) -> None:
    """Add --dataset argument to any training script's argparse."""
    parser.add_argument(
        "--dataset",
        type=int,
        default=1,
        choices=[1, 2],
        help="Which dataset to use: 1 = Core_TimeSeries (synthetic, RQ3), 2 = S&P 500 Macro-Financial (real, RQ1)",
    )


def load_splits(dataset: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test splits for the specified dataset."""
    root = repo_root()
    prefix = f"dataset{dataset}"
    processed = root / "data" / "processed"

    train = pd.read_parquet(processed / f"{prefix}_train.parquet")
    val = pd.read_parquet(processed / f"{prefix}_val.parquet")
    test = pd.read_parquet(processed / f"{prefix}_test.parquet")

    required = {"log_return", "rv", "regime"}
    for name, df in [("train", train), ("val", val), ("test", test)]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Dataset {dataset} {name} split missing columns: {missing}")

    return train, val, test