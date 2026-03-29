from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scalable_rqa_volatility.data.loaders import load_core_timeseries
from scalable_rqa_volatility.data.splits import SplitConfig, time_series_split
from scalable_rqa_volatility.logging_utils import get_logger
from scalable_rqa_volatility.utils.io import DataPaths, ensure_dirs, write_table
from scalable_rqa_volatility.utils.seed import set_global_seed
from scalable_rqa_volatility.volatility.labeling import RollingQuantileConfig, label_regimes_rolling_quantile
from scalable_rqa_volatility.volatility.realized import VolatilityConfig, add_returns_and_volatility


@dataclass(frozen=True)
class PipelineConfig:
    """Minimal pipeline configuration."""

    repo_root: Path
    dataset_path: Path
    seed: int = 42


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    cfg = PipelineConfig(
        repo_root=repo_root,
        dataset_path=repo_root / "data" / "raw" / "Core_TimeSeries.csv",
    )

    logger = get_logger()
    set_global_seed(cfg.seed)

    paths = DataPaths.from_repo_root(cfg.repo_root)
    ensure_dirs(paths.raw, paths.interim, paths.processed)

    logger.info(f"Loading dataset: {cfg.dataset_path}")
    df = load_core_timeseries(cfg.dataset_path)

    logger.info(f"Rows={len(df)} Cols={len(df.columns)} DateRange={df['Date'].min()}..{df['Date'].max()}")

    df = add_returns_and_volatility(df, VolatilityConfig(rv_window=20))
    df = df.dropna(subset=["log_return", "rv"]).reset_index(drop=True)

    df = label_regimes_rolling_quantile(
        df,
        RollingQuantileConfig(
            target_vol_col="rv",
            label_col="regime",
            lookback=252,
            quantile=0.7,
            min_periods=252,
        ),
    )
    df = df.dropna(subset=["regime"]).reset_index(drop=True)

    splits = time_series_split(df, SplitConfig(train_frac=0.7, val_frac=0.15))

    for name, part in splits.items():
        out_path = paths.processed / f"dataset1_{name}.parquet"
        write_table(part, out_path, fmt="parquet")
        logger.info(f"Saved {name}: {out_path} rows={len(part)}")

    summary = pd.DataFrame(
        {
            "split": list(splits.keys()),
            "rows": [len(splits[k]) for k in splits.keys()],
            "regime_mean": [float(splits[k]["regime"].mean()) for k in splits.keys()],
        }
    )
    write_table(summary, paths.processed / "dataset1_summary.csv", fmt="csv")
    logger.info("Saved summary: data/processed/dataset1_summary.csv")


if __name__ == "__main__":
    main()