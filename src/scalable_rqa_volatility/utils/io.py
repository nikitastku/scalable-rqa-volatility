from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd


@dataclass(frozen=True)
class DataPaths:
    """Standard project data paths relative to repository root."""

    root: Path
    raw: Path
    interim: Path
    processed: Path

    @staticmethod
    def from_repo_root(repo_root: Path) -> "DataPaths":
        data_dir = repo_root / "data"
        return DataPaths(
            root=repo_root,
            raw=data_dir / "raw",
            interim=data_dir / "interim",
            processed=data_dir / "processed",
        )


def ensure_dirs(*paths: Path) -> None:
    """Create directories if they do not exist."""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def read_table(path: Path) -> pd.DataFrame:
    """Read a tabular dataset from CSV or Parquet."""
    if not path.exists():
        raise FileNotFoundError(str(path))

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported file type: {suffix}")


def write_table(
    df: pd.DataFrame,
    path: Path,
    fmt: Literal["parquet", "csv"] = "parquet",
) -> None:
    """Write a tabular dataset to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(path, index=False)
        return
    if fmt == "csv":
        df.to_csv(path, index=False)
        return
    raise ValueError(f"Unsupported fmt: {fmt}")