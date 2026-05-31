"""
Load and validate the core time-series dataset.

This module defines the expected column schema for the raw core dataset and
provides a loader that reads the table, checks required OHLC columns, parses the
date column, sorts observations chronologically, converts price and volume
columns to numeric types, and raises clear validation errors when the input data
does not match the expected format.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scalable_rqa_volatility.utils.io import read_table


@dataclass(frozen=True)
class CoreDatasetSchema:
    date_col: str = "Date"
    open_col: str = "Open_Price"
    high_col: str = "High_Price"
    low_col: str = "Low_Price"
    close_col: str = "Close_Price"
    volume_col: str = "Volume"


def load_core_timeseries(path: Path, schema: CoreDatasetSchema | None = None) -> pd.DataFrame:
    schema = schema or CoreDatasetSchema()
    df = read_table(path)

    missing = [
        c
        for c in [schema.date_col, schema.open_col, schema.high_col, schema.low_col, schema.close_col]
        if c not in df.columns
    ]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    df[schema.date_col] = pd.to_datetime(df[schema.date_col], dayfirst=True, errors="raise")

    df = df.sort_values(schema.date_col).reset_index(drop=True)

    for c in [schema.open_col, schema.high_col, schema.low_col, schema.close_col]:
        df[c] = pd.to_numeric(df[c], errors="raise")

    if schema.volume_col in df.columns:
        df[schema.volume_col] = pd.to_numeric(df[schema.volume_col], errors="coerce")

    if df[schema.date_col].isna().any():
        raise ValueError("Found NaT values in date column after parsing.")
    if df[[schema.open_col, schema.high_col, schema.low_col, schema.close_col]].isna().any().any():
        raise ValueError("Found missing values in OHLC columns after numeric conversion.")

    return df