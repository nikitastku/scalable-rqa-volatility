from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]


def find_date_col(df: pd.DataFrame) -> str:
    for c in ["date", "Date", "ds", "DS", "time", "Time", "timestamp", "Timestamp"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if c.lower() in {"date", "ds", "time", "timestamp"}:
            return c
    if isinstance(df.index, pd.DatetimeIndex):
        return "__index__"
    raise ValueError(f"No date-like column found. Columns={list(df.columns)}")


def main() -> None:
    raw_path = ROOT / "data" / "raw" / "Core_TimeSeries.csv"
    train_path = ROOT / "data" / "processed" / "dataset1_train.parquet"

    raw = pd.read_csv(raw_path)
    train = pd.read_parquet(train_path)

    raw_date = find_date_col(raw)
    train_date = find_date_col(train)

    if raw_date == "__index__":
        raw = raw.reset_index().rename(columns={"index": "date_tmp"})
        raw_date = "date_tmp"
    if train_date == "__index__":
        train = train.reset_index().rename(columns={"index": "date_tmp"})
        train_date = "date_tmp"

    raw[raw_date] = pd.to_datetime(raw[raw_date])
    train[train_date] = pd.to_datetime(train[train_date])

    raw = raw.sort_values(raw_date).reset_index(drop=True)
    train = train.sort_values(train_date).reset_index(drop=True)

    if "Volatility_Range" not in raw.columns:
        raise ValueError(f"Raw file missing Volatility_Range. Columns={list(raw.columns)}")
    if "rv" not in train.columns or "regime" not in train.columns:
        raise ValueError(f"Processed file missing rv/regime. Columns={list(train.columns)}")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    axes[0].plot(raw[raw_date], raw["Volatility_Range"].astype(float))
    axes[0].set_title("Dataset 1 — Before processing (raw Volatility_Range)")
    axes[0].set_ylabel("Volatility_Range")
    axes[0].set_xlabel("Date")

    axes[1].plot(train[train_date], train["rv"].astype(float), label="rv")
    axes[1].plot(train[train_date], train["regime"].astype(float), linestyle="--", label="regime")
    axes[1].set_title("Dataset 1 — After processing (rv + regime)")
    axes[1].set_ylabel("value")
    axes[1].set_xlabel("Date")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()