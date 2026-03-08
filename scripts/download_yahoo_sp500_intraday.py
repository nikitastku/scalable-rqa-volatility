from __future__ import annotations
import argparse
import os
import sys
import time
from datetime import datetime, timezone
from typing import List, Tuple
import pandas as pd
import yfinance as yf

DEFAULT_STEM = os.path.join("data", "raw", "yahoo_sp500_2m_60d")

def fetch_sp500_tickers() -> List[str]:
    import urllib.request
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read()
    tables = pd.read_html(html)
    df = tables[0]
    tickers = df["Symbol"].astype(str).tolist()
    tickers = [t.replace(".", "-").strip() for t in tickers]
    seen = set()
    out = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _flatten_cols(cols) -> List[str]:
    flat = []
    for c in cols:
        if isinstance(c, tuple):
            parts = [str(p) for p in c if p not in ("", None)]
            flat.append("_".join(parts))
        else:
            flat.append(str(c))
    return flat

def download_one(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
        group_by="column",
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    df.columns = _flatten_cols(df.columns)
    if "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "timestamp"})
    elif "Date" in df.columns:
        df = df.rename(columns={"Date": "timestamp"})
    if "timestamp" not in df.columns:
        return pd.DataFrame()
    df["ticker"] = ticker
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    rename_map = {}
    for c in df.columns:
        if c.startswith("open"):
            rename_map[c] = "open"
        elif c.startswith("high"):
            rename_map[c] = "high"
        elif c.startswith("low"):
            rename_map[c] = "low"
        elif c.startswith("close") and "adj" not in c:
            rename_map[c] = "close"
        elif "adj" in c and "close" in c:
            rename_map[c] = "adj_close"
        elif c.startswith("volume"):
            rename_map[c] = "volume"
    df = df.rename(columns=rename_map)
    keep = ["timestamp", "open", "high", "low", "close", "adj_close", "volume", "ticker"]
    present = [c for c in keep if c in df.columns]
    df = df[present]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    return df

def write_incremental_csv(
    tickers: List[str],
    out_csv: str,
    period: str,
    interval: str,
    sleep_s: float,
    max_tickers: int | None = None,
) -> Tuple[str, int, List[str]]:
    if max_tickers is not None:
        tickers = tickers[:max_tickers]
    safe_mkdir(os.path.dirname(out_csv))
    if os.path.exists(out_csv):
        try:
            os.remove(out_csv)
        except PermissionError:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_csv = out_csv.replace(".csv", f"_locked_{ts}.csv")
    header_written = False
    failed: List[str] = []
    total_rows = 0
    for i, t in enumerate(tickers, start=1):
        try:
            df = download_one(t, period=period, interval=interval)
            if df.empty:
                failed.append(t)
                continue
            df.to_csv(out_csv, mode="a", index=False, header=not header_written)
            header_written = True
            total_rows += len(df)
            if i % 25 == 0:
                print(f"[{i}/{len(tickers)}] rows so far: {total_rows:,}")
            time.sleep(sleep_s)
        except Exception as ex:
            failed.append(t)
            time.sleep(max(sleep_s, 0.5))
            print(f"Failed {t}: {ex}", file=sys.stderr)
    return out_csv, total_rows, failed

def csv_to_parquet(csv_path: str, parquet_path: str) -> int:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df["ticker"] = df["ticker"].astype("string")
    df.to_parquet(parquet_path, index=False)
    return len(df)

def write_metadata(stem: str, tickers: List[str], failed: List[str], period: str, interval: str) -> None:
    meta_path = stem + "_meta.txt"
    ts = datetime.now(timezone.utc).isoformat()
    ok = [t for t in tickers if t not in set(failed)]
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"created_utc: {ts}\n")
        f.write(f"period: {period}\n")
        f.write(f"interval: {interval}\n")
        f.write(f"tickers_requested: {len(tickers)}\n")
        f.write(f"tickers_succeeded: {len(ok)}\n")
        f.write(f"tickers_failed: {len(failed)}\n")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default=DEFAULT_STEM)
    ap.add_argument("--period", default="60d")
    ap.add_argument("--interval", default="2m")
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--max_tickers", type=int, default=None)
    args = ap.parse_args()
    out_csv = args.stem + ".csv"
    out_parquet = args.stem + ".parquet"
    print("Fetching S&P 500 tickers...")
    tickers = fetch_sp500_tickers()
    print(f"Tickers fetched: {len(tickers)}")
    out_csv, total_rows, failed = write_incremental_csv(
        tickers=tickers,
        out_csv=out_csv,
        period=args.period,
        interval=args.interval,
        sleep_s=args.sleep,
        max_tickers=args.max_tickers,
    )
    print(f"Total rows written (CSV): {total_rows:,}")
    print(f"Failed tickers: {len(failed)}")
    try:
        rows_parquet = csv_to_parquet(out_csv, out_parquet)
        print(f"Rows written (Parquet): {rows_parquet:,}")
    except Exception as ex:
        print(f"Parquet conversion failed: {ex}", file=sys.stderr)
    write_metadata(args.stem, tickers, failed, args.period, args.interval)

if __name__ == "__main__":
    main()