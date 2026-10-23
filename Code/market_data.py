"""
market_data.py
==============
Frozen local price snapshots, so every script — and every rerun — sees
identical data.

yfinance returns dividend-adjusted prices, and Yahoo re-adjusts the whole
history after each dividend, so two downloads weeks apart differ slightly.
The first request for a ticker downloads SNAPSHOT_START → END_DATE once and
saves it to {data_dir}/{ticker}_close.csv; every later request reads that
file and slices it by date. All periods of a ticker therefore come from the
same download.

Functions:
  download_snapshot(ticker, data_dir)          → path of the saved CSV
  load_close(ticker, start, end, data_dir)     → pd.Series of close prices

CLI (create snapshots up front, before launching runs in parallel):
  py -3.11 market_data.py SPY QQQ GLD
  py -3.11 market_data.py SPY --refresh        # re-download — changes results!
"""

import argparse
import os
import sys

import pandas as pd
import yfinance as yf

# UTF-8 stdout: progress output uses → / ±, which crash a cp1252 Windows console
# or redirected log; line buffering keeps log files current during long runs.
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

SNAPSHOT_START = "2015-04-27"   # earliest period start used by the experiments (10y)
END_DATE       = "2025-01-01"
DATA_DIR       = "data"


def snapshot_path(ticker: str, data_dir: str = DATA_DIR) -> str:
    return os.path.join(data_dir, f"{ticker}_close.csv")


def download_snapshot(ticker: str, data_dir: str = DATA_DIR) -> str:
    """Download adjusted daily closes for SNAPSHOT_START ≤ date < END_DATE and save them."""
    df = yf.download(
        ticker,
        start=SNAPSHOT_START,
        end=END_DATE,
        interval="1d",
        progress=False,
        auto_adjust=True,      # pinned: the yfinance default changed across versions
    )
    if df.empty:
        raise ValueError(f"No data for {ticker}.")

    close = df["Close"]
    if isinstance(close, pd.DataFrame):   # newer yfinance: one column per ticker
        close = close.iloc[:, 0]
    close = close.dropna().rename("close")
    close.index.name = "date"

    # Write to a temp file then rename, so a reader never sees a partial file
    os.makedirs(data_dir, exist_ok=True)
    path = snapshot_path(ticker, data_dir)
    close.to_csv(path + ".tmp")
    os.replace(path + ".tmp", path)
    print(f"  Saved {ticker} snapshot: {len(close)} days "
          f"({close.index[0].date()} → {close.index[-1].date()}) → {path}")
    return path


def load_close(ticker: str, start: str, end: str = END_DATE, data_dir: str = DATA_DIR) -> pd.Series:
    """
    Close prices for start ≤ date < end (same convention as yf.download),
    read from the local snapshot; downloads the snapshot if it does not exist.
    """
    if start < SNAPSHOT_START or end > END_DATE:
        raise ValueError(f"Requested {start} → {end} is outside the snapshot range "
                         f"{SNAPSHOT_START} → {END_DATE}; widen it in market_data.py.")

    path = snapshot_path(ticker, data_dir)
    if not os.path.exists(path):
        download_snapshot(ticker, data_dir)

    close = pd.read_csv(path, index_col="date", parse_dates=True)["close"]
    close = close[(close.index >= start) & (close.index < end)]
    if close.empty:
        raise ValueError(f"No data for {ticker} in {start} → {end}.")
    return close


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Create local price snapshots.")
    p.add_argument("tickers", nargs="+", help="e.g. SPY QQQ GLD")
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--refresh", action="store_true",
                   help="re-download even if a snapshot exists (results will change)")
    args = p.parse_args()

    for t in (t.upper() for t in args.tickers):
        if os.path.exists(snapshot_path(t, args.data_dir)) and not args.refresh:
            print(f"  {t}: snapshot exists ({snapshot_path(t, args.data_dir)}) — kept. "
                  f"Use --refresh to re-download.")
        else:
            download_snapshot(t, args.data_dir)
