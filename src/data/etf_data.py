
"""Download and validate ETF OHLCV data using FinanceDataReader."""

from pathlib import Path

import FinanceDataReader as fdr
import numpy as np
import pandas as pd


TICKERS = ("SPY", "IEF", "GLD", "SHV")

START = "2010-01-01"
END = "2025-12-31"


def download_etf(ticker, start=START, end=END):
    """Download one ETF and standardize its columns."""

    df = fdr.DataReader(ticker, start, end)

    if df.empty:
        raise ValueError(f"{ticker}: empty response from FDR")

    df = df.copy()
    df.index = pd.to_datetime(df.index)

    required = ["Open", "High", "Low", "Close","Adj Close"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker}: missing columns {missing}")

    columns = required + (
        ["Volume"] if "Volume" in df.columns else []
    )

    df = df[columns].rename(
    columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
        }
    )

    df.index.name = "date"
    df = df.reset_index()
    df["tic"] = ticker

    df = df.sort_values("date").reset_index(drop=True)

    if df["date"].duplicated().any():
        raise ValueError(f"{ticker}: duplicate dates")

    if df[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError(f"{ticker}: missing OHLC prices")

    prices = df[["open", "high", "low", "close","adj_close"]]

    if prices.isna().any().any():
        raise ValueError(f"{ticker}: missing price values")

    if not np.isfinite(prices.to_numpy(dtype=float)).all():
        raise ValueError(f"{ticker}: non-finite prices")

    if (prices <= 0).any().any():
        raise ValueError(f"{ticker}: non-positive prices")

    if (df["high"] < df["low"]).any():
        raise ValueError(f"{ticker}: high < low")

    return df


def build_etf_panel(
    tickers=TICKERS,
    start=START,
    end=END,
    raw_dir="data/raw",
    processed_dir="data/processed",
):
    """Download, align and save an ETF panel."""

    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    frames = []

    for ticker in tickers:
        print(f"Downloading {ticker}...")

        df = download_etf(ticker, start, end)

        df.to_csv(
            raw_dir / f"{ticker}_{start}_{end}.csv",
            index=False,
        )

        print(
            f"  rows={len(df)}, "
            f"first={df['date'].min().date()}, "
            f"last={df['date'].max().date()}"
        )

        frames.append(df)

    panel = pd.concat(frames, ignore_index=True)

    # Retain only dates observed for every ETF.
    counts = panel.groupby("date")["tic"].nunique()
    common_dates = counts[counts == len(tickers)].index

    panel = panel[panel["date"].isin(common_dates)].copy()

    panel = panel.sort_values(
        ["date", "tic"]
    ).reset_index(drop=True)

    if panel.empty:
        raise ValueError("No common trading dates")

    if panel.duplicated(["date", "tic"]).any():
        raise ValueError("Duplicate date/ticker combinations")

    if panel.groupby("date")["tic"].nunique().ne(len(tickers)).any():
        raise ValueError("Incomplete ETF panel")

    output_path = processed_dir / "etf_daily_panel.csv"
    panel.to_csv(output_path, index=False)

    print("\nPanel validation PASS")
    print("Tickers:", sorted(panel["tic"].unique()))
    print("Trading dates:", panel["date"].nunique())
    print("Total rows:", len(panel))
    print("Saved:", output_path)

    return panel
