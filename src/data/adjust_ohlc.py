
"""Create adjusted ETF OHLC data from FinanceDataReader prices."""

from pathlib import Path

import numpy as np
import pandas as pd

from tests.test_finrl_env import df


INPUT_PATH = Path("data/processed/etf_daily_panel.csv")
OUTPUT_PATH = Path("data/processed/etf_adjusted_panel.csv")

TICKERS = {"SPY", "IEF", "GLD", "SHV"}


def build_adjusted_panel(
    input_path=INPUT_PATH,
    output_path=OUTPUT_PATH,
):
    df = pd.read_csv(input_path, parse_dates=["date"])

    required = {
        "date", "tic", "open", "high", "low",
        "close", "adj_close", "volume"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.sort_values(
        ["date", "tic"]
    ).reset_index(drop=True)

    if set(df["tic"].unique()) != TICKERS:
        raise ValueError("Unexpected ETF tickers")

    if df.duplicated(["date", "tic"]).any():
        raise ValueError("Duplicate date/ticker combinations")

    counts = df.groupby("date")["tic"].nunique()

    if not counts.eq(len(TICKERS)).all():
        raise ValueError("Incomplete ETF panel")

    price_columns = [
        "open", "high", "low", "close", "adj_close"
    ]

    if df[price_columns].isna().any().any():
        raise ValueError("Missing price values")

    if not np.isfinite(
        df[price_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite price values")

    if (df[price_columns] <= 0).any().any():
        raise ValueError("Non-positive prices")

    # Adjustment factor
    df["adjustment_factor"] = (
        df["adj_close"] / df["close"]
    )

    # Adjust OHLC using the same factor
    for col in ["open", "high", "low"]:
        df[f"adj_{col}"] = (
            df[col] * df["adjustment_factor"]
        )

    # Validate adjusted prices
    adjusted_columns = [
        "adj_open", "adj_high", "adj_low", "adj_close"
    ]

    if not np.isfinite(
        df[adjusted_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("Invalid adjusted OHLC")

    if (df["adj_high"] < df["adj_low"]).any():
        raise ValueError("Adjusted high < adjusted low")
    # Floating-point tolerance for adjusted OHLC
    tol = 1e-6

    invalid = (
    (df["adj_high"] + tol < df["adj_open"]) |
    (df["adj_high"] + tol < df["adj_close"]) |
    (df["adj_low"] - tol > df["adj_open"]) |
    (df["adj_low"] - tol > df["adj_close"])
    )

    if invalid.any():
        columns = [
            "date", "tic",
            "open", "high", "low", "close",
            "adj_open", "adj_high", "adj_low", "adj_close",
            "adjustment_factor",
        ]

        print("\nAdjusted OHLC inconsistency detected")
        print("Invalid rows:", invalid.sum())

        print(
            df.loc[invalid, columns]
            .to_string(index=False)
        )

        raise ValueError(
            "Adjusted OHLC consistency failure: inspect rows above"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    df.to_csv(output_path, index=False)

    print("Adjusted OHLC validation PASS")
    print("Rows:", len(df))
    print("Dates:", df["date"].nunique())
    print("Tickers:", sorted(df["tic"].unique()))
    print("Saved:", output_path)

    return df


if __name__ == "__main__":
    build_adjusted_panel()
