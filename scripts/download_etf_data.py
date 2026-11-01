
"""Run from the project root:

python -m scripts.download_etf_data
"""

from src.data.etf_data import build_etf_panel


def main():
    panel = build_etf_panel()

    print("\nData preview:")
    print(panel.head(12))

    print("\nMissing values:")
    print(panel.isna().sum())

    print("\nPrice summary:")
    print(
        panel.groupby("tic")["close"].agg(
            ["count", "min", "max", "mean"]
        )
    )


if __name__ == "__main__":
    main()
