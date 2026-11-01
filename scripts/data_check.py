import pandas as pd


df = pd.read_csv(
    "data/processed/etf_daily_panel.csv",
    parse_dates=["date"],
)

print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print("Tickers:", sorted(df["tic"].unique()))
print("Dates:", df["date"].nunique())

print("\nETF별 데이터 개수")
print(df.groupby("tic").size())

print("\n날짜별 ETF 개수")
print(df.groupby("date")["tic"].nunique().value_counts())

print("\n중복 데이터")
print(df.duplicated(["date", "tic"]).sum())