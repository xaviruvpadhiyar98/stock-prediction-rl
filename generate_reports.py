
import pandas as pd
from pathlib import Path
from config import TRAIN_TEST_SPLIT_PERCENT
from config import TICKERS, STOCK_DATA_SAVE_DIR, FILENAME


parquet_filename = Path(STOCK_DATA_SAVE_DIR) / FILENAME
df = pd.read_parquet(parquet_filename, engine="fastparquet")[
    ["Date", "Ticker", "Close"]
]
df = df.dropna(axis=0).reset_index(drop=True)
df.index = df["Date"].factorize()[0]
df["Buy/Sold/Hold"] = 0.0
df = df.loc[
    df.groupby(level=0).count()["Date"] == len(TICKERS)
]

train_size = df.index.values[-1] - int(
    df.index.values[-1] * TRAIN_TEST_SPLIT_PERCENT
)
train_df = df.loc[:train_size]
trade_df = df.loc[train_size + 1 :]
trade_df = trade_df.pivot(index="Date", columns="Ticker", values="Close").rename_axis(None, axis=1)
trade_df = trade_df[sorted(trade_df.columns.tolist())].reset_index()
trade_df


results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)
df = pd.read_parquet(results_dir/"all-model-eval-results.parquet", engine="fastparquet")
df


best_model = df.groupby(["model"]).last().sort_values("holdings", ascending=False).iloc[0].name
best_model


best_model_df = df[df["model"] == best_model].reset_index(drop=True)
first_row = {
    "model": best_model,
    "holdings": 100_000,
    "reward": 0
}
first_row.update({ticker:0 for ticker in TICKERS})
first_row = pd.DataFrame([first_row])
best_model_df = pd.concat([first_row, best_model_df]).reset_index(drop=True)
best_model_df.columns = [col+" Buy/Sell" if ".NS" in col else col for col in best_model_df.columns]
best_model_df


trade_df = pd.concat([trade_df, best_model_df], axis=1)
trade_df


daily_return_df = trade_df[["Date", "holdings"]].copy()
daily_return_df.loc[:, "daily_return"] = daily_return_df["holdings"].pct_change(1)
daily_return_df = daily_return_df.set_index("Date")
daily_return_df = pd.Series(daily_return_df["daily_return"], index=daily_return_df.index)
daily_return_df


from pyfolio import timeseries
perf_stats_all = timeseries.perf_stats(
    returns=daily_return_df,
    positions=None,
    transactions=None,
    turnover_denom="AGB",
)
print(perf_stats_all)


from pyfolio import create_full_tear_sheet

print(create_full_tear_sheet(returns=daily_return_df))


