import polars as pl
from pathlib import Path
from config import STOCK_DATA_SAVE_DIR, TRAIN_TEST_SPLIT_PERCENT
import numpy as np


def create_numpy_array(df, cols):
    arr = []
    for i, (name, data) in enumerate(df.groupby("Date")):
        new_arr = data.select(cols).to_numpy().flatten()
        arr.append(new_arr)
    return np.asarray(arr)


def main():
    parquet_filename = Path(STOCK_DATA_SAVE_DIR) / "random-778-tickers.parquet"
    df = pl.read_parquet(parquet_filename)[["Date", "Ticker", "Close"]]
    df = df.filter(df["Ticker"].str.ends_with(".NS"))
    TICKERS = df["Ticker"].unique()
    TICKERS = [t for t in TICKERS if t.endswith(".NS")]
    max_value_to_be_considered = (
        df.groupby("Ticker").count()["count"].value_counts(sort=True).row(0)[0]
    )

    features_df = pl.DataFrame()
    for i, ticker in enumerate(TICKERS):
        tmp_df = df.filter(pl.col("Ticker") == ticker).reverse()
        tmp_df = tmp_df.slice(0, max_value_to_be_considered).reverse()
        tmp_df = tmp_df.unique(subset=["Date"], maintain_order=True)
        if tmp_df.shape[0] == max_value_to_be_considered:
            tmp_df = tmp_df.with_columns(
                [
                    pl.col("Close").shift(hour).alias(f"PAST_{hour}_HOUR")
                    for hour in range(1, 15)
                ]
            )
            features_df = pl.concat([features_df, tmp_df], how="diagonal")

    technical_indicators = [f"PAST_{hour}_HOUR" for hour in range(1, 15)]

    features_df = features_df.drop_nulls()
    features_df = features_df.sort("Date", descending=False)
    features_df = features_df.filter(
        pl.count("Ticker").over("Date") == features_df["Ticker"].n_unique()
    )
    features_df = features_df.with_columns(pl.lit(0.0).alias("Buy/Sold/Hold"))

    total = features_df.groupby("Ticker").count().row(0)[1]
    train_size = total - int(total * TRAIN_TEST_SPLIT_PERCENT)
    test_size = total - train_size

    train_end_index = train_size * features_df["Ticker"].n_unique()
    trade_end_index = test_size * features_df["Ticker"].n_unique()

    train_df = features_df.slice(0, train_end_index)
    trade_df = features_df.slice(train_end_index, trade_end_index)

    cols = train_df.columns
    cols.remove("Date")
    cols.remove("Ticker")

    train_arrays = create_numpy_array(train_df, cols)
    trade_arrays = create_numpy_array(trade_df, cols)

    print(f"Train Array len {len(train_arrays)}")
    print(f"Trade Array len {len(trade_arrays)}")

    with open(f"{STOCK_DATA_SAVE_DIR}/train-trade.npy", "wb") as f:
        np.save(f, train_arrays, allow_pickle=True, fix_imports=True)
        np.save(f, trade_arrays, allow_pickle=True, fix_imports=True)
        np.save(f, np.array(TICKERS), allow_pickle=True, fix_imports=True)
        np.save(f, np.array(technical_indicators), allow_pickle=True, fix_imports=True)


if __name__ == "__main__":
    main()
