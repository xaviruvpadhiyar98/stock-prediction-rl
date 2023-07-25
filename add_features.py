from talib import BBANDS, EMA, RSI, MACD, DEMA, HT_TRENDLINE, KAMA


from talib import MA_Type
from config import (
    TICKERS,
    STOCK_DATA_SAVE_DIR,
    FILENAME,
    TRAIN_TEST_SPLIT_PERCENT,
    TECHNICAL_INDICATORS,
    NUMPY_FILENAME,
)
import pandas as pd
import numpy as np
from pathlib import Path
from logger_config import add_features_logger as log


def add_features():
    df = pd.read_parquet(Path(STOCK_DATA_SAVE_DIR) / FILENAME, engine="fastparquet")
    log.info(f"df loaded successfully \n{df.head().to_markdown()}")
    for ticker in TICKERS:
        close_value = df[df["Ticker"] == ticker]["Close"].values
        filter = df["Ticker"] == ticker
        df.loc[filter, "PAST_1_HOUR"] = df[filter]["Close"].shift(-1)
        df.loc[filter, "PAST_2_HOUR"] = df[filter]["Close"].shift(-2)
        df.loc[filter, "PAST_3_HOUR"] = df[filter]["Close"].shift(-3)
        df.loc[filter, "PAST_4_HOUR"] = df[filter]["Close"].shift(-4)
        df.loc[filter, "PAST_5_HOUR"] = df[filter]["Close"].shift(-5)
        df.loc[filter, "PAST_10_HOUR"] = df[filter]["Close"].shift(-10)
        df.loc[filter, "PAST_24_HOUR"] = df[filter]["Close"].shift(-24)
        df.loc[filter, "PAST_36_HOUR"] = df[filter]["Close"].shift(-36)
        df.loc[filter, "PAST_48_HOUR"] = df[filter]["Close"].shift(-48)
        # upper, middle, lower = BBANDS(close_value, matype=MA_Type.T3)
        # df.loc[filter, "BBAND_UPPER"] = upper
        # df.loc[filter, "BBAND_MIDDLE"] = middle
        # df.loc[filter, "BBAND_LOWER"] = lower
        # df.loc[filter, "HT_TRENDLINE"] = HT_TRENDLINE(close_value)
        df.loc[filter, "KAMA_30"] = KAMA(close_value, timeperiod=30)

        # df.loc[filter, "DEMA_21"] = DEMA(close_value, timeperiod=21)
        # df.loc[filter, "DEMA_55"] = DEMA(close_value, timeperiod=55)

        df.loc[filter, "RSI_14"] = RSI(close_value, timeperiod=14)
        df.loc[filter, "EMA_8"] = EMA(close_value, timeperiod=8)
        df.loc[filter, "EMA_21"] = EMA(close_value, timeperiod=21)
        log.info(f"df Addded features {df.head().to_markdown()}")

    df = df.dropna(axis=0).reset_index(drop=True)
    df.index = df["Date"].factorize()[0]
    df["Buy/Sold/Hold"] = 0.0
    df = df.loc[df.groupby(level=0).count()["Date"] == len(TICKERS)]
    log.info(f"After adding features \n{df.head().to_markdown()}")

    train_size = df.index.values[-1] - int(
        df.index.values[-1] * TRAIN_TEST_SPLIT_PERCENT
    )
    train_df = df.loc[:train_size]
    trade_df = df.loc[train_size + 1 :]
    log.info(
        f"Splitting Dataframe to train and trade at {TRAIN_TEST_SPLIT_PERCENT*100}%"
    )
    log.info(f"Train Dataframe \n{train_df.tail().to_markdown()}")
    log.info(f"Trade Dataframe \n{trade_df.head().to_markdown()}")

    train_arrays = np.array(
        train_df[["Close"] + TECHNICAL_INDICATORS + ["Buy/Sold/Hold"]]
        .groupby(train_df.index)
        .apply(np.array)
        .values.tolist(),
        dtype=np.float32,
    )
    log.info(f"Train Array - {len(train_arrays)}")

    trade_arrays = np.array(
        trade_df[["Close"] + TECHNICAL_INDICATORS + ["Buy/Sold/Hold"]]
        .groupby(trade_df.index)
        .apply(np.array)
        .values.tolist(),
        dtype=np.float32,
    )
    log.info(f"Trade Array - {len(trade_arrays)}")

    with open(f"{STOCK_DATA_SAVE_DIR}/{NUMPY_FILENAME}", "wb") as f:
        np.save(f, train_arrays)
        np.save(f, trade_arrays)
    log.info(f"Arrays saved to - {STOCK_DATA_SAVE_DIR}/{NUMPY_FILENAME}")


if __name__ == "__main__":
    add_features()
