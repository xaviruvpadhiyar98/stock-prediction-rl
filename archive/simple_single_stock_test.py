import numpy as np
import polars as pl
import yfinance as yf
from pathlib import Path
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from envs.stock_trading_env_using_numpy import StockTradingEnv
from gymnasium.wrappers.normalize import NormalizeReward
import random
import torch
from gymnasium.vector import SyncVectorEnv 

TICKERS = "SBIN.NS"
INTERVAL = "1h"
PERIOD = "360d"
MODEL_PREFIX = f"{TICKERS}_PPO"
NUM_ENVS = 10
# MODEL_PREFIX = f"{TICKERS}_A2C"
# MODEL_PREFIX = f"{TICKERS}_SAC"
# MODEL_PREFIX = f"{TICKERS}_DDPG"
# MODEL_PREFIX = f"{TICKERS}_TD3"


TRAIN_TEST_SPLIT_PERCENT = 0.15
PAST_HOURS = range(1, 15)
TECHNICAL_INDICATORS = [f"PAST_{hour}_HOUR" for hour in PAST_HOURS]
DATASET = Path("datasets")
TRAINED_MODEL_DIR = Path("trained_models")
TENSORBOARD_LOG_DIR = Path("tensorboard_log")

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def load_data():
    """returns following dataframe
    shape: (2_508, 3)
    ┌─────────────────────────┬────────────┬─────────┐
    │ Datetime                ┆ Close      ┆ Ticker  │
    │ ---                     ┆ ---        ┆ ---     │
    │ datetime[ns, UTC]       ┆ f64        ┆ str     │
    ╞═════════════════════════╪════════════╪═════════╡
    │ 2022-03-15 03:45:00 UTC ┆ 485.5      ┆ SBIN.NS │
    │ 2022-03-15 04:45:00 UTC ┆ 486.700012 ┆ SBIN.NS │
    │ 2022-03-15 05:45:00 UTC ┆ 488.549988 ┆ SBIN.NS │
    │ 2022-03-15 06:45:00 UTC ┆ 485.049988 ┆ SBIN.NS │
    │ …                       ┆ …          ┆ …       │
    │ 2023-08-25 06:45:00 UTC ┆ 571.099976 ┆ SBIN.NS │
    │ 2023-08-25 07:45:00 UTC ┆ 570.799988 ┆ SBIN.NS │
    │ 2023-08-25 08:45:00 UTC ┆ 569.799988 ┆ SBIN.NS │
    │ 2023-08-25 09:45:00 UTC ┆ 569.700012 ┆ SBIN.NS │
    └─────────────────────────┴────────────┴─────────┘
    """
    ticker_file = DATASET / TICKERS
    if not ticker_file.exists():
        yf.download(
            tickers=TICKERS,
            period=PERIOD,
            interval=INTERVAL,
            group_by="Ticker",
            auto_adjust=True,
            prepost=True,
        ).reset_index().to_parquet(ticker_file, index=False, engine="fastparquet")
    df = pl.read_parquet(ticker_file).select(["Datetime", "Close"])
    df = df.with_columns(pl.lit(TICKERS).alias("Ticker"))
    df = df.sort("Datetime", descending=False)
    return df


def add_past_hours(df):
    """
    shape: (2_494, 17)
    ┌─────────────┬────────────┬─────────┬─────────────┬───┬─────────────┬─────────────┬─────────────┬────────────┐
    │ Datetime    ┆ Close      ┆ Ticker  ┆ PAST_1_HOUR ┆ … ┆ PAST_11_HOU ┆ PAST_12_HOU ┆ PAST_13_HOU ┆ PAST_14_HO │
    │ ---         ┆ ---        ┆ ---     ┆ ---         ┆   ┆ R           ┆ R           ┆ R           ┆ UR         │
    │ datetime[ns ┆ f64        ┆ str     ┆ f64         ┆   ┆ ---         ┆ ---         ┆ ---         ┆ ---        │
    │ , UTC]      ┆            ┆         ┆             ┆   ┆ f64         ┆ f64         ┆ f64         ┆ f64        │
    ╞═════════════╪════════════╪═════════╪═════════════╪═══╪═════════════╪═════════════╪═════════════╪════════════╡
    │ 2022-03-17  ┆ 500.799988 ┆ SBIN.NS ┆ 491.75      ┆ … ┆ 485.049988  ┆ 488.549988  ┆ 486.700012  ┆ 485.5      │
    │ 03:45:00    ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ UTC         ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ 2022-03-17  ┆ 501.450012 ┆ SBIN.NS ┆ 500.799988  ┆ … ┆ 482.950012  ┆ 485.049988  ┆ 488.549988  ┆ 486.700012 │
    │ 04:45:00    ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ UTC         ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ 2022-03-17  ┆ 502.100006 ┆ SBIN.NS ┆ 501.450012  ┆ … ┆ 486.049988  ┆ 482.950012  ┆ 485.049988  ┆ 488.549988 │
    │ 05:45:00    ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ UTC         ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ 2022-03-17  ┆ 501.799988 ┆ SBIN.NS ┆ 502.100006  ┆ … ┆ 485.100006  ┆ 486.049988  ┆ 482.950012  ┆ 485.049988 │
    │ 06:45:00    ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ UTC         ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ …           ┆ …          ┆ …       ┆ …           ┆ … ┆ …           ┆ …           ┆ …           ┆ …          │
    │ 2023-08-25  ┆ 571.099976 ┆ SBIN.NS ┆ 571.549988  ┆ … ┆ 576.900024  ┆ 576.849976  ┆ 577.75      ┆ 573.700012 │
    │ 06:45:00    ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ UTC         ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ 2023-08-25  ┆ 570.799988 ┆ SBIN.NS ┆ 571.099976  ┆ … ┆ 580.700012  ┆ 576.900024  ┆ 576.849976  ┆ 577.75     │
    │ 07:45:00    ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ UTC         ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ 2023-08-25  ┆ 569.799988 ┆ SBIN.NS ┆ 570.799988  ┆ … ┆ 577.900024  ┆ 580.700012  ┆ 576.900024  ┆ 576.849976 │
    │ 08:45:00    ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ UTC         ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ 2023-08-25  ┆ 569.700012 ┆ SBIN.NS ┆ 569.799988  ┆ … ┆ 576.700012  ┆ 577.900024  ┆ 580.700012  ┆ 576.900024 │
    │ 09:45:00    ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ UTC         ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    └─────────────┴────────────┴─────────┴─────────────┴───┴─────────────┴─────────────┴─────────────┴────────────┘
    """  # noqa: E501
    df = df.with_columns(
        [pl.col("Close").shift(hour).alias(f"PAST_{hour}_HOUR") for hour in PAST_HOURS]
    )
    df = df.drop_nulls()
    return df


def train_test_split(df):
    """
    train_df ->
    ┌──────────────┬────────────┬─────────┬─────────────┬───┬──────────────┬──────────────┬──────────────┬──────────────┐
    │ Datetime     ┆ Close      ┆ Ticker  ┆ PAST_1_HOUR ┆ … ┆ PAST_12_HOUR ┆ PAST_13_HOUR ┆ PAST_14_HOUR ┆ Buy/Sold/Hol │
    │ ---          ┆ ---        ┆ ---     ┆ ---         ┆   ┆ ---          ┆ ---          ┆ ---          ┆ d            │
    │ datetime[ns, ┆ f64        ┆ str     ┆ f64         ┆   ┆ f64          ┆ f64          ┆ f64          ┆ ---          │
    │ UTC]         ┆            ┆         ┆             ┆   ┆              ┆              ┆              ┆ f64          │
    ╞══════════════╪════════════╪═════════╪═════════════╪═══╪══════════════╪══════════════╪══════════════╪══════════════╡
    │ 2023-06-07   ┆ 588.700012 ┆ SBIN.NS ┆ 589.099976  ┆ … ┆ 583.400024   ┆ 585.5        ┆ 587.0        ┆ 0.0          │
    │ 09:45:00 UTC ┆            ┆         ┆             ┆   ┆              ┆              ┆              ┆              │
    │ 2023-06-08   ┆ 592.700012 ┆ SBIN.NS ┆ 588.700012  ┆ … ┆ 584.099976   ┆ 583.400024   ┆ 585.5        ┆ 0.0          │
    │ 03:45:00 UTC ┆            ┆         ┆             ┆   ┆              ┆              ┆              ┆              │
    │ 2023-06-08   ┆ 593.25     ┆ SBIN.NS ┆ 592.700012  ┆ … ┆ 584.400024   ┆ 584.099976   ┆ 583.400024   ┆ 0.0          │
    │ 04:45:00 UTC ┆            ┆         ┆             ┆   ┆              ┆              ┆              ┆              │
    │ 2023-06-08   ┆ 589.900024 ┆ SBIN.NS ┆ 593.25      ┆ … ┆ 584.549988   ┆ 584.400024   ┆ 584.099976   ┆ 0.0          │
    │ 05:45:00 UTC ┆            ┆         ┆             ┆   ┆              ┆              ┆              ┆              │
    │ 2023-06-08   ┆ 590.900024 ┆ SBIN.NS ┆ 589.900024  ┆ … ┆ 586.099976   ┆ 584.549988   ┆ 584.400024   ┆ 0.0          │
    │ 06:45:00 UTC ┆            ┆         ┆             ┆   ┆              ┆              ┆              ┆              │
    └──────────────┴────────────┴─────────┴─────────────┴───┴──────────────┴──────────────┴──────────────┴──────────────┘


    trade_df ->
    ┌──────────────┬────────────┬─────────┬─────────────┬───┬──────────────┬──────────────┬──────────────┬──────────────┐
    │ Datetime     ┆ Close      ┆ Ticker  ┆ PAST_1_HOUR ┆ … ┆ PAST_12_HOUR ┆ PAST_13_HOUR ┆ PAST_14_HOUR ┆ Buy/Sold/Hol │
    │ ---          ┆ ---        ┆ ---     ┆ ---         ┆   ┆ ---          ┆ ---          ┆ ---          ┆ d            │
    │ datetime[ns, ┆ f64        ┆ str     ┆ f64         ┆   ┆ f64          ┆ f64          ┆ f64          ┆ ---          │
    │ UTC]         ┆            ┆         ┆             ┆   ┆              ┆              ┆              ┆ f64          │
    ╞══════════════╪════════════╪═════════╪═════════════╪═══╪══════════════╪══════════════╪══════════════╪══════════════╡
    │ 2023-06-08   ┆ 593.099976 ┆ SBIN.NS ┆ 590.900024  ┆ … ┆ 585.299988   ┆ 586.099976   ┆ 584.549988   ┆ 0.0          │
    │ 07:45:00 UTC ┆            ┆         ┆             ┆   ┆              ┆              ┆              ┆              │
    │ 2023-06-08   ┆ 588.549988 ┆ SBIN.NS ┆ 593.099976  ┆ … ┆ 587.200012   ┆ 585.299988   ┆ 586.099976   ┆ 0.0          │
    │ 08:45:00 UTC ┆            ┆         ┆             ┆   ┆              ┆              ┆              ┆              │
    │ 2023-06-08   ┆ 588.5      ┆ SBIN.NS ┆ 588.549988  ┆ … ┆ 588.75       ┆ 587.200012   ┆ 585.299988   ┆ 0.0          │
    │ 09:45:00 UTC ┆            ┆         ┆             ┆   ┆              ┆              ┆              ┆              │
    │ 2023-06-09   ┆ 584.700012 ┆ SBIN.NS ┆ 588.5       ┆ … ┆ 588.0        ┆ 588.75       ┆ 587.200012   ┆ 0.0          │
    │ 03:45:00 UTC ┆            ┆         ┆             ┆   ┆              ┆              ┆              ┆              │
    │ 2023-06-09   ┆ 580.900024 ┆ SBIN.NS ┆ 584.700012  ┆ … ┆ 590.0        ┆ 588.0        ┆ 588.75       ┆ 0.0          │
    │ 04:45:00 UTC ┆            ┆         ┆             ┆   ┆              ┆              ┆              ┆              │
    └──────────────┴────────────┴─────────┴─────────────┴───┴──────────────┴──────────────┴──────────────┴──────────────┘
    """  # noqa: E501
    total = df.shape[0]
    train_size = total - int(total * TRAIN_TEST_SPLIT_PERCENT)
    train_df = df.slice(0, train_size)
    trade_df = df.slice(train_size, total)
    return train_df, trade_df


def create_numpy_array(df):
    """
    returns array
    [
        [593.09997559 590.90002441 589.90002441 ... 586.09997559 584.54998779
            0.        ]
        [588.54998779 593.09997559 590.90002441 ... 585.29998779 586.09997559
            0.        ]
        [588.5        588.54998779 593.09997559 ... 587.20001221 585.29998779
            0.        ]
        ...
        [570.79998779 571.09997559 571.54998779 ... 576.84997559 577.75
            0.        ]
        [569.79998779 570.79998779 571.09997559 ... 576.90002441 576.84997559
            0.        ]
        [569.70001221 569.79998779 570.79998779 ... 580.70001221 576.90002441
            0.        ]
    ]
    """
    cols = df.columns
    cols.remove("Datetime")
    cols.remove("Ticker")
    df = df.unique("Datetime", maintain_order=True)
    df = df.select(cols)
    arr = []
    [arr.append(row) for row in df.iter_rows()]
    return np.asarray(arr)


def test_model(env, model, n_times=1):
    for _ in range(n_times):
        obs, _ = env.reset()
        while True:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            # print(info)
            if done:
                print(info)
                break
    return info

def resume_model_ppo(env):
    model_file = Path(TRAINED_MODEL_DIR) / f"{MODEL_PREFIX}.zip"
    if model_file.exists():
        model = PPO.load(
            model_file,
            env,
            verbose=0,
            tensorboard_log=TENSORBOARD_LOG_DIR,
            seed=SEED
        )
        return model
    raise


def main():
    df = load_data()
    df = add_past_hours(df)
    df = df.with_columns(pl.lit(0.0).alias("Buy/Sold/Hold"))
    train_df, trade_df = train_test_split(df)

    assert train_df.columns == trade_df.columns

    train_arrays = create_numpy_array(train_df)
    trade_arrays = create_numpy_array(trade_df)

    train_env = Monitor(StockTradingEnv(train_arrays, [TICKERS]))
    trade_env = Monitor(StockTradingEnv(trade_arrays, [TICKERS]))

    model = resume_model_ppo(train_env)
    test_model(trade_env, model, 1)


if __name__ == "__main__":
    main()
