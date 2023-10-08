import numpy as np
import polars as pl
import yfinance as yf
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO, A2C
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gymnasium.wrappers.normalize import NormalizeReward
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold
from envs.stock_trading_env import StockTradingEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
import psutil
from subprocess import run, PIPE
import torch
from talib import (
    RSI,
    EMA,
    MACD,
    STOCH,
    ADX,
    BBANDS,
    ROC,
    ATR,
    CCI,
)
import json
from stable_baselines3.common.utils import set_random_seed
from optuna import Trial, create_study, create_trial
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.exceptions import TrialPruned
from typing import Any
from typing import Dict
import torch.nn as nn
import random


TICKERS = "SBIN.NS"
INTERVAL = "1h"
PERIOD = "360d"
MODEL = "PPO"
MODEL_PREFIX = f"{TICKERS}_{MODEL}"


NUM_ENVS = 16
N_STEPS = 512
TIME_STAMPS = 8 * 4

N_STARTUP_TRIALS = 100000
N_TRIALS = 50


TRAIN_TEST_SPLIT_PERCENT = 0.15
PAST_HOURS = range(1, 15)
TECHNICAL_INDICATORS = [f"PAST_{hour}_HOUR" for hour in PAST_HOURS]
DATASET = Path("datasets")
DATASET.mkdir(parents=True, exist_ok=True)
TRAINED_MODEL_DIR = Path("trained_models")
TRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_LOG_DIR = Path("tensorboard_log")
TENSORBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)


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
    df = pl.read_parquet(ticker_file).select(["Datetime", "Close", "High", "Low"])
    df = df.with_columns(pl.lit(TICKERS).alias("Ticker"))
    df = df.sort("Datetime", descending=False)
    return df


def add_technical_indicators(df):
    """
    ┌─────────────────────────┬────────────┬─────────┬───────────┐
    │ Datetime                ┆ Close      ┆ Ticker  ┆ RSI       │
    │ ---                     ┆ ---        ┆ ---     ┆ ---       │
    │ datetime[ns, UTC]       ┆ f64        ┆ str     ┆ f64       │
    ╞═════════════════════════╪════════════╪═════════╪═══════════╡
    │ 2022-04-07 03:45:00 UTC ┆ 512.25     ┆ SBIN.NS ┆ NaN       │
    │ 2022-04-07 04:45:00 UTC ┆ 514.599976 ┆ SBIN.NS ┆ NaN       │
    │ 2022-04-07 05:45:00 UTC ┆ 519.25     ┆ SBIN.NS ┆ NaN       │
    │ 2022-04-07 06:45:00 UTC ┆ 519.0      ┆ SBIN.NS ┆ NaN       │
    │ …                       ┆ …          ┆ …       ┆ …         │
    │ 2023-09-18 04:45:00 UTC ┆ 606.599976 ┆ SBIN.NS ┆ 75.303253 │
    │ 2023-09-18 05:45:00 UTC ┆ 605.400024 ┆ SBIN.NS ┆ 71.233586 │
    │ 2023-09-18 06:45:00 UTC ┆ 605.5      ┆ SBIN.NS ┆ 71.372404 │
    │ 2023-09-18 07:45:00 UTC ┆ 605.849976 ┆ SBIN.NS ┆ 71.8839   │
    └─────────────────────────┴────────────┴─────────┴───────────┘
    """
    df = df.with_columns(
        pl.lit(RSI(df.select("Close").to_series(), timeperiod=14)).alias("RSI"),
        pl.lit(EMA(df.select("Close").to_series(), timeperiod=9)).alias("EMA9"),
        pl.lit(EMA(df.select("Close").to_series(), timeperiod=21)).alias("EMA21"),
        pl.lit(
            MACD(
                df.select("Close").to_series(),
                fastperiod=12,
                slowperiod=26,
                signalperiod=9,
            )[0]
        ).alias("MACD_Line"),
        pl.lit(
            MACD(
                df.select("Close").to_series(),
                fastperiod=12,
                slowperiod=26,
                signalperiod=9,
            )[1]
        ).alias("Signal_Line"),
        pl.lit(
            MACD(
                df.select("Close").to_series(),
                fastperiod=12,
                slowperiod=26,
                signalperiod=9,
            )[2]
        ).alias("MACD_Histogram"),
        pl.lit(
            ADX(
                df.select("High").to_series(),
                df.select("Low").to_series(),
                df.select("Close").to_series(),
                timeperiod=14,
            )
        ).alias("ADX"),
        pl.lit(BBANDS(df.select("Close").to_series(), timeperiod=20)[0]).alias(
            "Upper_BollingerBand"
        ),
        pl.lit(BBANDS(df.select("Close").to_series(), timeperiod=20)[1]).alias(
            "Middle_BollingerBand"
        ),
        pl.lit(BBANDS(df.select("Close").to_series(), timeperiod=20)[2]).alias(
            "Lower_BollingerBand"
        ),
        pl.lit(ROC(df.select("Close").to_series(), timeperiod=10)).alias("ROC"),
        pl.lit(
            ATR(
                df.select("High").to_series(),
                df.select("Low").to_series(),
                df.select("Close").to_series(),
                timeperiod=14,
            )
        ).alias("ATR"),
        pl.lit(
            CCI(
                df.select("High").to_series(),
                df.select("Low").to_series(),
                df.select("Close").to_series(),
                timeperiod=14,
            )
        ).alias("CCI"),
    )
    df = df.drop_nulls()
    df = pl.from_pandas(df.to_pandas().dropna())
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


def create_torch_array(arr, device="cuda:0"):
    arr = np.asarray(arr).astype(np.float32)
    return torch.from_numpy(arr).to(device)


def make_env(env_id, array, tickers, use_tensor, seed, rank):
    def thunk():
        env = Monitor(env_id(array, [tickers], use_tensor))
        env.reset(seed=seed+rank)
        return env

    return thunk


def get_train_trade_environment(
    framework="sb", tickers=[TICKERS], num_envs=NUM_ENVS, seed=SEED
):
    df = load_data()
    df = add_past_hours(df)
    df = add_technical_indicators(df)
    df = df.with_columns(pl.lit(0.0).alias("Buy/Sold/Hold"))
    train_df, trade_df = train_test_split(df)

    assert train_df.columns == trade_df.columns

    train_arrays = create_numpy_array(train_df)
    trade_arrays = create_numpy_array(trade_df)

    if framework == "sb":
        train_envs = DummyVecEnv(
            [
                make_env(StockTradingEnv, train_arrays, tickers, False, seed, i)
                for i in range(num_envs)
            ]
        )
        trade_env = Monitor(StockTradingEnv(trade_arrays, tickers))
        check_env(trade_env)
    elif framework == "cleanrl":
        train_arrays = create_torch_array(train_arrays)
        trade_arrays = create_torch_array(trade_arrays)

    return train_envs, trade_env


def test_model(env, model, seed):
    obs, _ = env.reset(seed=seed)
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            env.close()
            return info


class TensorboardCallback(BaseCallback):
    """ """

    def __init__(self, eval_env: Monitor, model_path: Path, seed: int):
        self.eval_env = eval_env
        self.seed = seed
        self.model_path = model_path
        super().__init__()

    def log(self, info, key):
        unnecessary_keys = ["TimeLimit.truncated", "terminal_observation", "episode"]
        for k, v in info.items():
            self.logger.record(f"{key}/{k}", v, unnecessary_keys)

    def log_gpu(self):
        gpu_query = "utilization.gpu,utilization.memory"
        format = "csv,noheader,nounits"
        gpu_util, gpu_memory = run(
            [
                "nvidia-smi",
                f"--query-gpu={gpu_query}",
                f"--format={format}",
            ],
            encoding="utf-8",
            stdout=PIPE,
            stderr=PIPE,
            check=True,
        ).stdout.split(",")
        info = {
            "utilization": float(gpu_util.strip()),
            "memory": float(gpu_memory.strip()),
        }
        self.log(info, "gpu")

    def log_cpu(self):
        cpu_percent = psutil.cpu_percent()
        memory_usage_percent = psutil.virtual_memory().percent
        info = {
            "utilization": cpu_percent,
            "memory": memory_usage_percent,
        }
        self.log(info, "cpu")

    def _on_step(self) -> bool:

        if (self.n_calls % 10) == 0:

            # find ending environments
            infos = self.locals["infos"]
            end_envs = {
                i: info["cummulative_profit_loss"]
                for i, info in enumerate(infos)
                if "episode" in info 
            }
            if not end_envs:
                return True

            self.log_gpu()
            self.log_cpu()


            sorted_env = sorted(end_envs, reverse=True)
            best_env_id = sorted_env[0]
            best_env_info = infos[best_env_id]
            best_env_info["env_id"] = best_env_id
            best_env_info["env"] = "train"


            self.log(best_env_info, key="train")
            Path("sb_best_env.json").write_text(json.dumps(best_env_info))
            # print(json.dumps(best_env_info, indent=4, default=str))
            t_info = test_model(self.eval_env, self.model, best_env_id)
            t_info["env"] = "trade"
            self.log(t_info, key="trade")
            # print(json.dumps(t_info, indent=4, default=float))
            if t_info["cummulative_profit_loss"] > 500:
                self.model.save(self.model_path.parent / f"{t_info['cummulative_profit_loss']}.zip")

        return True

            # raise

            # print(json.dumps(self.locals, indent=4, default=str))


            # if "episode" in info:
            #     self.log(info, key="train")

            #     t_info = test_model(self.eval_env, self.model, self.seed)
            #     print(json.dumps(t_info, indent=4, default=float))
            #     if t_info["cummulative_profit_loss"] > 500:
            #         self.model.save(self.model_path.parent / f"{t_info['cummulative_profit_loss']}.zip")
            #     # self.model.save(self.model_path)
            # return True

        # if (self.n_calls > 0) and (self.n_calls % self.save_freq) == 0:
        #     trade_model = PPO.load(Path(TRAINED_MODEL_DIR) / f"{MODEL_PREFIX}.zip")
        #     obs, info = self.eval_env.reset(seed=SEED)
        #     while True:
        #         action, _ = trade_model.predict(obs, deterministic=True)
        #         obs, reward, done, truncated, info = self.eval_env.step(action)
        #         if done or truncated:
        #             break
        #     self.log(info, key="trade")
        #     print("Eval Data Result - ", info)

        #     trade_holdings = int(info["cummulative_profit_loss"])
        #     if trade_holdings < 0:
        #         return True

        #     model_path = Path(TRAINED_MODEL_DIR) / self.model_prefix
        #     available_model_files = list(model_path.rglob("*.zip"))
        #     if not available_model_files:
        #         return True

        #     available_model_holdings = [
        #         int(f.stem.split("-")[-1]) for f in available_model_files
        #     ]
        #     available_model_holdings.sort()

        #     if (not available_model_holdings) and (trade_holdings > 0):
        #         model_filename = model_path / f"{trade_holdings}.zip"
        #         self.model.save(model_filename)
        #         print(f"Saving model checkpoint to {model_filename}")
        #         return True

        #     if trade_holdings > available_model_holdings[0]:
        #         file_to_remove = model_path / f"{available_model_holdings[0]}.zip"
        #         file_to_remove.unlink()
        #         model_filename = model_path / f"{trade_holdings}.zip"
        #         self.model.save(model_filename)
        #         print(f"Removed {file_to_remove} and Added {model_filename} file.")
        #         return True
        # return True





class OptunaCallback(BaseCallback):
    """ """

    def __init__(self, eval_env: Monitor, num_envs: int):
        self.eval_env = eval_env
        self.num_envs = num_envs
        super().__init__()

    def _on_step(self) -> bool:
        """
        self.locals - 
        dict_keys(['self', 'total_timesteps', 'callback', 'log_interval', 'tb_log_name', 'reset_num_timesteps', 'progress_bar', 'iteration', 'env', 'rollout_buffer', 'n_rollout_steps', 'n_steps', 'obs_tensor', 'actions', 'values', 'log_probs', 'clipped_actions', 'new_obs', 'rewards', 'dones', 'infos'])
        """


        if (self.n_calls * self.num_envs == self.locals["total_timesteps"]):


            # find ending environments
            infos = self.locals["infos"]
            end_envs = {
                i: info["cummulative_profit_loss"]
                for i, info in enumerate(infos)
                if "episode" in info 
            }

            if not end_envs:
                return True

            sorted_env = sorted(end_envs, reverse=True)
            best_env_id = sorted_env[0]
            best_env_info = infos[best_env_id]
            best_env_info["env_id"] = best_env_id
            best_env_info["env"] = "train"

        
            Path("sb_best_env.json").write_text(json.dumps(best_env_info))
            t_info = test_model(self.eval_env, self.model, best_env_id)
            t_info["env"] = "trade"
            print(json.dumps(t_info, indent=4, default=float))
        
        return True


def get_ppo_model(env, n_steps, seed):
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=5e-4,
        n_steps=n_steps,
        batch_size=64,
        n_epochs=16,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.3,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.05,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        stats_window_size=100,
        tensorboard_log=TENSORBOARD_LOG_DIR,
        policy_kwargs=dict(
            net_arch=[256, 256],
        ),
        verbose=0,
        seed=seed,
        device="auto",
        _init_setup_model=True,
    )
    return model


def get_best_ppo_model(env, seed):
    """
    [I 2023-10-04 16:41:00,893] Trial 6 finished with value: 42.300048828125 and parameters: {'batch_size': 128, 'n_steps': 512, 'gamma': 0.95, 'learning_rate': 1.9341219418904578e-05, 'lr_schedule': 'constant', 'ent_coef': 1.1875984002464866e-06, 'clip_range': 0.2, 'n_epochs': 20, 'gae_lambda': 1.0, 'max_grad_norm': 2, 'vf_coef': 0.029644396080155226, 'net_arch': 'small', 'ortho_init': True, 'activation_fn': 'relu'}. Best is trial 6 with value: 42.300048828125.
    """

    model = PPO(
        "MlpPolicy",
        env,
        # learning_rate=linear_schedule(9.2458929157504e-05),
        learning_rate=3.7141262285419446e-05,
        n_steps=512,
        batch_size=128,
        n_epochs=20,
        gamma=0.95,
        gae_lambda=1.0,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=1.1875984002464866e-06,
        vf_coef=0.029644396080155226,
        max_grad_norm=0.8,
        tensorboard_log=TENSORBOARD_LOG_DIR,
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            activation_fn=nn.Tanh,
            ortho_init=True,
        ),
        verbose=0,
        seed=seed,
        device="auto",
        _init_setup_model=True,
    )
    return model


def get_a2c_model(env, n_steps, seed):
    model = A2C(
        "MlpPolicy",
        env,
        # learning_rate=5e-4,
        # n_steps=n_steps,
        # batch_size=64,
        # n_epochs=16,
        # gamma=0.99,
        # gae_lambda=0.95,
        # clip_range=0.3,
        # clip_range_vf=None,
        # normalize_advantage=True,
        ent_coef=0.05,
        # vf_coef=0.5,
        # max_grad_norm=0.5,
        # use_sde=False,
        # sde_sample_freq=-1,
        # target_kl=None,
        # stats_window_size=100,
        tensorboard_log=TENSORBOARD_LOG_DIR,
        # policy_kwargs=dict(
        #     net_arch=[256, 256],
        # ),
        verbose=0,
        seed=seed,
        device="auto",
        _init_setup_model=True,
    )
    return model


def load_ppo_model(env=None):
    model_file = Path(TRAINED_MODEL_DIR) / f"{MODEL_PREFIX}.zip"
    model = PPO.load(
        model_file,
        env,
        verbose=0,
        tensorboard_log=TENSORBOARD_LOG_DIR,
        print_system_info=False,
    )
    return model


def sample_ppo_params(trial: Trial) -> dict[str, Any]:
    """
    Sampler for PPO hyperparams.

    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    n_steps = trial.suggest_categorical(
        "n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    )
    gamma = trial.suggest_categorical(
        "gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
    )
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    lr_schedule = "constant"

    # Uncomment to enable learning rate schedule
    lr_schedule = trial.suggest_categorical("lr_schedule", ["linear", "constant"])
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    gae_lambda = trial.suggest_categorical(
        "gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0]
    )
    max_grad_norm = trial.suggest_categorical(
        "max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5]
    )
    vf_coef = trial.suggest_float("vf_coef", 0.01, 1, log=True)
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])

    # Uncomment for gSDE (continuous actions)
    # log_std_init = trial.suggest_uniform("log_std_init", -4, 1)
    # Uncomment for gSDE (continuous action)
    # sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])

    # Orthogonal initialization
    # ortho_init = False
    ortho_init = trial.suggest_categorical("ortho_init", [False, True])
    activation_fn = trial.suggest_categorical(
        "activation_fn", ["tanh", "relu", "elu", "leaky_relu"]
    )
    # activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    # TODO: account when using multiple envs
    if batch_size > n_steps:
        batch_size = n_steps

    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)

    # Independent networks usually work best
    # when not working with images
    net_arch = {
        "small": dict(pi=[64, 64], vf=[64, 64]),
        "medium": dict(pi=[256, 256], vf=[256, 256]),
    }[net_arch]

    activation_fn = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }[activation_fn]

    return {
        "policy": "MlpPolicy",
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        # "sde_sample_freq": sde_sample_freq,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
    }


def sample_a2c_params(trial: Trial) -> Dict[str, Any]:
    """Sampler for A2C hyperparameters."""
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
    gae_lambda = 1.0 - trial.suggest_float("gae_lambda", 0.001, 0.2, log=True)
    n_steps = 2 ** trial.suggest_int("exponent_n_steps", 3, 10)
    learning_rate = trial.suggest_float("lr", 1e-5, 1, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    ortho_init = trial.suggest_categorical("ortho_init", [False, True])
    net_arch = trial.suggest_categorical("net_arch", ["tiny", "small"])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    # Display true values.
    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("gae_lambda_", gae_lambda)
    trial.set_user_attr("n_steps", n_steps)

    net_arch = (
        {"pi": [64], "vf": [64]}
        if net_arch == "tiny"
        else {"pi": [64, 64], "vf": [64, 64]}
    )

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "max_grad_norm": max_grad_norm,
        "policy_kwargs": {
            "net_arch": net_arch,
            "activation_fn": activation_fn,
            "ortho_init": ortho_init,
        },
    }


def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func
