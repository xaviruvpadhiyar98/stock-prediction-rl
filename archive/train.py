from time import perf_counter
from typing import List, Tuple

from stock_trading_env import StockTradingEnv
from config import (
    STOCK_DATA_SAVE_DIR,
    TICKERS,
    SEED,
    TRAIN_TEST_SPLIT_PERCENT,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    FILENAME,
)
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from random import seed as random_seed
from torch import manual_seed
from torch.cuda import manual_seed as cuda_seed
from pathlib import Path
from logger_config import train_logger as log
import pandas as pd
import polars as pl
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from concurrent.futures import ProcessPoolExecutor

# TRY NOT TO MODIFY: seeding
random_seed(SEED)
np.random.seed(SEED)
manual_seed(SEED)
cuda_seed(SEED)


class TensorboardCallback(BaseCallback):
    def __init__(self, save_freq: int, model_prefix: str, eval_env: Monitor):
        self.save_freq = save_freq
        self.model_prefix = model_prefix
        self.eval_env = eval_env
        super().__init__()

    def _on_step(self) -> bool:
        self.logger.record(
            key="train/holdings",
            value=self.locals["infos"][0]["holdings"],
        )
        self.logger.record(key="train/reward", value=self.locals["rewards"][0])
        self.logger.record(key="train/n_calls", value=self.n_calls)
        shares = self.locals["infos"][0]["shares"]
        for k, v in shares.items():
            self.logger.record(key=f"train/shares/{k}", value=v)

        if (self.n_calls > 0) and (self.n_calls % self.save_freq) == 0:
            obs, _ = self.eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs)
                obs, reward, done, truncated, info = self.eval_env.step(action)

            model_path = Path(TRAINED_MODEL_DIR) / self.model_prefix
            available_model_files = list(model_path.rglob("*.zip"))
            available_model_holdings = [
                int(f.stem.split("-")[-1]) for f in available_model_files
            ]
            available_model_holdings.sort()
            trade_holdings = int(info["holdings"])
            self.logger.record(key="trade/holdings", value=trade_holdings)

            if not available_model_holdings:
                model_filename = model_path / f"{trade_holdings}.zip"
                self.model.save(model_filename)
                log.info(f"Saving model checkpoint to {model_filename}")

            # elif len(available_model_holdings) < 5:
            #     model_filename = model_path / f"{trade_holdings}.zip"
            #     self.model.save(model_filename)
            #     log.info(f"Saving model checkpoint to {model_filename}")

            else:
                if trade_holdings > available_model_holdings[0]:
                    file_to_remove = model_path / f"{available_model_holdings[0]}.zip"
                    file_to_remove.unlink()
                    model_filename = model_path / f"{trade_holdings}.zip"
                    self.model.save(model_filename)
                    log.info(
                        f"Removed {file_to_remove} and Added {model_filename} file."
                    )

            # periodic save model for continue training later
            model_filename = Path(TRAINED_MODEL_DIR) / "a2c_model.zip"
            self.model.save(model_filename)
        return True


def load_df() -> pl.DataFrame:
    parquet_filename = Path(STOCK_DATA_SAVE_DIR) / "random-778-tickers.parquet"
    df = pl.read_parquet(parquet_filename)[["Date", "Ticker", "Close"]]
    log.info(f"Data loaded successfully into Dataframe\n{df.head()}")
    return df


def add_features(
    df: pd.DataFrame, past_hours: Tuple[int]
) -> Tuple[pd.DataFrame, List[str]]:
    features_df = pl.DataFrame()
    for i, ticker in enumerate(TICKERS):
        feature_columns = []
        print(i, ticker)
        for hour in past_hours:
            feature_col = f"PAST_{abs(hour)}_HOUR"
            tmp_df = df.filter(pl.col("Ticker") == ticker).with_columns(
                pl.col("Close").shift(hour).alias(feature_col)
            )
            feature_columns.append(feature_col)
            features_df = pl.concat([features_df, tmp_df], how="diagonal")
            # print(tmp_df)
            # import sys

            # sys.exit()

        # filter = features_df["Ticker"] == ticker
        # for hour in past_hours:
        #     feature_col = f"PAST_{abs(hour)}_HOUR"
        #     df.with_columns((pl.col("a") ** 2).alias(feature_col))

        #     features_df.loc[filter, feature_col] = features_df[filter]["Close"].shift(
        #         hour
        #     )

    log.info(f"Addded features to Dataframe\n{features_df.head()}")
    return features_df, feature_columns


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_df = df.copy().dropna(axis=0).reset_index(drop=True)
    cleaned_df.index = cleaned_df["Date"].factorize()[0]
    cleaned_df["Buy/Sold/Hold"] = 0.0
    cleaned_df = cleaned_df.loc[
        cleaned_df.groupby(level=0).count()["Date"] == len(TICKERS)
    ]
    log.info(f"Cleaned Dataframe \n{cleaned_df.head().to_markdown()}")
    return cleaned_df


def split_train_test(
    df: pd.DataFrame, technical_indicators: List[str]
) -> Tuple[np.array, np.array]:
    train_size = df.index.values[-1] - int(
        df.index.values[-1] * TRAIN_TEST_SPLIT_PERCENT
    )
    train_df = df.loc[:train_size]
    trade_df = df.loc[train_size + 1 :]

    train_arrays = np.array(
        train_df[["Close"] + list(technical_indicators) + ["Buy/Sold/Hold"]]
        .groupby(train_df.index)
        .apply(np.array)
        .values.tolist(),
        dtype=np.float32,
    )
    trade_arrays = np.array(
        trade_df[["Close"] + list(technical_indicators) + ["Buy/Sold/Hold"]]
        .groupby(trade_df.index)
        .apply(np.array)
        .values.tolist(),
        dtype=np.float32,
    )
    log.info(f"Technical Indicators used {technical_indicators}")
    log.info(
        f"Successfully split the DataFrame into {len(train_arrays)} ({(1-TRAIN_TEST_SPLIT_PERCENT)*100}%) training data and {len(trade_arrays)} ({TRAIN_TEST_SPLIT_PERCENT*100}%) trading data."
    )
    return train_arrays, trade_arrays


def train(
    train_arrays: np.array,
    trade_arrays: np.array,
    feature_columns: List[str],
    identifier: str,
    total_timestamp: int,
):
    start_time = perf_counter()
    train_env = Monitor(StockTradingEnv(train_arrays, TICKERS, feature_columns))
    trade_env = Monitor(StockTradingEnv(trade_arrays, TICKERS, feature_columns))

    log.info(f"Identifier for saving and tensorboard logging- {identifier}")

    model_file = Path(TRAINED_MODEL_DIR) / "a2c_model.zip"
    if model_file.exists():
        model = A2C.load(
            Path(TRAINED_MODEL_DIR) / "a2c_model.zip",
            train_env,
            verbose=0,
            tensorboard_log=TENSORBOARD_LOG_DIR,
        )
        log.info("Existing checkpoint found, resuming...")
    else:
        model = A2C(
            "MlpPolicy", train_env, verbose=0, tensorboard_log=TENSORBOARD_LOG_DIR
        )
        log.info("Creating a new model")

    model.learn(
        total_timesteps=total_timestamp,
        callback=TensorboardCallback(
            save_freq=4096, model_prefix=identifier, eval_env=trade_env
        ),
        tb_log_name=identifier,
        reset_num_timesteps=False,
    )
    obs, _ = trade_env.reset()
    for i in range(len(trade_arrays)):
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = trade_env.step(action)
    log.info(f"Result of Trade env\n{info}")
    log.info(f"Training took {perf_counter() - start_time: .2f} seconds.")


def find_best_feature():
    model_files = Path(TENSORBOARD_LOG_DIR).rglob("events*")
    trade_holdings = []
    for file in model_files:
        ea = EventAccumulator(file.as_posix())
        ea.Reload()
        scalers = ea.Tags()["scalars"]
        if "trade/holdings" in scalers:
            trade_holdings.append(
                {"file": file, "trade_holdings": ea.Scalars("trade/holdings")[-1].value}
            )
        else:
            log.error(f"No Scalar value found in {file}")
    trade_holdings.sort(key=lambda x: x["trade_holdings"], reverse=True)
    log.info("Showing top 5 trade_holding_features")
    print(json.dumps(trade_holdings, indent=4, default=str))
    technical_indicators = trade_holdings[0]["file"].parent
    technical_indicators = technical_indicators.stem.split("-")[2:]
    technical_indicators[-1] = technical_indicators[-1].rsplit("_", 1)[0]
    return technical_indicators


def generate_past_hours_permutation(start: int, end: int) -> List[dict]:
    """
    Returns
        [
            [
                "PAST_6_HOUR",
                "PAST_5_HOUR"
            ],
            [
                "PAST_5_HOUR",
                "PAST_4_HOUR"
            ],
            ............
            [
                "PAST_5_HOUR",
                "PAST_4_HOUR",
                "PAST_3_HOUR",
                "PAST_2_HOUR",
                "PAST_1_HOUR"
            ],
            [
                "PAST_6_HOUR",
                "PAST_5_HOUR",
                "PAST_4_HOUR",
                "PAST_3_HOUR",
                "PAST_2_HOUR",
                "PAST_1_HOUR"
            ]
        ]
    """
    past_n_hours = []
    for prev_item, curr_item in permutations(range(start, end), 2):
        features = [f"PAST_{abs(i)}_HOUR" for i in (range(prev_item, curr_item + 1))]
        if features:
            past_n_hours.append(features)
    past_n_hours.sort(key=len)
    return past_n_hours


def generate_past_hours_permutation(start: int, end: int) -> List[dict]:
    s = range(start, end)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def filter_only_NS_df(df):
    return df.filter(df["Ticker"].str.ends_with(".NS"))


def run(i: int, past_hour, model_name, total_timestamp):
    df = load_df()
    df = filter_only_NS_df(df)
    log.info(f"{i}. Past Hour - {past_hour}")
    df, feature_columns = add_features(df, past_hour)
    print(df)
    print(feature_columns)
    # log.info(f"Starting with {past_hour} indicators and {feature_columns}")
    # df = clean_df(df)
    # train_arrays, trade_arrays = split_train_test(
    #     df, technical_indicators=feature_columns
    # )
    # identifier = f"{model_name}/{total_timestamp}/PAST_{'-'.join([str(abs(x)) for x in past_hour])}_HOUR"
    # train(train_arrays, trade_arrays, feature_columns, identifier, total_timestamp)


def main():
    model_name = "a2c"
    total_timestamp = 1_000_000
    past_hour = tuple(range(-50, 0, 1))
    run(0, past_hour, model_name, total_timestamp)
    # past_hours = generate_past_hours_permutation(-6, 0)
    # print(list(past_hours))
    # past_hours = [()]
    # with ProcessPoolExecutor(max_workers=2) as e:
    #     [
    #         e.submit(run, i, past_hour, model_name, total_timestamp)
    #         for i, past_hour in enumerate(past_hours)
    #     ]
    # for i, past_hour in enumerate(past_hours):
    #     e.submit()

    # if i == 10:
    #     break


if __name__ == "__main__":
    main()
