from typing import List, Tuple
from stock_trading_env import StockTradingEnv
from config import (
    FILENAME,
    NUMPY_FILENAME,
    STOCK_DATA_SAVE_DIR,
    TICKERS,
    TECHNICAL_INDICATORS,
    SEED,
    TRAIN_TEST_SPLIT_PERCENT,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
)
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from random import seed as random_seed
from torch import manual_seed
from pathlib import Path
from logger_config import benchmark_logger as log
import pandas as pd
from itertools import permutations
import re
from concurrent.futures import as_completed, ProcessPoolExecutor
from time import perf_counter

# TRY NOT TO MODIFY: seeding
random_seed(SEED)
np.random.seed(SEED)
manual_seed(SEED)
TECHNICAL_INDICATORS = []






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
        return True


def load_df() -> pd.DataFrame:
    parquet_filename = Path(STOCK_DATA_SAVE_DIR) / FILENAME
    df = pd.read_parquet(parquet_filename, engine="fastparquet")[["Date", "Ticker", "Close"]]
    log.info(f"Data loaded successfully into Dataframe\n{df.head().to_markdown()}")
    return df


def add_features(df: pd.DataFrame, experiment:dict) -> pd.DataFrame:
    if not experiment["technical_indicators"]:
        log.info(f"No features to add to Dataframe\n{df.head().to_markdown()}")
        return df
    features_df = df.copy()
    for ticker in TICKERS:
        close_value = features_df[features_df["Ticker"] == ticker]["Close"].values
        filter = features_df["Ticker"] == ticker
        for technical_indicator in experiment["technical_indicators"]:
            period = re.search(r"\d+", technical_indicator).group()
            features_df.loc[filter, technical_indicator] = features_df[filter]["Close"].shift(-int(period))

    log.info(f"Addded features to Dataframe\n{features_df.head().to_markdown()}")
    return features_df

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_df = df.copy().dropna(axis=0).reset_index(drop=True)
    cleaned_df.index = cleaned_df["Date"].factorize()[0]
    cleaned_df["Buy/Sold/Hold"] = 0.0
    cleaned_df = cleaned_df.loc[cleaned_df.groupby(level=0).count()["Date"] == len(TICKERS)]
    log.info(f"Cleaned Dataframe \n{cleaned_df.head().to_markdown()}")
    return cleaned_df


def split_train_test(df: pd.DataFrame, technical_indicators: List[str]) -> Tuple[np.array, np.array]:
    train_size = df.index.values[-1] - int(
        df.index.values[-1] * TRAIN_TEST_SPLIT_PERCENT
    )
    train_df = df.loc[:train_size]
    trade_df = df.loc[train_size + 1 :]


    train_arrays = np.array(
        train_df[["Close"] + technical_indicators + ["Buy/Sold/Hold"]]
        .groupby(train_df.index)
        .apply(np.array)
        .values.tolist(),
        dtype=np.float32,
    )
    trade_arrays = np.array(
        trade_df[["Close"] + technical_indicators + ["Buy/Sold/Hold"]]
        .groupby(trade_df.index)
        .apply(np.array)
        .values.tolist(),
        dtype=np.float32,
    )
    log.info(f"Successfully split the DataFrame into {len(train_arrays)} ({(1-TRAIN_TEST_SPLIT_PERCENT)*100}%) training data and {len(trade_arrays)} ({TRAIN_TEST_SPLIT_PERCENT*100}%) trading data.")
    return train_arrays, trade_arrays



def generate_past_hours_permutation(start: int, end: int) -> List[dict]:
    items = []
    for item in (permutations(range(start, end), 2)):
        if item[0] < item[1]:
            features = ([f"PAST_{abs(i)}_HOUR" for i in range(item[0], item[1]+1)])
            items.append({
                "id": f"close-price-past-hours-({item[0]})-end-({item[1]})",
                "value": item,
                "technical_indicators": features
                })
    return items


def train(df: pd.DataFrame, experiment: dict, model_name: str):
    start_time = perf_counter()
    full_df = df.copy()
    full_df = add_features(full_df, experiment)
    full_df = clean_df(full_df)
    train_arrays, trade_arrays = split_train_test(full_df, technical_indicators=experiment["technical_indicators"])

    train_env = Monitor(StockTradingEnv(train_arrays, TICKERS, experiment["technical_indicators"]))
    trade_env = Monitor(StockTradingEnv(trade_arrays, TICKERS, experiment["technical_indicators"]))
    identifier = '-'.join(experiment['technical_indicators'])
    identifier = "close-price" if not identifier else identifier
    MODEL_PREFIX = f"{model_name}/{identifier}"
    TOTAL_TIMESTAMP = 50_000
    tensorboard_log = Path(f"{TENSORBOARD_LOG_DIR}/{model_name}")
    model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log=tensorboard_log)
    model.learn(
        total_timesteps=TOTAL_TIMESTAMP,
        callback=TensorboardCallback(
            save_freq=4096, model_prefix=MODEL_PREFIX, eval_env=trade_env
        ),
        tb_log_name=f"ppo-{TOTAL_TIMESTAMP}-{identifier}",
    )
    obs, _ = trade_env.reset()
    for i in range(len(trade_arrays)):
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = trade_env.step(action)
    print(info)
    log.info(f"Training took {perf_counter() - start_time: .2f} seconds.")


def benchmark():
    Path(TRAINED_MODEL_DIR).mkdir(parents=True, exist_ok=True)
    model_name = "ppo"
    EXPERIMENTS = [{"id":"close-price", "value": "Close", "technical_indicators": []}]
    past_hours_experiments = generate_past_hours_permutation(start=-48, end=0)
    EXPERIMENTS.extend(past_hours_experiments)
    df = load_df()
    with ProcessPoolExecutor(max_workers=2) as e:
        [e.submit(train, df, exp, model_name) for exp in EXPERIMENTS]


if __name__ == "__main__":
    # train()
    benchmark()
