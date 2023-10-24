import numpy as np
import polars as pl
from pathlib import Path
from envs.stock_trading_env_using_numpy import StockTradingEnv
from gymnasium.vector import SyncVectorEnv
import random
import torch
from tqdm import tqdm
from utils import (
    load_data,
    add_past_hours,
    train_test_split,
    create_numpy_array,
    make_env,
)

TICKERS = "SBIN.NS"
INTERVAL = "1h"
PERIOD = "360d"
MODEL_PREFIX = f"{TICKERS}_PPO"

TRAIN_TEST_SPLIT_PERCENT = 0.15
PAST_HOURS = range(1, 15)
TECHNICAL_INDICATORS = [f"PAST_{hour}_HOUR" for hour in PAST_HOURS]
DATASET = Path("datasets")
DATASET.mkdir(parents=True, exist_ok=True)
TRAINED_MODEL_DIR = Path("trained_models")
TENSORBOARD_LOG_DIR = Path("tensorboard_log")

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def main():
    df = load_data()
    df = add_past_hours(df)
    df = df.with_columns(pl.lit(0.0).alias("Buy/Sold/Hold"))
    train_df, trade_df = train_test_split(df)

    assert train_df.columns == trade_df.columns

    train_arrays = create_numpy_array(train_df)
    trade_arrays = create_numpy_array(trade_df)

    NUM_ENVS = 1000
    M = len(train_arrays)  # length of each sequence

    train_envs = SyncVectorEnv(
        [
            make_env(StockTradingEnv, train_arrays, [TICKERS], seed)
            for seed in range(NUM_ENVS)
        ]
    )
    trade_env = SyncVectorEnv(
        [make_env(StockTradingEnv, trade_arrays, [TICKERS], 1337)]
    )

    actions = np.random.choice(
        [StockTradingEnv.BUY, StockTradingEnv.SELL, StockTradingEnv.HOLD],
        size=(M, NUM_ENVS),
    )
    obs, _ = train_envs.reset()
    for action in tqdm(actions):
        *_, done, _, info = train_envs.step(action)
        if done[0]:
            break

    infos = info["final_info"]
    new_infos = {}
    holdings = 0
    for i, item in enumerate(infos):
        if item["holdings"] > holdings:
            new_infos = item
            new_infos["action"] = actions.T[i]
    print(new_infos)
    train_envs.close()

    action = new_infos["action"]
    trade_env.reset()
    for a in action:
        *_, done, _, info = trade_env.step([a])
        if done:
            break
    info = info["final_info"][0]
    print(info)
    trade_env.close()


if __name__ == "__main__":
    main()
