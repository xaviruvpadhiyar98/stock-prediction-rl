import numpy as np
import polars as pl
from pathlib import Path
from envs.stock_trading_env_using_numpy import StockTradingEnv
from stable_baselines3.common.monitor import Monitor
import random
import torch
from utils import (
    load_data,
    add_past_hours,
    train_test_split,
    create_numpy_array,
    get_ppo_model,
    load_ppo_model,
    test_model,
    TensorboardCallback,
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

    train_env = Monitor(StockTradingEnv(train_arrays, [TICKERS]))
    trade_env = Monitor(StockTradingEnv(trade_arrays, [TICKERS]))

    # model = get_ppo_model(train_env, SEED)
    model = load_ppo_model(train_env)

    model.learn(
        total_timesteps=2_000_000,
        callback=TensorboardCallback(
            save_freq=4096, model_prefix=MODEL_PREFIX, eval_env=trade_env, seed=SEED
        ),
        tb_log_name="ppo_cummulative_reward",
        log_interval=1,
        progress_bar=True,
        reset_num_timesteps=False,
    )
    test_model(trade_env, model, SEED)
    train_env.close()


if __name__ == "__main__":
    main()