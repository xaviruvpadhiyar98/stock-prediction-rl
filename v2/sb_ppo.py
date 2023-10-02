import numpy as np
import polars as pl
from pathlib import Path
from envs.stock_trading_env import StockTradingEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
import random
import torch
from utils import (
    load_data,
    add_past_hours,
    train_test_split,
    create_numpy_array,
    make_env,
    add_technical_indicators,
    get_ppo_model,
    get_best_ppo_model,
    get_a2c_model,
    load_ppo_model,
    test_model,
    TensorboardCallback,
)

TICKERS = "SBIN.NS"
INTERVAL = "1h"
PERIOD = "360d"
MODEL_SUFFIX = "PPO"
MODEL_PREFIX = f"{TICKERS}_{MODEL_SUFFIX}"
NUM_ENVS = 4096 * 16
NUM_ENVS = 16
N_STEPS = 512
TIME_STAMPS = 8
TOTAL_TIME_STAMPS = TIME_STAMPS * NUM_ENVS * N_STEPS


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
    df = add_technical_indicators(df)
    df = df.with_columns(pl.lit(0.0).alias("Buy/Sold/Hold"))
    train_df, trade_df = train_test_split(df)

    assert train_df.columns == trade_df.columns

    train_arrays = create_numpy_array(train_df)
    trade_arrays = create_numpy_array(trade_df)

    train_envs = DummyVecEnv(
        [
            make_env(StockTradingEnv, train_arrays, [TICKERS], False, SEED, i)
            for i in range(NUM_ENVS)
        ]
    )
    trade_env = Monitor(StockTradingEnv(trade_arrays, [TICKERS]))
    check_env(trade_env)

    trained_model = get_best_ppo_model(train_envs, N_STEPS, SEED)
    # model = get_a2c_model(train_envs, N_STEPS, SEED)
    # model = load_ppo_model(train_envs)

    trained_model.learn(
        total_timesteps=TOTAL_TIME_STAMPS,
        callback=TensorboardCallback(
            save_freq=1, model_prefix=MODEL_PREFIX, eval_env=trade_env, seed=SEED
        ),
        tb_log_name=f"sb_single_step_reward_early_stopping_best_{MODEL_SUFFIX}_model",
        log_interval=1,
        progress_bar=True,
        reset_num_timesteps=False,
    )

    print(test_model(trade_env, trained_model, SEED))
    trained_model.save(Path(TRAINED_MODEL_DIR) / f"{MODEL_PREFIX}.zip")
    trade_model = load_ppo_model()
    print(test_model(trade_env, trade_model, SEED))

    train_envs.close()
    trade_env.close()


if __name__ == "__main__":
    main()
