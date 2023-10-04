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
from utils import *

TRAIN_ENVS, TRADE_ENV = get_train_trade_environment()

def main():


    

    trained_model = get_best_ppo_model(TRAIN_ENVS, SEED)
    # model = get_a2c_model(train_envs, N_STEPS, SEED)
    # model = load_ppo_model(train_envs)

    NUM_ENVS = 512
    N_STEPS = 16
    TIME_STAMPS = 32

    TOTAL_TIME_STAMPS = TIME_STAMPS * NUM_ENVS * N_STEPS

    trained_model.learn(
        total_timesteps=TOTAL_TIME_STAMPS,
        callback=TensorboardCallback(
            save_freq=1, model_prefix=MODEL_PREFIX, eval_env=TRADE_ENV, seed=SEED
        ),
        tb_log_name=f"sb_single_step_reward_early_stopping_best_{MODEL}_model",
        log_interval=1,
        progress_bar=True,
        reset_num_timesteps=False,
    )

    print(test_model(TRADE_ENV, trained_model, SEED))
    trained_model.save(Path(TRAINED_MODEL_DIR) / f"{MODEL_PREFIX}.zip")
    trade_model = load_ppo_model()
    print(test_model(TRADE_ENV, trade_model, SEED))

    TRAIN_ENVS.close()
    TRADE_ENV.close()


if __name__ == "__main__":
    main()
