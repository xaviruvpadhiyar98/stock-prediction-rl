import numpy as np
import polars as pl
from pathlib import Path
from envs.stock_trading_env import StockTradingEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from utils import *
from stable_baselines3 import PPO


TRAIN_ENVS, TRADE_ENV = get_train_trade_environment()


def main():
    model_file = TRAINED_MODEL_DIR / "39-301.75079345703125.zip"

    model = PPO.load(
        model_file,
        TRAIN_ENVS,
        tensorboard_log=TENSORBOARD_LOG_DIR,
    )
    info = test_model(TRADE_ENV, model, SEED)
    print(info)

    # Resume Training
    NUM_ENVS = 512
    TOTAL_TIME_STAMPS = 64 * NUM_ENVS * model.n_steps
    # tb_log_name = f"sb_{MODEL}_resume_best_model_from_optuna_{int(info['cummulative_profit_loss'])}"
    tb_log_name = f"sb_single_step_reward_early_stopping_best_{MODEL}_model"

    model.learn(
        total_timesteps=TOTAL_TIME_STAMPS,
        callback=TensorboardCallback(
            save_freq=1, model_prefix=MODEL_PREFIX, eval_env=TRADE_ENV, seed=SEED
        ),
        tb_log_name=tb_log_name,
        log_interval=1,
        progress_bar=True,
        reset_num_timesteps=False,
    )
    info = test_model(TRADE_ENV, model, SEED)
    model_file = TRAINED_MODEL_DIR / f"{MODEL_PREFIX}.zip"
    model.save(model_file)
    trade_model = PPO.load(model_file, TRAIN_ENVS)
    t_info = test_model(TRADE_ENV, model, SEED)
    print(info)
    print(t_info)
    assert info["cummulative_profit_loss"] == t_info["cummulative_profit_loss"]


if __name__ == "__main__":
    main()
