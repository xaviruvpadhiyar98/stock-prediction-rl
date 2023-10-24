from stock_prediction_rl.envs.numpy.stock_trading_env import StockTradingEnv
from stock_prediction_rl.sb.utils import (
    load_data,
    makedirs,
    create_numpy_array,
    create_envs,
    get_ppo_model,
    get_default_ppo_model,
    PPOCallback,
    A2CCallback
)
from pathlib import Path
from stable_baselines3 import PPO, A2C, TD3
import random
import torch
import numpy as np


SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def main():
    ticker = "SBIN.NS"
    trained_model_dir = Path("trained_models")
    tensorboard_log = Path("tensorboard_log")
    model_name = "PPO"
    seed = 1
    num_envs = 20
    multiplier = 1_000_000

    makedirs()
    train_df, trade_df = load_data(ticker)
    train_arrays = create_numpy_array(train_df)
    trade_arrays = create_numpy_array(trade_df)
    train_ending_index = len(train_arrays) - 1

    train_envs = create_envs(
        StockTradingEnv, train_arrays, num_envs=num_envs, mode="train", seed=seed
    )
    trade_envs = create_envs(
        StockTradingEnv, trade_arrays, num_envs=num_envs, mode="trade", seed=seed
    )


    model_filename = trained_model_dir / f"sb_{model_name}_{ticker}_single_digit_reward_default_parameters"
    if model_filename.exists():
        # model = PPO.load(model_filename, env=train_envs, print_system_info=True)
        model = TD3.load(model_filename, env=train_envs, print_system_info=True)
        reset_num_timesteps = False
        print(f"Loading the model...")
    else:
        # model = PPO(policy="MlpPolicy", env=train_envs, tensorboard_log=tensorboard_log, ent_coef=0.2)
        model = TD3(policy="MlpPolicy", env=train_envs, tensorboard_log=tensorboard_log, ent_coef=0.2)
        reset_num_timesteps = True
        # # model = get_ppo_model(train_envs, seed=seed)
        # model = get_default_ppo_model(train_envs, seed=seed)

    total_timesteps = num_envs * model.n_steps * multiplier
    tb_log_name = model_filename.stem

    try:
        model.learn(
            total_timesteps=total_timesteps,
            # callback=A2CCallback(eval_envs=trade_envs),
            callback=PPOCallback(eval_envs=trade_envs),
            tb_log_name=tb_log_name,
            log_interval=1,
            progress_bar=True,
            reset_num_timesteps=reset_num_timesteps,
        )
    except KeyboardInterrupt:
        ...
    model.save(model_filename)
    train_envs.close()
    trade_envs.close()


if __name__ == "__main__":
    main()