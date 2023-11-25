# from stock_prediction_rl.envs.numpy.stock_trading_env import StockTradingEnv
from stock_prediction_rl.envs.numpy.stock_trading_validation_env import StockTradingEnv
from stock_prediction_rl.sb.utils import (
    load_data,
    makedirs,
    create_numpy_array,
    create_envs,
    linear_schedule,
    PPOCallback,
)
from pathlib import Path
from stable_baselines3 import PPO
import random
import torch
import numpy as np
import torch.nn as nn


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
    num_envs = 100
    multiplier = 1_000_000
    model_filename = (
        trained_model_dir
        / f"sb_{model_name}_{ticker}_train_on_validation_dataset_only_buy_with_hyper_parameters"
    )
    tb_log_name = model_filename.stem

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

    eval_envs = create_envs(
        StockTradingEnv, trade_arrays, num_envs=num_envs, mode="trade", seed=seed
    )

    if model_filename.exists():
        model = PPO.load(model_filename, env=trade_envs, print_system_info=True)
        reset_num_timesteps = False
        print(f"Loading the model...")
    else:
        model = PPO(
            policy="MlpPolicy",
            env=trade_envs,
            tensorboard_log=tensorboard_log,
            ent_coef=2.8582703822117275e-06,
            batch_size=8,
            n_steps=16,
            n_epochs=5,
            gamma=0.98,
            gae_lambda=0.98,
            max_grad_norm=0.9,
            vf_coef=0.6727147564528313,
            learning_rate=linear_schedule(5.957264071067262e-05),
            clip_range=0.2,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                activation_fn=nn.Tanh,
                ortho_init=True,
            ),
        )
        reset_num_timesteps = True

    total_timesteps = num_envs * model.n_steps * multiplier

    try:
        model.learn(
            total_timesteps=total_timesteps,
            # callback=A2CCallback(eval_envs=eval_envs),
            callback=PPOCallback(eval_envs=eval_envs),
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
