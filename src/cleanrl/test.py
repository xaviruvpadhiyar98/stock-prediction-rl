from envs.stock_trading_env_tensor import StockTradingEnv
from src.envs.simple_stock_trading_env_tensor import StockTradingEnv as OnlyBuySellEnv

from sb.utils import (
    load_data,
    makedirs,
    create_numpy_array,
)
from cleanrl.utils import create_torch_array, create_envs, Agent
from pathlib import Path
import torch
import torch.optim as optim
from tqdm import tqdm
from time import perf_counter
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


LEARNING_RATE = 5e-4
EPS = 1e-5
TOTAL_TIMESTEPS = 1_000_000
NUM_STEPS = 1024
NUM_ENVS = 16
EVAL_NUM_ENVS = 2 * 4
BATCH_SIZE = NUM_ENVS * NUM_STEPS
NUM_MINIBATCHES = 32
MINIBATCH_SIZE = BATCH_SIZE // NUM_MINIBATCHES
NUM_UPDATES = TOTAL_TIMESTEPS // BATCH_SIZE
GAE_LAMBDA = 0.95
GAMMA = 0.99
UPDATE_EPOCHS = 8
NORM_ADV = True
CLIP_COEF = 0.2
CLIP_VLOSS = True
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
TARGET_KL = None
CHECKPOINT_FREQUENCY = 1


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ticker = "SBIN.NS"
    trained_model_dir = Path("trained_models")
    model_name = "PPO"
    seed = 1337
    makedirs()
    _, trade_df = load_data(ticker)
    trade_arrays = create_torch_array(create_numpy_array(trade_df), device)

    trade_envs = create_envs(
        OnlyBuySellEnv, trade_arrays, num_envs=1, mode="trade", seed=seed
    )


    model_filename = trained_model_dir / f"{model_name}_{ticker}_cleanrl"
    model_filename = trained_model_dir / f"{model_name}_{ticker}_OnlyBuySellEnv_cleanrl"
    trade_agent = Agent(trade_envs).to(device)
    if model_filename.exists():
        print(f"Loading existing model from {model_filename}")
        trade_agent.load_state_dict(torch.load(model_filename, map_location=device))

    
    trade_obs, _ = trade_envs.reset()
    with torch.inference_mode():
        while True:
            t_action, _, _, _ = trade_agent.get_action_and_value(trade_obs, deterministic=True)
            trade_obs, reward, done, truncation, t_info = trade_envs.step(t_action)
            if "final_info" not in t_info:
                # print(t_info)
                ...
            else:
                final_info = t_info["final_info"]
                print(final_info)
                break

    trade_envs.close()

if __name__ == "__main__":
    main()
