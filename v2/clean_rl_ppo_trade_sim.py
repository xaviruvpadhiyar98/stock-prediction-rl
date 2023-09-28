import numpy as np
import polars as pl
from pathlib import Path
from envs.stock_trading_env_using_tensor import StockTradingEnv
import random
import torch
from v2.clean_rl_ppo_agent import Agent
from gymnasium.vector import SyncVectorEnv
from utils import (
    load_data,
    add_past_hours,
    add_technical_indicators,
    train_test_split,
    create_torch_array,
    make_env,
)
from copy import deepcopy
from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TICKERS = "SBIN.NS"
SEED = 1337
NUM_ENVS = 2**4
TRAINED_MODEL_DIR = Path("trained_models")
TENSORBOARD_LOG_DIR = Path("tensorboard_log")
MODEL_SAVE_FILE = TRAINED_MODEL_DIR / "clean_rl_agent_ppo.pt"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def main():
    df = load_data()
    df = add_technical_indicators(df)
    df = add_past_hours(df)
    df = df.with_columns(pl.lit(0.0).alias("Buy/Sold/Hold"))
    train_df, trade_df = train_test_split(df)

    assert train_df.columns == trade_df.columns

    trade_arrays = create_torch_array(trade_df, device=DEVICE)

    trade_envs = SyncVectorEnv(
        [make_env(StockTradingEnv, trade_arrays, TICKERS) for _ in range(NUM_ENVS)]
    )
    trade_agent = Agent(trade_envs).to(DEVICE)
    trade_agent.load_state_dict(torch.load(MODEL_SAVE_FILE, map_location=DEVICE))
    trade_agent.eval()
    obs, info = trade_envs.reset(seed=SEED)
    while True:
        action, _, _, _ = trade_agent.get_action_and_value(obs)
        obs, rewards, done, _, info = trade_envs.step(action)
        if "final_info" in info:
            final_info = info["final_info"]
            for i, fi in enumerate(final_info):
                yield (i, fi["cummulative_profit_loss"])
            break


if __name__ == "__main__":
    results = [x for x in main()]
    profit_loss_array = torch.Tensor([r[1] for r in results])
    idx = profit_loss_array.argmax()
    print(results[idx])
