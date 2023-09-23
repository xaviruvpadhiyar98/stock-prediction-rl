import numpy as np
import polars as pl
import json
from pathlib import Path
from envs.stock_trading_env_using_tensor import StockTradingEnv
import random
import torch
from ppo_agent import Agent
from gymnasium.vector import SyncVectorEnv
from utils import (
    load_data,
    add_past_hours,
    train_test_split,
    create_torch_array,
    make_env,
)
from copy import deepcopy
from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TICKERS = "SBIN.NS"
SEED = 1337
NUM_ENVS = 2**7
BEST_ENV = 89
TRAINED_MODEL_DIR = Path("trained_models")
TENSORBOARD_LOG_DIR = Path("tensorboard_log")
MODEL_SAVE_FILE = TRAINED_MODEL_DIR / "clean_rl_agent_ppo.pt"

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

    trade_arrays = create_torch_array(trade_df, device=DEVICE)
    trade_envs = SyncVectorEnv(
        [make_env(StockTradingEnv, trade_arrays, TICKERS) for _ in range(NUM_ENVS)]
    )
    trade_agent = Agent(trade_envs).to(DEVICE)
    trade_agent.load_state_dict(torch.load(MODEL_SAVE_FILE, map_location=DEVICE))
    trade_agent.eval()
    obs, info = trade_envs.reset(seed=SEED)
    new_info = {}
    for k, v in deepcopy(info).items():
        if k.startswith("_"):
            continue
        new_info[k] = v[BEST_ENV]
    infos = [new_info]
    while True:
        action, _, _, _ = trade_agent.get_action_and_value(obs)
        obs, rewards, done, _, info = trade_envs.step(action)
        if "final_info" in info:
            final_info = deepcopy(info["final_info"])[BEST_ENV]
            infos.append(final_info)
            break

        new_info = {}
        for k, v in deepcopy(info).items():
            if k.startswith("_"):
                continue
            new_info[k] = v[BEST_ENV]
        infos.append(new_info)

        # i = deepcopy(info)
        #     item = v[103]
        #     if item is None:
        #         i[k] = np.NAN
        #     else:
        #         i[k] = item
        # infos.append(i)

    Path("results").write_text(json.dumps(infos, indent=4, default=str))
    df = pl.DataFrame(infos)
    cols = [
        "close_price",
        "action",
        "portfolio_value",
        "shares_holdings",
        "available_amount",
        "avg_buy_price",
        "avg_sell_price",
        "shares_bought",
        "shares_sold",
        "buy_prices_with_commission",
        "sell_prices_with_commission",
        "profit_or_loss",
        "cummulative_profit_loss",
        "good_buys",
        "good_sells",
        "good_holds",
        "bad_buys",
        "bad_sells",
        "bad_holds",
        "reward",
    ]
    df = df.select(cols)
    print(df)
    df.write_excel("results.xlsx")


if __name__ == "__main__":
    main()
