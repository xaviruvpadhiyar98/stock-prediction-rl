import numpy as np
import polars as pl
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


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TICKERS = "SBIN.NS"
SEED = 1337
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

    # env setup
    trade_envs = SyncVectorEnv([make_env(StockTradingEnv, trade_arrays, TICKERS, SEED)])

    trade_agent = Agent(trade_envs).to(DEVICE)
    trade_agent.load_state_dict(torch.load(MODEL_SAVE_FILE, map_location=DEVICE))
    trade_agent.eval()
    obs, _ = trade_envs.reset(seed=SEED)
    infos = []
    while True:
        action, _, _, _ = trade_agent.get_action_and_value(obs)
        obs, rewards, done, _, info = trade_envs.step(action)
        for k, v in info.items():
            if v is None:
                info[k] = np.NAN
            else:
                info[k] = v[0]
        info["reward"] = rewards
        infos.append(info)
        if done:
            break

    # print(infos[0])
    # print([info["reward"] for info in infos])
    df = pl.DataFrame(infos)
    cols = [
        "close_price",
        "action",
        "shares_bought",
        "shares_sold",
        "buy_prices_with_commission",
        "sell_prices_with_commission",
        "profit_or_loss",
        "cummulative_profit_loss",
        "available_amount",
        "shares_holdings",
        "good_buys",
        "good_sells",
        "good_holds",
        "bad_buys",
        "bad_sells",
        "bad_holds",
        "successful_trades",
        "unsuccessful_trades",
        "reward",
    ]
    # cols = [col for col in df.columns if not col.startswith("_")]
    df = df.select(cols)
    # df = df.select(pl.all().fill_null(0))
    print(df)
    df.write_excel("results.xlsx")


if __name__ == "__main__":
    main()
