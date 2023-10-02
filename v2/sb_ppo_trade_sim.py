import numpy as np
import polars as pl
from pathlib import Path
from envs.stock_trading_env import StockTradingEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
import random
import torch
from utils import (
    load_data,
    add_past_hours,
    train_test_split,
    create_numpy_array,
    get_ppo_model,
    add_technical_indicators,
    load_ppo_model,
    test_model,
    TensorboardCallback,
)
from stable_baselines3 import PPO

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
    df = add_technical_indicators(df)
    df = add_past_hours(df)
    df = df.with_columns(pl.lit(0.0).alias("Buy/Sold/Hold"))
    train_df, trade_df = train_test_split(df)

    assert train_df.columns == trade_df.columns

    train_arrays = create_numpy_array(train_df)
    trade_arrays = create_numpy_array(trade_df)

    train_env = Monitor(StockTradingEnv(train_arrays, [TICKERS]))
    trade_env = Monitor(StockTradingEnv(trade_arrays, [TICKERS]))

    check_env(trade_env)

    model_file = Path(TRAINED_MODEL_DIR) / "23-2723.8499145507812.zip"
    model = PPO.load(
        model_file,
        verbose=0,
        tensorboard_log=TENSORBOARD_LOG_DIR,
        print_system_info=False,
    )


    obs, info = trade_env.reset(seed=SEED)
    infos = [info]
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = trade_env.step(action)
        infos.append(info)
        if done or truncated:
            break

    # pl.DataFrame(infos).write_excel("sb_results.xlsx", column_widths=120)
    print(info)
    df = pl.DataFrame(infos)
    cols = [
        "action",
        "close_price",
        # "past_hour_mean",
        "previous_portfolio_value",
        "current_portfolio_value",
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
        # "good_buys",
        # "good_sells",
        # "good_holds",
        # "bad_buys",
        # "bad_sells",
        # "bad_holds",
        # "unsuccessful_buys",
        # "unsuccessful_sells",
        # "unsuccessful_holds",
        # "successful_buys",
        # "successful_sells",
        # "successful_holds",
        "reward",
    ]
    df.select(cols).write_excel("sb_results.xlsx", column_widths=100)



if __name__ == "__main__":
    main()
