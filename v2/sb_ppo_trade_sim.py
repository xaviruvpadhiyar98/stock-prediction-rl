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

    model_file = Path(TRAINED_MODEL_DIR) / "39-301.75079345703125.zip"
    trade_model = PPO.load(model_file)

    obs, info = TRADE_ENV.reset(seed=SEED)
    infos = [info]
    while True:
        action, _ = trade_model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = TRADE_ENV.step(action)
        infos.append(info)
        if done or truncated:
            break


    print(info)



    # # pl.DataFrame(infos).write_excel("sb_results.xlsx", column_widths=120)
    # print(info)
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
