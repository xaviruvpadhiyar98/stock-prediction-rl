from envs.stock_trading_env import StockTradingEnv
from sb.utils import (
    load_data,
    makedirs,
    create_numpy_array,
    create_envs,
    get_ppo_model,
    TensorboardCallback,
)
from pathlib import Path
import json


def main():
    ticker = "SBIN.NS"
    trained_model_dir = Path("trained_models")
    model_name = "PPO"
    seed = 1337
    num_envs = 2
    multiplier = 1000

    makedirs()
    train_df, trade_df = load_data(ticker)
    train_arrays = create_numpy_array(train_df)
    trade_arrays = create_numpy_array(trade_df)
    train_ending_index = len(train_arrays) - 1

    train_env = StockTradingEnv(train_arrays, seed=seed)
    model = get_ppo_model(train_env, seed=seed)

    actions = [2, 0, 1, 2, 0, 0]

    obs, info = train_env.reset()
    print(obs[-4:], info)
    print()

    for action in actions:
        obs, reward, done, truncated, info = train_env.step(action)
        print(json.dumps(info, indent=4))
        print()
        # break


if __name__ == "__main__":
    main()
