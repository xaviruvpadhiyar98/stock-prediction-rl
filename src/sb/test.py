from src.envs.simple_stock_trading_env_numpy import StockTradingEnv
# from envs.stock_trading_env import StockTradingEnv
from sb.utils import (
    load_data,
    makedirs,
    create_numpy_array,
    create_envs,
)
from pathlib import Path
from stable_baselines3 import PPO
import json


def main():
    ticker = "SBIN.NS"
    trained_model_dir = Path("trained_models")
    model_name = "PPO"
    seed = 1337
    num_envs = 1
    multiplier = 1000

    makedirs()
    train_df, trade_df = load_data(ticker)
    train_arrays = create_numpy_array(train_df)
    trade_arrays = create_numpy_array(trade_df)



    train_envs = create_envs(
        StockTradingEnv, train_arrays, num_envs=num_envs, mode="train", seed=seed
    )
    trade_envs = create_envs(
        StockTradingEnv, trade_arrays, num_envs=num_envs, mode="trade", seed=seed
    )

    model_filename = trained_model_dir / f"{model_name}_{ticker}_sb"
    model = PPO.load(model_filename, env=trade_envs, force_reset=False)
    results = []

    for env in trade_envs.envs:
        obs, t_info = env.reset(seed=seed)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, t_info = env.step(action.item())
            if done or truncated:
                print(json.dumps(t_info, indent=4, default=str))
                results.append(t_info)
                break

    Path("results").write_text(json.dumps(results, default=str, indent=4))
    train_envs.close()
    trade_envs.close()


if __name__ == "__main__":
    main()
