from stock_prediction_rl.envs.numpy.stock_trading_env import StockTradingEnv
# from envs.stock_trading_env import StockTradingEnv
from stock_prediction_rl.sb.utils import (
    load_data,
    makedirs,
    create_numpy_array,
    create_envs,
)
from pathlib import Path
from stable_baselines3 import PPO, A2C
import json


def main():
    ticker = "SBIN.NS"
    trained_model_dir = Path("trained_models")
    model_name = "A2C"
    seed = 1337
    num_envs = 16


    makedirs()
    train_df, trade_df = load_data(ticker)
    train_array = create_numpy_array(train_df)
    trade_arrays = create_numpy_array(trade_df)

    trade_envs = create_envs(
        StockTradingEnv, trade_arrays, num_envs=num_envs, mode="trade", seed=seed
    )

    model_filename = trained_model_dir / f"sb_{model_name}_{ticker}_default_parameters"
    model = A2C.load(model_filename, env=trade_envs, force_reset=False)
    results = []

    obs = trade_envs.reset()
    counter = 0
    while counter < num_envs:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = trade_envs.step(action)
        for i in range(num_envs):
            if dones[i]:
                print(infos[i]["cummulative_profit_loss"])
                counter += 1



    # for env in trade_envs.envs:
    #     obs, t_info = env.reset(seed=seed)
    #     while True:
    #         action, _ = model.predict(obs, deterministic=True)
    #         obs, reward, done, truncated, t_info = env.step(action.item())
    #         if done or truncated:
    #             print(json.dumps(t_info, indent=4, default=str))
    #             results.append(t_info)
    #             break

    trade_envs.close()


if __name__ == "__main__":
    main()
