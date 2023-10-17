from envs.stock_trading_env import StockTradingEnv
from sb.utils import (
    load_data,
    makedirs,
    create_numpy_array,
    create_envs,
    get_ppo_model,
    get_default_ppo_model,
    TensorboardCallback,
)
from pathlib import Path
from stable_baselines3 import PPO


def main():
    ticker = "SBIN.NS"
    trained_model_dir = Path("trained_models")
    model_name = "PPO"
    seed = 1337
    num_envs = 256
    multiplier = 1000

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
    trade_env = trade_envs.envs[0]

    model_filename = trained_model_dir / f"{model_name}_{ticker}"
    if model_filename.exists():
        model = PPO.load(model_filename, env=train_envs)
        reset_num_timesteps = False
    else:
        model = get_ppo_model(train_envs, seed=seed)
        model = get_default_ppo_model(train_envs, seed=seed)
        reset_num_timesteps = True

    total_timesteps = num_envs * model.n_steps * multiplier
    tb_log_name = f"{model_name}_{ticker}_{model.n_steps}_{num_envs}"

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=TensorboardCallback(
                eval_envs=trade_envs, train_ending_index=train_ending_index
            ),
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
