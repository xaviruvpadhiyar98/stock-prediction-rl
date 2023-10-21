# from envs.stock_trading_env import StockTradingEnv
from envs.stock_trading_env_numpy_buy_sell import StockTradingEnv
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
    seed = 40
    num_envs = 32
    multiplier = 40

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


    model_filename = trained_model_dir / f"{model_name}_{ticker}_sb"
    if model_filename.exists():
        model = PPO.load(model_filename, env=train_envs, print_system_info=True)
        reset_num_timesteps = False
        print(f"Loading the model...")
    else:
        # model = get_ppo_model(train_envs, seed=seed)
        model = get_default_ppo_model(train_envs, seed=seed)
        reset_num_timesteps = True

    total_timesteps = num_envs * model.n_steps * multiplier
    tb_log_name = f"{model_name}_{ticker}_{model.n_steps}_{num_envs}_sb"

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
