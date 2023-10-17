from envs.stock_trading_env import StockTradingEnv
from sb.utils import (
    load_data,
    makedirs,
    create_numpy_array,
    create_envs,
    sample_ppo_params,
    TensorboardCallback,
)
from pathlib import Path
from stable_baselines3 import PPO
from optuna import Trial, create_study
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

def objective(trial: Trial) -> float:
    ticker = "SBIN.NS"
    trained_model_dir = Path("trained_models")
    model_name = "PPO"
    seed = 1337
    num_envs = 20
    multiplier = 100

    makedirs()
    train_df, trade_df = load_data(ticker)
    train_arrays = create_numpy_array(train_df)
    trade_arrays = create_numpy_array(trade_df)
    train_ending_index = len(train_arrays) - 1
    trade_ending_index = len(trade_arrays) - 1

    train_envs = create_envs(
        StockTradingEnv, train_arrays, num_envs=num_envs, mode="train", seed=seed
    )
    trade_envs = create_envs(
        StockTradingEnv, trade_arrays, num_envs=num_envs, mode="trade", seed=seed
    )

    hp = sample_ppo_params(trial)
    hp.update({"env": train_envs, "seed": seed})
    model = PPO(**hp)
    reset_num_timesteps = True

    trade_env = trade_envs.envs[0]
    total_timesteps = num_envs * model.n_steps * multiplier


    try:
        model.learn(
            total_timesteps=total_timesteps,
            log_interval=None,
            progress_bar=True,
            reset_num_timesteps=reset_num_timesteps,
        )
    except KeyboardInterrupt:
        ...
    
    trade_model = PPO(**hp)
    parameters = model.get_parameters()
    trade_model.set_parameters(parameters)

    ending_infos = []
    for trade_env in trade_envs.envs:
        obs, t_info = trade_env.reset(seed=trade_env.seed)
        while True:
            action, _ = trade_model.predict(obs, deterministic=True)
            obs, reward, done, truncated, t_info = trade_env.step(action.item())
            if done or truncated:
                ending_infos.append(t_info)
                trade_env.close()
                break
    
    best_cummulative_profit_loss = 0
    best_seed = 0
    for info in ending_infos:
        if info["index"] == trade_ending_index:
            cpl = info["cummulative_profit_loss"]
            if cpl > best_cummulative_profit_loss:
                best_cummulative_profit_loss = cpl
                best_seed = info["seed"]
    
    print(f"{best_seed=}. {best_cummulative_profit_loss=}")
    return best_cummulative_profit_loss


def main():
    N_STARTUP_TRIALS = 102
    SEED = 1337
    N_TRIALS = 50

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS, seed=SEED+1)
    study = create_study(
        sampler=sampler, direction="maximize", pruner=HyperbandPruner()
    )

    try:
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    main()
