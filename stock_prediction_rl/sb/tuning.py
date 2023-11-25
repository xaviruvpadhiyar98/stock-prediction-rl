from stock_prediction_rl.envs.numpy.stock_trading_validation_env import StockTradingEnv
from stock_prediction_rl.sb.utils import (
    load_data,
    makedirs,
    create_numpy_array,
    create_envs,
    sample_ppo_params,
    sample_a2c_params,
)
from pathlib import Path
from stable_baselines3 import PPO, A2C
from optuna import Trial, create_study
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler


def objective(trial: Trial) -> float:
    ticker = "SBIN.NS"
    seed = 1337
    num_envs = 20
    multiplier = 40
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

    # hp = sample_ppo_params(trial)
    hp = sample_a2c_params(trial)
    hp.update({"env": trade_envs, "seed": seed, "device": "cpu"})
    model = A2C(**hp)
    reset_num_timesteps = True

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

    # trade_model = PPO(**hp)
    trade_model = A2C(**hp)
    parameters = model.get_parameters()
    trade_model.set_parameters(parameters)

    ending_infos = []
    obs = trade_envs.reset()
    counter = 0
    final_reward = 0
    while counter < num_envs:
        action, _ = trade_model.predict(obs)
        obs, rewards, dones, infos = trade_envs.step(action)
        for i in range(num_envs):
            if dones[i]:
                if infos[i]["episode"]["r"] > final_reward:
                    final_reward = infos[i]["episode"]["r"]
                counter += 1

    print(f"{final_reward=}")
    return final_reward


def main():
    N_STARTUP_TRIALS = 10
    SEED = 42
    N_TRIALS = 50

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS, seed=SEED + 1)
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


# [I 2023-10-28 01:54:05,925] Trial 9 finished with value: 350.0 and parameters: {'batch_size': 128, 'n_steps': 32, 'gamma': 0.99, 'learning_rate': 1.4692919768226823e-05, 'lr_schedule': 'constant', 'ent_coef': 0.03237245745270013, 'clip_range': 0.3, 'n_epochs': 5, 'gae_lambda': 0.95, 'max_grad_norm': 5, 'vf_coef': 0.018167636778965695, 'net_arch': 'medium', 'ortho_init': False, 'activation_fn': 'tanh'}. Best is trial 9 with value: 350.0.


# [I 2023-10-28 15:18:14,371] Trial 1 finished with value: 352.0 and parameters: {'batch_size': 8, 'n_steps': 16, 'gamma': 0.98, 'learning_rate': 5.957264071067262e-05, 'lr_schedule': 'linear', 'ent_coef': 2.8582703822117275e-06, 'clip_range': 0.2, 'n_epochs': 5, 'gae_lambda': 0.98, 'max_grad_norm': 0.9, 'vf_coef': 0.6727147564528313, 'net_arch': 'medium', 'ortho_init': True, 'activation_fn': 'tanh'}. Best is trial 1 with value: 352.0.

# A2C
# [I 2023-10-28 17:33:38,406] Trial 0 finished with value: 352.0 and parameters: {'gamma': 0.0020322854432411525, 'max_grad_norm': 0.3227001658898119, 'gae_lambda': 0.01839881485129108, 'exponent_n_steps': 6, 'lr': 0.0012642676064986635, 'ent_coef': 2.0527863302764663e-06, 'ortho_init': True, 'net_arch': 'tiny', 'activation_fn': 'tanh'}. Best is trial 0 with value: 352.0.

# A2C
# Best trial:
#   Value:  352.0
#   Params:
#     gamma: 0.0020322854432411525
#     max_grad_norm: 0.3227001658898119
#     gae_lambda: 0.01839881485129108
#     exponent_n_steps: 6
#     lr: 0.0012642676064986635
#     ent_coef: 2.0527863302764663e-06
#     ortho_init: True
#     net_arch: tiny
#     activation_fn: tanh
#   User attrs:
#     gamma_: 0.9979677145567588
#     gae_lambda_: 0.981601185148709
#     n_steps: 64


# [I 2023-10-28 18:05:49,923] Trial 1 finished with value: 352.0 and parameters: {'gamma': 0.98, 'normalize_advantage': False, 'max_grad_norm': 0.6, 'use_rms_prop': False, 'gae_lambda': 0.9, 'n_steps': 8, 'lr_schedule': 'constant', 'learning_rate': 0.0011872144854203988, 'ent_coef': 2.8582703822117275e-06, 'vf_coef': 0.5509779053244193, 'ortho_init': False, 'net_arch': 'small', 'activation_fn': 'tanh'}. Best is trial 1 with value: 352.0.

# A2C
# Number of finished trials:  50
# Best trial:
#   Value:  352.0
#   Params:
#     gamma: 0.98
#     normalize_advantage: False
#     max_grad_norm: 0.6
#     use_rms_prop: False
#     gae_lambda: 0.9
#     n_steps: 8
#     lr_schedule: constant
#     learning_rate: 0.0011872144854203988
#     ent_coef: 2.8582703822117275e-06
#     vf_coef: 0.5509779053244193
#     ortho_init: False
#     net_arch: small
#     activation_fn: tanh
#   User attrs:
