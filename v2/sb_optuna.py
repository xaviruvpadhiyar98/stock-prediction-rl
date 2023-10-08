from stable_baselines3 import PPO

from utils import *

from optuna import Trial, create_study
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

# TRAIN_ENVS, TRADE_ENV = get_train_trade_environment(
#     framework="sb", num_envs=NUM_ENVS, seed=SEED
# )


def objective(trial: Trial) -> float:

    MODEL = "PPO"
    FRAMEWORK = "sb"
    LEARN_DESCRIBE = "best_param_early_stopping"
    SEED = 1337
    NUM_ENVS = 256
    N_STEPS = 64

    tb_log_name = (
        f"{MODEL}_{FRAMEWORK}_" f"{LEARN_DESCRIBE}_{NUM_ENVS}_" f"{N_STEPS}_{SEED}"
    )

    multiplier = 200
    total_timesteps = NUM_ENVS * N_STEPS * multiplier

    train_envs, trade_env = get_train_trade_environment(
        framework="sb", num_envs=NUM_ENVS, seed=SEED
    )

    hp = sample_ppo_params(trial)
    hp.update({"env": train_envs, "seed": SEED})
    trained_model = PPO(**hp)
    assert trained_model.ent_coef == hp["ent_coef"]

    N_STEPS = hp["n_steps"]
    multiplier = 2
    total_timesteps = NUM_ENVS * N_STEPS * multiplier
    model_path = Path(TRAINED_MODEL_DIR) / f"{MODEL}.zip"

    trained_model.learn(
        total_timesteps=total_timesteps,
        callback=OptunaCallback(eval_env=trade_env, num_envs=NUM_ENVS),
        tb_log_name=tb_log_name,
        log_interval=1,
        progress_bar=True,
        reset_num_timesteps=True,
    )

    try:
        sb_best_env = json.loads(Path("sb_best_env.json").read_text())
        seed = sb_best_env["env_id"]
    except:
        seed = SEED
    info = test_model(trade_env, trained_model, seed)
    cummulative_profit_loss = info["cummulative_profit_loss"]
    train_envs.close()
    trade_env.close()
    return cummulative_profit_loss


def main():
    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS, seed=SEED)
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


"""
Best trial:
  Value:  7490.798767089844
  Params: 
    batch_size: 16
    n_steps: 16
    gamma: 0.98
    learning_rate: 3.7141262285419446e-05
    lr_schedule: linear
    ent_coef: 0.0003689138501403059
    clip_range: 0.1
    n_epochs: 20
    gae_lambda: 0.98
    max_grad_norm: 0.8
    vf_coef: 0.015611337828753173
    net_arch: medium
    ortho_init: True
    activation_fn: tanh
  User attrs:
"""
