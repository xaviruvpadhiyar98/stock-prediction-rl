import numpy as np
import polars as pl
from pathlib import Path
from envs.stock_trading_env import StockTradingEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

import torch
from utils import *

from optuna import Trial, create_study
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

TRAIN_ENVS, TRADE_ENV = get_train_trade_environment()

def objective(trial: Trial) -> float:
    hp = sample_ppo_params(trial)
    hp.update({"env": TRAIN_ENVS, "seed":SEED})
    model = PPO(**hp)

    TIME_STAMPS = 4
    NUM_ENVS = 16
    N_STEPS = hp["n_steps"]
    TOTAL_TIME_STAMPS = TIME_STAMPS * NUM_ENVS * N_STEPS

    model.learn(TOTAL_TIME_STAMPS)

    info = test_model(TRADE_ENV, model, SEED)
    cummulative_profit_loss = info["cummulative_profit_loss"]
    
    filename = TRAINED_MODEL_DIR / f"{trial.number}-{cummulative_profit_loss}.zip"
    model.save(filename)
    
    trade_model = PPO.load(filename)
    info = test_model(TRADE_ENV, trade_model, SEED)
    trade_cummulative_profit_loss = info["cummulative_profit_loss"]

    assert cummulative_profit_loss == trade_cummulative_profit_loss
    return info["cummulative_profit_loss"]


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

    TRAIN_ENVS.close()
    TRADE_ENV.close()


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
