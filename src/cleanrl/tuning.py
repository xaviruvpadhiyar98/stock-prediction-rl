from envs.stock_trading_env_tensor import StockTradingEnv
from src.envs.simple_stock_trading_env_tensor import StockTradingEnv as OnlyBuySellEnv

from sb.utils import (
    load_data,
    makedirs,
    create_numpy_array,
)
from cleanrl.utils import create_torch_array, sample_ppo_params, train
from pathlib import Path
import torch
import torch.optim as optim
from tqdm import tqdm
from time import perf_counter
import numpy as np
import torch.nn as nn

import json
from optuna import Trial, create_study
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

from subprocess import run, PIPE

def log_gpu():
    gpu_query = "utilization.gpu,utilization.memory"
    format = "csv,noheader,nounits"
    gpu_util, gpu_memory = run(
        [
            "nvidia-smi",
            f"--query-gpu={gpu_query}",
            f"--format={format}",
        ],
        encoding="utf-8",
        stdout=PIPE,
        stderr=PIPE,
        check=True,
    ).stdout.split(",")
    info = {
        "utilization": float(gpu_util.strip()),
        "memory": float(gpu_memory.strip()),
    }
    print(info)


def objective(trial: Trial) -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ticker = "SBIN.NS"
    seed = 40
    makedirs()
    train_df, trade_df = load_data(ticker)
    train_arrays = create_torch_array(create_numpy_array(train_df), device)
    trade_arrays = create_torch_array(create_numpy_array(trade_df), device)


    hp = sample_ppo_params(trial)
    cummulative_profit_loss = train(OnlyBuySellEnv, train_arrays, trade_arrays, device, seed, hp)
    log_gpu()

    return cummulative_profit_loss



def main():
    N_STARTUP_TRIALS = 100
    N_TRIALS = 50

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
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
