from pathlib import Path
import yfinance as yf
import polars as pl
from talib import *
from talib import PPO as ta_PPO
import numpy as np
from datetime import datetime, timezone, timedelta
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
import torch.nn as nn
import psutil
from subprocess import run, PIPE
import json


def makedirs():
    dirs = ["datasets", "trained_models", "tensorboard_log"]
    for dir in dirs:
        Path(dir).mkdir(parents=True, exist_ok=True)


def load_data(ticker="SBIN.NS"):
    datasets = Path("datasets")
    period = "360d"
    interval = "1h"
    train_test_split_percent = 0.15
    past_hours = range(1, 30)
    past_actions = range(1, 30)
    ticker_file = datasets / ticker
    if not ticker_file.exists():
        (
            yf.download(
                tickers=ticker,
                period=period,
                interval=interval,
                group_by="Ticker",
                auto_adjust=True,
                prepost=True,
            )
            .reset_index()
            .to_parquet(ticker_file, index=False, engine="fastparquet")
        )
    df = (
        pl.read_parquet(ticker_file)
        .select(["Datetime", "Close", "High", "Low"])
        .with_columns(pl.lit(ticker).alias("Ticker"))
        .sort("Datetime", descending=False)
        .with_columns(
            [
                pl.col("Close").shift(hour).alias(f"Past{hour}Hour")
                for hour in past_hours
            ]
        )
        .with_columns(
            RSI=pl.map_groups(pl.col("Close"), lambda x: RSI(x[0], timeperiod=14)),
            EMA9=pl.map_groups(pl.col("Close"), lambda x: EMA(x[0], timeperiod=9)),
            EMA21=pl.map_groups(pl.col("Close"), lambda x: EMA(x[0], timeperiod=21)),
            MACD=pl.map_groups(
                pl.col("Close"),
                lambda x: MACD(x[0], fastperiod=12, slowperiod=26, signalperiod=9)[0],
            ),  # MACD line
            MACD_SIGNAL=pl.map_groups(
                pl.col("Close"),
                lambda x: MACD(x[0], fastperiod=12, slowperiod=26, signalperiod=9)[1],
            ),  # Signal line
            BBANDS_UPPER=pl.map_groups(
                pl.col("Close"), lambda x: BBANDS(x[0], timeperiod=20)[0]
            ),  # Upper Bollinger Band
            BBANDS_MIDDLE=pl.map_groups(
                pl.col("Close"), lambda x: BBANDS(x[0], timeperiod=20)[1]
            ),  # Middle Bollinger Band
            BBANDS_LOWER=pl.map_groups(
                pl.col("Close"), lambda x: BBANDS(x[0], timeperiod=20)[2]
            ),  # Lower Bollinger Band
            ADX=pl.map_groups(
                (pl.col("High"), pl.col("Low"), pl.col("Close")),
                lambda x: ADX(x[0], x[1], x[2], timeperiod=14),
            ),
            STOCH_K=pl.map_groups(
                (pl.col("High"), pl.col("Low"), pl.col("Close")),
                lambda x: STOCH(x[0], x[1], x[2])[0],
            ),  # %K line
            STOCH_D=pl.map_groups(
                (pl.col("High"), pl.col("Low"), pl.col("Close")),
                lambda x: STOCH(x[0], x[1], x[2])[1],
            ),  # %D line
            ATR=pl.map_groups(
                (pl.col("High"), pl.col("Low"), pl.col("Close")),
                lambda x: ATR(x[0], x[1], x[2], timeperiod=14),
            ),
            CCI=pl.map_groups(
                (pl.col("High"), pl.col("Low"), pl.col("Close")),
                lambda x: CCI(x[0], x[1], x[2], timeperiod=14),
            ),
            MOM=pl.map_groups(pl.col("Close"), lambda x: MOM(x[0], timeperiod=10)),
            ROC=pl.map_groups(pl.col("Close"), lambda x: ROC(x[0], timeperiod=10)),
            WILLR=pl.map_groups(
                (pl.col("High"), pl.col("Low"), pl.col("Close")),
                lambda x: WILLR(x[0], x[1], x[2], timeperiod=14),
            ),
            PPO=pl.map_groups(
                pl.col("Close"), lambda x: ta_PPO(x[0], fastperiod=12, slowperiod=26)
            ),
        )
        # Previous action = Buy = 2, HOLD = 1, SELL = 0
        .with_columns([pl.lit(1).alias(f"Previous{n}Action") for n in past_actions])
        .with_columns(
            pl.lit(10000).alias("PortfolioValue"),
            pl.lit(10000).alias("AvailableAmount"),
            pl.lit(0).alias("SharesHolding"),
            pl.lit(0).alias("CummulativeProfitLoss"),
        )
        .drop_nulls()
        .unique("Datetime")
    )
    df = pl.from_pandas(df.to_pandas().dropna()).sort("Datetime")

    total = df.shape[0]
    train_size = total - int(total * train_test_split_percent)
    train_df = df.slice(0, train_size).sort("Datetime")
    trade_df = df.slice(train_size, total).sort("Datetime")
    cols = train_df.columns
    print(train_df)
    print(trade_df)
    print(cols)
    return train_df, trade_df


def create_numpy_array(df):
    cols = df.columns
    cols.remove("Datetime")
    cols.remove("Ticker")
    df = df.select(cols)
    arr = []
    [arr.append(row) for row in df.iter_rows()]
    return np.asarray(arr)


def make_env(env_id, array, mode, seed, rank):
    def thunk():
        env = Monitor(env_id(array, mode, seed=seed + rank))
        env.reset(seed=seed + rank)
        return env

    return thunk


def create_envs(env, array, mode, num_envs=1, seed=1337):
    envs = DummyVecEnv([make_env(env, array, mode, seed, i) for i in range(num_envs)])
    return envs


def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def get_ppo_model(env, seed):
    """
    [I 2023-10-08 11:08:05,519] Trial 6 finished with value: 43.45013427734375 and parameters: {'batch_size': 128, 'n_steps': 512, 'gamma': 0.95, 'learning_rate': 1.9341219418904578e-05, 'lr_schedule': 'constant', 'ent_coef': 1.1875984002464866e-06, 'clip_range': 0.2, 'n_epochs': 20, 'gae_lambda': 1.0, 'max_grad_norm': 2, 'vf_coef': 0.029644396080155226, 'net_arch': 'small', 'ortho_init': True, 'activation_fn': 'relu'}. Best is trial 6 with value: 43.45013427734375.
    """
    tensorboard_log = Path("tensorboard_log")

    model = PPO(
        "MlpPolicy",
        env,
        # learning_rate=linear_schedule(9.2458929157504e-05),
        learning_rate=3.7141262285419446e-05,
        n_steps=32,
        batch_size=8,
        n_epochs=5,
        gamma=0.95,
        gae_lambda=1.0,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=1.1875984002464866e-06,
        vf_coef=0.029644396080155226,
        max_grad_norm=0.8,
        tensorboard_log=tensorboard_log,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=nn.Tanh,
            ortho_init=True,
        ),
        verbose=0,
        seed=seed,
        device="auto",
        _init_setup_model=True,
    )
    return model

def get_default_ppo_model(env, seed):
    """
    [I 2023-10-08 11:08:05,519] Trial 6 finished with value: 43.45013427734375 and parameters: {'batch_size': 128, 'n_steps': 512, 'gamma': 0.95, 'learning_rate': 1.9341219418904578e-05, 'lr_schedule': 'constant', 'ent_coef': 1.1875984002464866e-06, 'clip_range': 0.2, 'n_epochs': 20, 'gae_lambda': 1.0, 'max_grad_norm': 2, 'vf_coef': 0.029644396080155226, 'net_arch': 'small', 'ortho_init': True, 'activation_fn': 'relu'}. Best is trial 6 with value: 43.45013427734375.
    """
    tensorboard_log = Path("tensorboard_log")

    model = PPO(
        "MlpPolicy",
        env,
        # learning_rate=linear_schedule(0.00001),
        learning_rate=0.00001,
        # learning_rate=3.7141262285419446e-05,
        n_steps=256,
        batch_size=32,
        n_epochs=5,
        # gamma=0.95,
        # gae_lambda=1.0,
        clip_range=0.3,
        # clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.3,
        # vf_coef=0.01,
        max_grad_norm=0.8,
        tensorboard_log=tensorboard_log,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=nn.Tanh,
            ortho_init=True,
        ),
        verbose=0,
        seed=seed,
        device="auto",
        _init_setup_model=True,
    )
    return model


def test_model(env, model, seed):
    env.env.seed = seed
    obs, _ = env.reset(seed=seed)
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            env.close()
            return info


class TensorboardCallback(BaseCallback):
    """ """

    def __init__(self, eval_envs: Monitor, train_ending_index: int):
        super().__init__()
        self.eval_envs = eval_envs
        self.train_ending_index = train_ending_index

    def log(self, info, key):
        unnecessary_keys = ["TimeLimit.truncated", "terminal_observation", "episode"]
        for k, v in info.items():
            self.logger.record(f"{key}/{k}", v, unnecessary_keys)

    def log_gpu(self):
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
        self.log(info, "gpu")

    def log_cpu(self):
        cpu_percent = psutil.cpu_percent()
        memory_usage_percent = psutil.virtual_memory().percent
        info = {
            "utilization": cpu_percent,
            "memory": memory_usage_percent,
        }
        self.log(info, "cpu")

    def log_best_env(self, ending_infos):
        sorted_env = sorted(
            ending_infos, key=lambda x: x["cummulative_profit_loss"], reverse=True
        )
        best_env_info = sorted_env[0]
        best_env_info["env"] = "train"
        best_env_info["iteration"] = self.locals["iteration"]
        best_env_info["n_calls"] = self.n_calls
        print(json.dumps(best_env_info, indent=4, default=str))
        self.log(best_env_info, key="train")
        return best_env_info["seed"]

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        # print(json.dumps(infos[0], indent=4, sort_keys=True, default=str))

        ending_infos = [
            info for info in infos if info["index"] == self.train_ending_index
        ]
        if not ending_infos:
            return True

        self.log_gpu()
        self.log_cpu()
        best_env_id = self.log_best_env(ending_infos)

        trade_model = get_ppo_model(env=self.eval_envs, seed=best_env_id)
        parameters = self.model.get_parameters()
        trade_model.set_parameters(parameters)

        for env in self.eval_envs.envs:
            if env.seed == best_env_id:
                eval_env = env

        obs, t_info = eval_env.reset(seed=best_env_id)
        while True:
            action, _ = trade_model.predict(obs, deterministic=True)
            obs, reward, done, truncated, t_info = eval_env.step(action.item())
            if done or truncated:
                eval_env.close()
                break

        # t_info = test_model(self.eval_env, trade_model, best_env_id)
        t_info["env"] = "trade"
        print(json.dumps(t_info, indent=4, default=str))
        self.log(t_info, key="trade")
        return True


def sample_ppo_params(trial):
    """
    Sampler for PPO hyperparams.

    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    n_steps = trial.suggest_categorical(
        "n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    )
    gamma = trial.suggest_categorical(
        "gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
    )
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    lr_schedule = "constant"

    # Uncomment to enable learning rate schedule
    lr_schedule = trial.suggest_categorical("lr_schedule", ["linear", "constant"])
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    gae_lambda = trial.suggest_categorical(
        "gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0]
    )
    max_grad_norm = trial.suggest_categorical(
        "max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5]
    )
    vf_coef = trial.suggest_float("vf_coef", 0.01, 1, log=True)
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])

    # Uncomment for gSDE (continuous actions)
    # log_std_init = trial.suggest_uniform("log_std_init", -4, 1)
    # Uncomment for gSDE (continuous action)
    # sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])

    # Orthogonal initialization
    # ortho_init = False
    ortho_init = trial.suggest_categorical("ortho_init", [False, True])
    activation_fn = trial.suggest_categorical(
        "activation_fn", ["tanh", "relu", "elu", "leaky_relu"]
    )
    # activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    # TODO: account when using multiple envs
    if batch_size > n_steps:
        batch_size = n_steps

    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)

    # Independent networks usually work best
    # when not working with images
    net_arch = {
        "small": dict(pi=[64, 64], vf=[64, 64]),
        "medium": dict(pi=[256, 256], vf=[256, 256]),
    }[net_arch]

    activation_fn = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }[activation_fn]

    return {
        "policy": "MlpPolicy",
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        # "sde_sample_freq": sde_sample_freq,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
    }


if __name__ == "__main__":
    ...
