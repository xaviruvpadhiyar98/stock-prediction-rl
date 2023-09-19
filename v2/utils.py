import numpy as np
import polars as pl
import yfinance as yf
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gymnasium.wrappers.normalize import NormalizeReward
import torch

TICKERS = "SBIN.NS"
INTERVAL = "1h"
PERIOD = "360d"
MODEL_PREFIX = f"{TICKERS}_PPO"

TRAIN_TEST_SPLIT_PERCENT = 0.15
PAST_HOURS = range(1, 15)
TECHNICAL_INDICATORS = [f"PAST_{hour}_HOUR" for hour in PAST_HOURS]
DATASET = Path("datasets")
DATASET.mkdir(parents=True, exist_ok=True)
TRAINED_MODEL_DIR = Path("trained_models")
TENSORBOARD_LOG_DIR = Path("tensorboard_log")


def load_data():
    """returns following dataframe
    shape: (2_508, 3)
    ┌─────────────────────────┬────────────┬─────────┐
    │ Datetime                ┆ Close      ┆ Ticker  │
    │ ---                     ┆ ---        ┆ ---     │
    │ datetime[ns, UTC]       ┆ f64        ┆ str     │
    ╞═════════════════════════╪════════════╪═════════╡
    │ 2022-03-15 03:45:00 UTC ┆ 485.5      ┆ SBIN.NS │
    │ 2022-03-15 04:45:00 UTC ┆ 486.700012 ┆ SBIN.NS │
    │ 2022-03-15 05:45:00 UTC ┆ 488.549988 ┆ SBIN.NS │
    │ 2022-03-15 06:45:00 UTC ┆ 485.049988 ┆ SBIN.NS │
    │ …                       ┆ …          ┆ …       │
    │ 2023-08-25 06:45:00 UTC ┆ 571.099976 ┆ SBIN.NS │
    │ 2023-08-25 07:45:00 UTC ┆ 570.799988 ┆ SBIN.NS │
    │ 2023-08-25 08:45:00 UTC ┆ 569.799988 ┆ SBIN.NS │
    │ 2023-08-25 09:45:00 UTC ┆ 569.700012 ┆ SBIN.NS │
    └─────────────────────────┴────────────┴─────────┘
    """
    ticker_file = DATASET / TICKERS
    if not ticker_file.exists():
        yf.download(
            tickers=TICKERS,
            period=PERIOD,
            interval=INTERVAL,
            group_by="Ticker",
            auto_adjust=True,
            prepost=True,
        ).reset_index().to_parquet(ticker_file, index=False, engine="fastparquet")
    df = pl.read_parquet(ticker_file).select(["Datetime", "Close"])
    df = df.with_columns(pl.lit(TICKERS).alias("Ticker"))
    df = df.sort("Datetime", descending=False)
    return df


def add_past_hours(df):
    """
    shape: (2_494, 17)
    ┌─────────────┬────────────┬─────────┬─────────────┬───┬─────────────┬─────────────┬─────────────┬────────────┐
    │ Datetime    ┆ Close      ┆ Ticker  ┆ PAST_1_HOUR ┆ … ┆ PAST_11_HOU ┆ PAST_12_HOU ┆ PAST_13_HOU ┆ PAST_14_HO │
    │ ---         ┆ ---        ┆ ---     ┆ ---         ┆   ┆ R           ┆ R           ┆ R           ┆ UR         │
    │ datetime[ns ┆ f64        ┆ str     ┆ f64         ┆   ┆ ---         ┆ ---         ┆ ---         ┆ ---        │
    │ , UTC]      ┆            ┆         ┆             ┆   ┆ f64         ┆ f64         ┆ f64         ┆ f64        │
    ╞═════════════╪════════════╪═════════╪═════════════╪═══╪═════════════╪═════════════╪═════════════╪════════════╡
    │ 2022-03-17  ┆ 500.799988 ┆ SBIN.NS ┆ 491.75      ┆ … ┆ 485.049988  ┆ 488.549988  ┆ 486.700012  ┆ 485.5      │
    │ 03:45:00    ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ UTC         ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ 2022-03-17  ┆ 501.450012 ┆ SBIN.NS ┆ 500.799988  ┆ … ┆ 482.950012  ┆ 485.049988  ┆ 488.549988  ┆ 486.700012 │
    │ 04:45:00    ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ UTC         ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ 2022-03-17  ┆ 502.100006 ┆ SBIN.NS ┆ 501.450012  ┆ … ┆ 486.049988  ┆ 482.950012  ┆ 485.049988  ┆ 488.549988 │
    │ 05:45:00    ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ UTC         ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ 2022-03-17  ┆ 501.799988 ┆ SBIN.NS ┆ 502.100006  ┆ … ┆ 485.100006  ┆ 486.049988  ┆ 482.950012  ┆ 485.049988 │
    │ 06:45:00    ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ UTC         ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ …           ┆ …          ┆ …       ┆ …           ┆ … ┆ …           ┆ …           ┆ …           ┆ …          │
    │ 2023-08-25  ┆ 571.099976 ┆ SBIN.NS ┆ 571.549988  ┆ … ┆ 576.900024  ┆ 576.849976  ┆ 577.75      ┆ 573.700012 │
    │ 06:45:00    ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ UTC         ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ 2023-08-25  ┆ 570.799988 ┆ SBIN.NS ┆ 571.099976  ┆ … ┆ 580.700012  ┆ 576.900024  ┆ 576.849976  ┆ 577.75     │
    │ 07:45:00    ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ UTC         ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ 2023-08-25  ┆ 569.799988 ┆ SBIN.NS ┆ 570.799988  ┆ … ┆ 577.900024  ┆ 580.700012  ┆ 576.900024  ┆ 576.849976 │
    │ 08:45:00    ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ UTC         ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ 2023-08-25  ┆ 569.700012 ┆ SBIN.NS ┆ 569.799988  ┆ … ┆ 576.700012  ┆ 577.900024  ┆ 580.700012  ┆ 576.900024 │
    │ 09:45:00    ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ UTC         ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    └─────────────┴────────────┴─────────┴─────────────┴───┴─────────────┴─────────────┴─────────────┴────────────┘
    """  # noqa: E501
    df = df.with_columns(
        [pl.col("Close").shift(hour).alias(f"PAST_{hour}_HOUR") for hour in PAST_HOURS]
    )
    df = df.drop_nulls()
    return df


def train_test_split(df):
    """
    train_df ->
    ┌──────────────┬────────────┬─────────┬─────────────┬───┬──────────────┬──────────────┬──────────────┬──────────────┐
    │ Datetime     ┆ Close      ┆ Ticker  ┆ PAST_1_HOUR ┆ … ┆ PAST_12_HOUR ┆ PAST_13_HOUR ┆ PAST_14_HOUR ┆ Buy/Sold/Hol │
    │ ---          ┆ ---        ┆ ---     ┆ ---         ┆   ┆ ---          ┆ ---          ┆ ---          ┆ d            │
    │ datetime[ns, ┆ f64        ┆ str     ┆ f64         ┆   ┆ f64          ┆ f64          ┆ f64          ┆ ---          │
    │ UTC]         ┆            ┆         ┆             ┆   ┆              ┆              ┆              ┆ f64          │
    ╞══════════════╪════════════╪═════════╪═════════════╪═══╪══════════════╪══════════════╪══════════════╪══════════════╡
    │ 2023-06-07   ┆ 588.700012 ┆ SBIN.NS ┆ 589.099976  ┆ … ┆ 583.400024   ┆ 585.5        ┆ 587.0        ┆ 0.0          │
    │ 09:45:00 UTC ┆            ┆         ┆             ┆   ┆              ┆              ┆              ┆              │
    │ 2023-06-08   ┆ 592.700012 ┆ SBIN.NS ┆ 588.700012  ┆ … ┆ 584.099976   ┆ 583.400024   ┆ 585.5        ┆ 0.0          │
    │ 03:45:00 UTC ┆            ┆         ┆             ┆   ┆              ┆              ┆              ┆              │
    │ 2023-06-08   ┆ 593.25     ┆ SBIN.NS ┆ 592.700012  ┆ … ┆ 584.400024   ┆ 584.099976   ┆ 583.400024   ┆ 0.0          │
    │ 04:45:00 UTC ┆            ┆         ┆             ┆   ┆              ┆              ┆              ┆              │
    │ 2023-06-08   ┆ 589.900024 ┆ SBIN.NS ┆ 593.25      ┆ … ┆ 584.549988   ┆ 584.400024   ┆ 584.099976   ┆ 0.0          │
    │ 05:45:00 UTC ┆            ┆         ┆             ┆   ┆              ┆              ┆              ┆              │
    │ 2023-06-08   ┆ 590.900024 ┆ SBIN.NS ┆ 589.900024  ┆ … ┆ 586.099976   ┆ 584.549988   ┆ 584.400024   ┆ 0.0          │
    │ 06:45:00 UTC ┆            ┆         ┆             ┆   ┆              ┆              ┆              ┆              │
    └──────────────┴────────────┴─────────┴─────────────┴───┴──────────────┴──────────────┴──────────────┴──────────────┘


    trade_df ->
    ┌──────────────┬────────────┬─────────┬─────────────┬───┬──────────────┬──────────────┬──────────────┬──────────────┐
    │ Datetime     ┆ Close      ┆ Ticker  ┆ PAST_1_HOUR ┆ … ┆ PAST_12_HOUR ┆ PAST_13_HOUR ┆ PAST_14_HOUR ┆ Buy/Sold/Hol │
    │ ---          ┆ ---        ┆ ---     ┆ ---         ┆   ┆ ---          ┆ ---          ┆ ---          ┆ d            │
    │ datetime[ns, ┆ f64        ┆ str     ┆ f64         ┆   ┆ f64          ┆ f64          ┆ f64          ┆ ---          │
    │ UTC]         ┆            ┆         ┆             ┆   ┆              ┆              ┆              ┆ f64          │
    ╞══════════════╪════════════╪═════════╪═════════════╪═══╪══════════════╪══════════════╪══════════════╪══════════════╡
    │ 2023-06-08   ┆ 593.099976 ┆ SBIN.NS ┆ 590.900024  ┆ … ┆ 585.299988   ┆ 586.099976   ┆ 584.549988   ┆ 0.0          │
    │ 07:45:00 UTC ┆            ┆         ┆             ┆   ┆              ┆              ┆              ┆              │
    │ 2023-06-08   ┆ 588.549988 ┆ SBIN.NS ┆ 593.099976  ┆ … ┆ 587.200012   ┆ 585.299988   ┆ 586.099976   ┆ 0.0          │
    │ 08:45:00 UTC ┆            ┆         ┆             ┆   ┆              ┆              ┆              ┆              │
    │ 2023-06-08   ┆ 588.5      ┆ SBIN.NS ┆ 588.549988  ┆ … ┆ 588.75       ┆ 587.200012   ┆ 585.299988   ┆ 0.0          │
    │ 09:45:00 UTC ┆            ┆         ┆             ┆   ┆              ┆              ┆              ┆              │
    │ 2023-06-09   ┆ 584.700012 ┆ SBIN.NS ┆ 588.5       ┆ … ┆ 588.0        ┆ 588.75       ┆ 587.200012   ┆ 0.0          │
    │ 03:45:00 UTC ┆            ┆         ┆             ┆   ┆              ┆              ┆              ┆              │
    │ 2023-06-09   ┆ 580.900024 ┆ SBIN.NS ┆ 584.700012  ┆ … ┆ 590.0        ┆ 588.0        ┆ 588.75       ┆ 0.0          │
    │ 04:45:00 UTC ┆            ┆         ┆             ┆   ┆              ┆              ┆              ┆              │
    └──────────────┴────────────┴─────────┴─────────────┴───┴──────────────┴──────────────┴──────────────┴──────────────┘
    """  # noqa: E501
    total = df.shape[0]
    train_size = total - int(total * TRAIN_TEST_SPLIT_PERCENT)
    train_df = df.slice(0, train_size)
    trade_df = df.slice(train_size, total)
    return train_df, trade_df


def create_numpy_array(df):
    """
    returns array
    [
        [593.09997559 590.90002441 589.90002441 ... 586.09997559 584.54998779
            0.        ]
        [588.54998779 593.09997559 590.90002441 ... 585.29998779 586.09997559
            0.        ]
        [588.5        588.54998779 593.09997559 ... 587.20001221 585.29998779
            0.        ]
        ...
        [570.79998779 571.09997559 571.54998779 ... 576.84997559 577.75
            0.        ]
        [569.79998779 570.79998779 571.09997559 ... 576.90002441 576.84997559
            0.        ]
        [569.70001221 569.79998779 570.79998779 ... 580.70001221 576.90002441
            0.        ]
    ]
    """
    cols = df.columns
    cols.remove("Datetime")
    cols.remove("Ticker")
    df = df.unique("Datetime", maintain_order=True)
    df = df.select(cols)
    arr = []
    [arr.append(row) for row in df.iter_rows()]
    return np.asarray(arr)


def create_torch_array(df, device):
    arr = create_numpy_array(df)
    arr = np.asarray(arr).astype(np.float32)
    return torch.from_numpy(arr).to(device)


def make_env(env_id, array, tickers, seed):
    def thunk():
        env = env_id(array, [tickers], seed)
        return env

    return thunk


def test_model(env, model, seed):
    obs, _ = env.reset(seed=seed)
    while True:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        if done:
            return info


class TensorboardCallback(BaseCallback):
    def __init__(self, save_freq: int, model_prefix: str, eval_env: Monitor, seed: int):
        self.save_freq = save_freq
        self.model_prefix = model_prefix
        self.eval_env = eval_env
        self.seed = seed
        super().__init__()

    def log(self, info, key):
        unnecessary_keys = ["TimeLimit.truncated", "terminal_observation", "episode"]
        for k, v in info.items():
            if k not in unnecessary_keys:
                self.logger.record(f"{key}/{k}", v)

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        if "episode" in info:
            self.log(info, key="train")

        if (self.n_calls > 0) and (self.n_calls % self.save_freq) == 0:
            info = test_model(self.eval_env, self.model, seed=self.seed)
            self.log(info, key="trade")
            trade_holdings = int(info["holdings"])

            model_path = Path(TRAINED_MODEL_DIR) / self.model_prefix
            available_model_files = list(model_path.rglob("*.zip"))
            available_model_holdings = [
                int(f.stem.split("-")[-1]) for f in available_model_files
            ]
            available_model_holdings.sort()

            if not available_model_holdings:
                model_filename = model_path / f"{trade_holdings}.zip"
                self.model.save(model_filename)
                print(f"Saving model checkpoint to {model_filename}")

            else:
                if trade_holdings > available_model_holdings[0]:
                    file_to_remove = model_path / f"{available_model_holdings[0]}.zip"
                    file_to_remove.unlink()
                    model_filename = model_path / f"{trade_holdings}.zip"
                    self.model.save(model_filename)
                    print(f"Removed {file_to_remove} and Added {model_filename} file.")

            # periodic save model for continue training later
            self.model.save(Path(TRAINED_MODEL_DIR) / f"{MODEL_PREFIX}.zip")
        return True


def get_ppo_model(env, seed):
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=5e-4,
        n_steps=4096,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.3,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        stats_window_size=100,
        tensorboard_log=TENSORBOARD_LOG_DIR,
        policy_kwargs=dict(
            net_arch=[64, 64],
        ),
        verbose=0,
        seed=seed,
        device="auto",
        _init_setup_model=True,
    )
    return model


def load_ppo_model(env):
    model_file = Path(TRAINED_MODEL_DIR) / f"{MODEL_PREFIX}.zip"
    model = PPO.load(
        model_file,
        env,
        verbose=0,
        tensorboard_log=TENSORBOARD_LOG_DIR,
    )
    return model
