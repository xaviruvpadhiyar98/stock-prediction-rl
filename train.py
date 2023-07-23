from stock_trading_env import StockTradingEnv
from config import (
    NUMPY_FILENAME,
    STOCK_DATA_SAVE_DIR,
    TICKERS,
    TECHNICAL_INDICATORS,
    SEED,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
)
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from random import seed as random_seed
from torch import manual_seed
from pathlib import Path
from logger_config import train_logger as log

# TRY NOT TO MODIFY: seeding
random_seed(SEED)
np.random.seed(SEED)
manual_seed(SEED)
BEST_HOLDING = 0


class TensorboardCallback(BaseCallback):
    def __init__(self, save_freq: int, model_prefix: str, eval_env: Monitor):
        self.save_freq = save_freq
        self.model_prefix = model_prefix
        self.eval_env = eval_env
        super().__init__()

    def _on_step(self) -> bool:
        self.logger.record(
            key="train/holdings",
            value=self.locals["infos"][0]["holdings"],
        )
        self.logger.record(key="train/reward", value=self.locals["rewards"][0])
        self.logger.record(key="train/n_calls", value=self.n_calls)
        shares = self.locals["infos"][0]["shares"]
        for k, v in shares.items():
            self.logger.record(key=f"train/shares/{k}", value=v)

        if (self.n_calls > 0) and (self.n_calls % self.save_freq) == 0:
            obs, _ = self.eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs)
                obs, reward, done, truncated, info = self.eval_env.step(action)

            model_path = Path(TRAINED_MODEL_DIR) / self.model_prefix
            available_model_files = list(model_path.rglob("*.zip"))
            available_model_holdings = [
                int(f.stem.split("-")[-1]) for f in available_model_files
            ]
            available_model_holdings.sort()
            trade_holdings = int(info["holdings"])
            self.logger.record(key="trade/holdings", value=trade_holdings)

            if not available_model_holdings:
                model_filename = model_path / f"{trade_holdings}.zip"
                self.model.save(model_filename)
                log.info(f"Saving model checkpoint to {model_filename}")

            elif len(available_model_holdings) < 5:
                model_filename = model_path / f"{trade_holdings}.zip"
                self.model.save(model_filename)
                log.info(f"Saving model checkpoint to {model_filename}")

            else:
                if trade_holdings > available_model_holdings[0]:
                    file_to_remove = model_path / f"{available_model_holdings[0]}.zip"
                    file_to_remove.unlink()
                    model_filename = model_path / f"{trade_holdings}.zip"
                    self.model.save(model_filename)
                    log.info(
                        f"Removed {file_to_remove} and Added {model_filename} file."
                    )
        return True


def train():
    with open(f"{STOCK_DATA_SAVE_DIR}/{NUMPY_FILENAME}", "rb") as f:
        train_arrays = np.load(f)
        trade_arrays = np.load(f)

    Path(TRAINED_MODEL_DIR).mkdir(parents=True, exist_ok=True)

    train_env = Monitor(StockTradingEnv(train_arrays, TICKERS, TECHNICAL_INDICATORS))
    trade_env = Monitor(StockTradingEnv(trade_arrays, TICKERS, TECHNICAL_INDICATORS))

    MODEL_NAME = "ppo"
    IDENTIFIER = "only-close-price-rsi-14-emi-8-emi-21-past-hours-htTrendline"
    MODEL_PREFIX = f"{MODEL_NAME}/{IDENTIFIER}"
    TOTAL_TIMESTAMP = 500_000
    tensorboard_log = Path(f"{TENSORBOARD_LOG_DIR}/{MODEL_NAME}")
    model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log=tensorboard_log)
    model.learn(
        total_timesteps=TOTAL_TIMESTAMP,
        callback=TensorboardCallback(
            save_freq=4096, model_prefix=MODEL_PREFIX, eval_env=trade_env
        ),
        tb_log_name=f"ppo-{TOTAL_TIMESTAMP}-{IDENTIFIER}",
    )
    obs, _ = trade_env.reset()
    for i in range(len(trade_arrays)):
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = trade_env.step(action)
    print(info)


if __name__ == "__main__":
    train()
