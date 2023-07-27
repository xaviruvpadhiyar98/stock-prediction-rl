from pathlib import Path
from config import TRAINED_MODEL_DIR, STOCK_DATA_SAVE_DIR, FILENAME, SEED, TICKERS
import pandas as pd
from random import seed as random_seed
import numpy as np
from torch import manual_seed
from torch.cuda import manual_seed as cuda_seed
from logger_config import eval_logger as log
from typing import List, Tuple
from train import load_df, add_features, clean_df, split_train_test
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stock_trading_env import StockTradingEnv

# TRY NOT TO MODIFY: seeding
random_seed(SEED)
np.random.seed(SEED)
manual_seed(SEED)
cuda_seed(SEED)




saved_models = Path(TRAINED_MODEL_DIR).rglob("*.zip")
results = []
for saved_model in saved_models:
    log.info(f"Using Model - {saved_model}")
    df = load_df()
    past_hour = tuple(saved_model.parent.stem.split("_")[1].split('-'))
    past_hour = [int(ph) for ph in past_hour if ph]
    log.info(f"Using Past Hours - {past_hour}")
    df, feature_columns = add_features(df, past_hour)
    log.info(f"Starting with {past_hour} indicators and {feature_columns}")
    df = clean_df(df)
    train_arrays, trade_arrays = split_train_test(
        df, technical_indicators=feature_columns
    )
    model = PPO.load(saved_model)
    trade_env = Monitor(StockTradingEnv(trade_arrays, TICKERS, feature_columns))
    obs, _ = trade_env.reset()
    for i in range(len(trade_arrays)):
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = trade_env.step(action)
        result = {
            "model": saved_model.as_posix(),
            "holdings": info["holdings"],
            "reward": info["reward"]
        }
        result.update(info["shares"])
        results.append(result)
        log.info(f"Result of Trade env\n{info}")

results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame(results).to_parquet(results_dir/"all-model-eval-results.parquet", index=False, engine="fastparquet")