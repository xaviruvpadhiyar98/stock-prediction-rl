import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from gymnasium.spaces import Dict, Discrete, Box, MultiDiscrete
from pathlib import Path
import polars as pl

ACTION_MAP = {0: "HOLD", 1: "BUY", 2: "SELL"}
TICKER = "SBIN.NS"
TRAIN_FILE = Path("datasets") / f"{TICKER}_train"
EVAL_FILE = Path("datasets") / f"{TICKER}_trade"

CLOSE_PRICES = pl.read_parquet(TRAIN_FILE)["Close"].to_numpy()
EVAL_CLOSE_PRICES = pl.read_parquet(EVAL_FILE)["Close"].to_numpy()


class StockTradingEnv(gym.Env):
    """
    Observations
        [Close_Price, Available_Amount, Shares_Holdings, Buy_Price]
        Close_Price -
        Available_Amount -
        Shares_Holdings -
        Buy_Price -
    Actions
        [HOLD, BUY, SELL]
        [0, 1, 2]
    """

    def __init__(self, close_prices):
        super().__init__()

        self.close_prices = close_prices
        low = np.min(close_prices)
        high = np.max(close_prices) * 10
        self.length = len(self.close_prices)

        self.observation_space = Box(
            low=np.array([0, 0, 0, 0], np.float32),
            high=np.array([high, 30_000, 1000, 30_000], np.float32),
            shape=(4,),
            dtype=np.float32,
        )
        self.action_space = Discrete(3)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.seed = seed
        self.counter = 0
        self.good_trade = 0
        self.bad_trade = 0
        self.buy_counter = 0
        self.sell_counter = 0
        self.hold_counter = 0
        self.total_profit = 0
        self.good_hold = 0
        self.bad_hold = 0
        self.good_buy = 0
        self.bad_buy = 0
        self.good_sell = 0
        self.bad_sell = 0

        close_price = self.close_prices[self.counter]
        available_amount = 10_000
        shares_holding = 0
        buy_price = 0
        self.state = np.array(
            [close_price, available_amount, shares_holding, buy_price], dtype=np.float32
        )
        return self.state, {}

    def step(self, action):
        reward = 0
        profit = 0
        shares_bought = 0
        shares_sold = 0
        truncated = False
        close_price = self.state[0]
        available_amount = self.state[1]
        shares_holding = self.state[2]
        buy_price = self.state[3]
        description = ""

        predicted_action = ACTION_MAP[action]

        if predicted_action == "BUY":
            self.buy_counter += 1
            if close_price > available_amount:
                reward -= 50_000
                self.bad_trade += 1
                description = f"{close_price} > {available_amount}. Cannot Buy Shares"
                truncated = True

            else:
                shares_bought = available_amount // close_price

                buy_price = close_price * shares_bought
                shares_holding += shares_bought
                available_amount -= buy_price

                reward += shares_bought
                self.good_trade += 1
                description = f"{shares_bought} shares bought at {close_price:.2f}"

        elif predicted_action == "SELL":
            self.sell_counter += 1
            if shares_holding == 0:
                reward -= 50_000
                self.bad_trade += 1
                description = f"{shares_holding} shares available. Cannot Sell Shares"
                truncated = True
            else:
                shares_sold = shares_holding
                sell_price = close_price * shares_holding
                available_amount += sell_price
                profit = sell_price - buy_price
                if profit > 0:
                    self.good_sell += 1
                else:
                    self.bad_sell += 1

                shares_holding = 0
                buy_price = 0

                reward += profit
                self.good_trade += 1
                description = f"{shares_sold} shares sold at {close_price:.2f} with profit of {profit}"

        elif predicted_action == "HOLD":
            self.hold_counter += 1
            if shares_holding == 0:
                description = f"{shares_holding} shares holding."
            else:
                diff = buy_price - (close_price * shares_holding)
                reward += diff 
                if diff > 0:
                    h_desc = "GOOD"
                    self.good_hold += 1
                else:
                    h_desc = "BAD"
                    self.bad_hold += 1
                description = f"{h_desc} Holding {shares_holding} shares at {buy_price:.2f} {diff=}"
        else:
            raise ValueError(f"{action} should be in [0,1,2]")


        self.total_profit += profit
        done = self.counter == (self.length - 1)
        info = {
            "seed": self.seed,
            "counter": self.counter,
            "close_price": close_price,
            "predicted_action": predicted_action,
            "description": description,
            "available_amount": available_amount,
            "shares_holdings": shares_holding,
            "buy_price": buy_price,
            "reward": reward,
            "done": done,
            "truncated": truncated,
            "good_trade": self.good_trade,
            "bad_trade": self.bad_trade,
            "profit": profit,
            "shares_sold": shares_sold,
            "shares_bought": shares_bought,
            "buy_counter": self.buy_counter,
            "sell_counter": self.sell_counter,
            "hold_counter": self.hold_counter,
            "correct %": round(((self.good_trade + 1) / (self.counter + 1)) * 100, 2),
            "wrong %": round(((self.bad_trade + 1) / (self.counter + 1)) * 100, 2),
            "total_profit": self.total_profit,
            "good_hold": self.good_hold,
            "bad_hold": self.bad_hold,
            "good_sell": self.good_sell,
            "bad_sell": self.bad_sell
        }

        if done or truncated:
            return self.state, float(reward), done, truncated, info

        self.counter += 1
        close_price = self.close_prices[self.counter]
        self.state = np.array(
            [close_price, available_amount, shares_holding, buy_price], dtype=np.float32
        )
        return self.state, float(reward), done, truncated, info

    def close(self):
        pass


class EvalCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        pass

    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        infos = self.locals["infos"]
        sorted_infos = sorted(infos, key=lambda x: x["counter"], reverse=True)
        best_info = sorted_infos[0]
        for k, v in best_info.items():
            self.logger.record(f"info/{k}", v)

    def _on_training_end(self) -> None:
        pass


check_env(StockTradingEnv(CLOSE_PRICES))
model_name = "stock_trading_a2c"
num_envs = 64
vec_env = VecNormalize(
    make_vec_env(
        StockTradingEnv,
        env_kwargs={"close_prices": CLOSE_PRICES},
        n_envs=num_envs,
        seed=1337,
    ),
    training=True,
)
eval_vec_env = (
    make_vec_env(
        StockTradingEnv,
        env_kwargs={"close_prices": EVAL_CLOSE_PRICES},
        n_envs=num_envs,
        seed=1337,
    )
    # training=False,
)

model = A2C.load(
    f"trained_models/{model_name}.zip",
    vec_env,
    print_system_info=True,
    device="cpu",
)

results = []
obs = eval_vec_env.reset()
print(obs)
# raise
counter = 0
while counter < num_envs:
    action, _ = model.predict(obs, deterministic=False)
    obs, rewards, dones, infos = eval_vec_env.step(action)
    results.append(infos)
    for i in range(len(infos)):
        # result = infos[i].copy()
        # print(result)
        # raise
        # results.append(result)
        if dones[i]:
            # print(infos[i]['counter'], infos[i]['total_profit'], infos[i]['predicted_action'])
            counter += 1


# mean_reward, _ = evaluate_policy(model, eval_vec_env, n_eval_episodes=10)
# print(f"Before Learning Mean reward: {mean_reward}")



# mean_reward, _ = evaluate_policy(model, eval_vec_env, n_eval_episodes=10)
# print(f"After Learning Mean reward: {mean_reward}")

for result in results:
    print(result[0])
    # print(results.append(result))