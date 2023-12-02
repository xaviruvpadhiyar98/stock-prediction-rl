import gymnasium as gym
from stable_baselines3 import A2C, PPO
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
        self.total_reward = 0
        self.good_sell_profit = 0
        self.bad_sell_profit = 0
        self.good_hold_profit = 0
        self.bad_hold_profit = 0

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
                reward += profit

                if profit > 0:
                    self.good_sell += 1
                    self.good_sell_profit += profit
                else:
                    self.bad_sell += 1
                    # reward += (profit * 2)
                    self.bad_sell_profit += profit

                shares_holding = 0
                buy_price = 0                
                self.good_trade += 1
                description = f"{shares_sold} shares sold at {close_price:.2f} with profit of {profit}"

        elif predicted_action == "HOLD":
            self.hold_counter += 1
            if shares_holding == 0:
                description = f"{shares_holding} shares holding."
            else:
                profit = buy_price - (close_price * shares_holding)
                reward += profit

                if profit > 0:
                    h_desc = "GOOD"
                    self.good_hold += 1
                    self.good_hold_profit += profit
                else:
                    h_desc = "BAD"
                    self.bad_hold += 1
                    # reward += (profit * 2)
                    self.bad_hold_profit += profit
                description = f"{h_desc} Holding {shares_holding} shares at {buy_price:.2f} profit of {profit}"
        else:
            raise ValueError(f"{action} should be in [0,1,2]")


        self.total_profit += profit
        self.total_reward += reward
        done = self.counter == (self.length - 1)


        if self.counter > 0:
            if self.good_trade > 0:
                correct_percent = round((self.good_trade / self.counter) * 100, 2)                    
            else:
                correct_percent = 0
            if self.bad_trade > 0:
                wrong_percent = round((self.bad_trade / self.counter) * 100, 2)                    
            else:
                wrong_percent = 0
        else:
            correct_percent = 0
            wrong_percent = 0

        portfolio_value = shares_holding * close_price + available_amount
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
            "correct %": correct_percent,
            "wrong %": wrong_percent,
            "total_profit": self.total_profit,
            "good_hold": self.good_hold,
            "bad_hold": self.bad_hold,
            "good_sell": self.good_sell,
            "bad_sell": self.bad_sell,
            "total_reward": self.total_reward,
            "good_sell_profit": self.good_sell_profit,
            "bad_sell_profit": self.bad_sell_profit,
            "good_hold_profit": self.good_hold_profit,
            "bad_hold_profit": self.bad_hold_profit,
            "portfolio_value": portfolio_value
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
        self.test_and_log()

    def _on_rollout_start(self) -> None:
        pass

    def test_and_log(self) -> None:
        eval_vec_env = make_vec_env(
            StockTradingEnv,
            env_kwargs={"close_prices": EVAL_CLOSE_PRICES},
            n_envs=1,
            seed=1337
        )

        trade_model = PPO('MlpPolicy', eval_vec_env)
        trade_model.set_parameters(self.model.get_parameters())
        episode_rewards, episode_lengths = evaluate_policy(trade_model, eval_vec_env, n_eval_episodes=1, return_episode_rewards=True)
        self.logger.record(f"trade/ep_len", episode_lengths[0])
        self.logger.record(f"trade/ep_reward", episode_rewards[0])



    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        infos = self.locals["infos"]
        sorted_infos = sorted(infos, key=lambda x: x["counter"], reverse=True)
        best_info = sorted_infos[0]
        for k, v in best_info.items():
            self.logger.record(f"info/{k}", v)
        
        self.test_and_log()

    def _on_training_end(self) -> None:
        self.test_and_log()


check_env(StockTradingEnv(CLOSE_PRICES))
model_name = "stock_trading_ppo"
num_envs = 8
vec_env = make_vec_env(
    StockTradingEnv,
    env_kwargs={"close_prices": CLOSE_PRICES},
    n_envs=num_envs,
    seed=1337,
)


if Path(f"trained_models/{model_name}.zip").exists():
    reset_num_timesteps = False
    model = PPO.load(
        f"trained_models/{model_name}.zip",
        vec_env,
        print_system_info=True,
        device="auto",
    )
else:
    reset_num_timesteps = True
    model = PPO(
        "MlpPolicy",
        vec_env,
        # n_steps=128,
        verbose=2,
        device="auto",
        ent_coef=0.05,
        tensorboard_log="tensorboard_log",
    )


model.learn(
    total_timesteps=10_000_000,
    progress_bar=True,
    reset_num_timesteps=reset_num_timesteps,
    callback=EvalCallback(),
    tb_log_name=model_name,
)


model.save("trained_models/" + model_name)
