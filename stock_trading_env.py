from collections import deque
from gymnasium import Env, spaces
import numpy as np


class StockTradingEnv(Env):
    HMAX = 10
    INITIAL_AMOUNT = 100_000
    BUY_COST = SELL_COST = 0.001

    def __init__(self, arrays, tickers, features):
        self.arrays = arrays
        self.tickers = tickers
        self.features = features
        self.num_of_features = len(features)
        self.num_of_tickers = len(tickers)
        self.action_space = spaces.Box(
            -1, 1, shape=(self.num_of_tickers,), dtype=np.int32
        )
        self.obs_formula = 1 + (self.num_of_tickers * (1 + self.num_of_features + 1))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_formula,),
            dtype=np.float32,
        )
        self.HOLDINGS = deque([self.INITIAL_AMOUNT], 2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reward = 0.0
        self.info = {}
        self.state = self.generate_state(reset=True)
        return self.state, self.info

    def step(self, actions):
        actions = np.rint(actions)
        done = bool(self.index == len(self.arrays) - 1)
        if done:
            truncated = False
            return (self.state, self.reward, done, truncated, self.info)

        for ticker_index, action in enumerate(actions):
            if action == 0:
                continue

            close_price_index = 1 + ticker_index * (1 + self.num_of_features + 1)
            price_per_share = self.state[close_price_index]
            number_of_shares_index = (
                1
                + ticker_index * (1 + self.num_of_features + 1)
                + (1 + self.num_of_features)
            )
            if action == 1.0:
                self.buy(price_per_share, number_of_shares_index, ticker_index)
            if action == -1.0:
                self.sell(price_per_share, number_of_shares_index, ticker_index)

        holdings, shares_available = self.get_holdings()
        self.HOLDINGS.append(holdings)
        self.reward = self.calculate_reward()
        share = {x: y for x, y in zip(self.tickers, shares_available)}
        self.state = self.generate_state()
        # self.REWARDS.append(self.reward)
        # self.SHARES.append(share)
        # self.ACTIONS.append(actions)
        # self.info = {
        #     "currentHoldings": self.HOLDINGS,
        #     "shares": self.SHARES,
        #     "rewards": self.REWARDS,
        #     "actions": self.ACTIONS
        # }
        self.info = {
            "holdings": holdings,
            "shares": share,
            "reward": self.reward,
            "action": actions,
        }
        truncated = False
        return (self.state, self.reward, done, truncated, self.info)

    def buy(self, price_per_share, number_of_shares_index, ticker_index):
        shares = min(self.state[0] // price_per_share, self.HMAX)
        buy_prices_with_commission = (price_per_share * (1 + self.BUY_COST)) * shares
        self.state[0] -= buy_prices_with_commission
        self.state[number_of_shares_index] += shares

    def sell(self, price_per_share, number_of_shares_index, ticker_index):
        shares = self.state[number_of_shares_index]
        if shares > 0:
            shares = min(shares, self.HMAX)
            sell_prices_with_commission = (
                price_per_share * (1 + self.SELL_COST)
            ) * shares
            self.state[0] += sell_prices_with_commission
            self.state[number_of_shares_index] -= shares

    def get_holdings(self):
        close_prices = self.state[1 :: 1 + self.num_of_features + 1]
        shares_available = self.state[
            1 + 1 + self.num_of_features :: 1 + self.num_of_features + 1
        ]

        holdings = np.sum(np.multiply(close_prices, shares_available)) + self.state[0]
        return holdings, shares_available

    def generate_state(self, reset=False):
        if not reset:
            self.index += 1
        else:
            self.index = 0

        vals = self.arrays[self.index].reshape(-1)
        state = np.array([self.INITIAL_AMOUNT])
        state = np.append(state, vals)

        if reset:
            return state

        return self.update_new_states_with_old_values(self.state, state)

    def update_new_states_with_old_values(self, old_state, new_state):
        old_state = self.state
        new_state[0] = old_state[0]

        start = 1 + 1 + self.num_of_features
        end = 1 + self.num_of_features + 1

        new_state[start::end] = old_state[start::end]

        return new_state

    def calculate_reward(self):
        change_in_holdings = self.HOLDINGS[-2] - self.HOLDINGS[-1]
        if change_in_holdings > 0:
            return change_in_holdings * 0.1
        return change_in_holdings * -0.2


if __name__ == "__main__":
    from config import (
        NUMPY_FILENAME,
        STOCK_DATA_SAVE_DIR,
        TICKERS,
        TECHNICAL_INDICATORS,
    )
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    with open(f"{STOCK_DATA_SAVE_DIR}/{NUMPY_FILENAME}", "rb") as f:
        train_arrays = np.load(f)
        trade_arrays = np.load(f)

    train_env = Monitor(StockTradingEnv(train_arrays, TICKERS, TECHNICAL_INDICATORS))
    trade_env = Monitor(StockTradingEnv(trade_arrays, TICKERS, TECHNICAL_INDICATORS))
    model = PPO("MlpPolicy", train_env, verbose=1)

    for _ in range(3):
        obs, _ = trade_env.reset()
        for i in range(len(trade_arrays)):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = trade_env.step(action)
        print(info)

    model.learn(total_timesteps=10_000)

    obs, _ = trade_env.reset()
    for i in range(len(trade_arrays)):
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = trade_env.step(action)
    print(info)
