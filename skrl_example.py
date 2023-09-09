# import the environment wrapper and gym
from skrl.envs.wrappers.torch import wrap_env
import numpy as np
from collections import deque
import torch
from gymnasium import Env, spaces

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StockTradingEnv(Env):
    HMAX = 1
    AMOUNT = torch.Tensor([10_000]).to(DEVICE)
    BUY_COST = SELL_COST = 0.001

    def __init__(self, arrays, tickers):
        self.arrays = arrays
        self.tickers = tickers
        self.action_space = spaces.Box(-1, 1, shape=(len(tickers),), dtype=np.int8)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1 + len(arrays[0]),),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.index = 0
        self.reward = 0.0
        self.info = {}
        self.HOLDINGS = deque([self.AMOUNT], 2)
        self.state = self.generate_state(reset=True)
        self.tracking_buy_sell = []

        return self.state, self.info

    def step(self, actions):
        actions = actions[0].round().item()
        done = bool(self.index == len(self.arrays) - 1)
        if done:
            truncated = False
            return (self.state, self.reward, done, truncated, self.info)
        self.index += 1

        if actions > 0:
            self.buy()
        elif actions < 0:
            self.sell()
        else:
            self.hold()

            # print(f"Holding {self.state[-1]} Available Amount {self.state[0]:.2f}")

        holdings = self.get_holdings()
        self.HOLDINGS.append(holdings)
        # self.reward = self.calculate_reward()
        self.state = self.generate_state()
        self.info = {
            "holdings": holdings,
            "available_amount": self.state[0],
            "change_in_holdings": self.HOLDINGS[-2] - self.HOLDINGS[-1],
            "shares_holding": self.state[-1],
            "reward": self.reward,
            "action": actions,
            # "buy_sell_event": self.tracking_buy_sell,
        }
        truncated = False
        return (self.state, self.reward, done, truncated, self.info)

    def buy(self):
        available_amount = self.state[0]
        close_price = self.state[1]

        shares = min(available_amount // close_price, self.HMAX)
        if shares > 0:
            buy_prices_with_commission = (close_price * (1 + self.BUY_COST)) * shares
            self.state[0] -= buy_prices_with_commission
            self.state[-1] += shares
            # self.tracking_buy_sell.append({"buy": buy_prices_with_commission})
            # print(f"Bought {shares} at {buy_prices_with_commission:.2f}")

            current_portfolio_value = self.get_holdings()
            portfolio_change = current_portfolio_value - self.AMOUNT
            stock_profit = current_portfolio_value - buy_prices_with_commission
            self.reward = portfolio_change + stock_profit

    def sell(self):
        close_price = self.state[1]
        shares = self.state[-1]

        if shares > 0:
            shares = min(shares, self.HMAX)
            sell_prices_with_commission = (close_price * (1 + self.BUY_COST)) * shares
            self.state[0] += sell_prices_with_commission
            self.state[-1] -= shares
            # self.tracking_buy_sell.append({"sell": sell_prices_with_commission})
            # print(f"Sold {shares} at {sell_prices_with_commission:.2f}")

            current_portfolio_value = self.get_holdings()
            portfolio_change = current_portfolio_value - self.AMOUNT
            stock_profit = sell_prices_with_commission - current_portfolio_value
            self.reward = portfolio_change + stock_profit

    def hold(self):
        current_portfolio_value = self.get_holdings()
        self.reward = current_portfolio_value - self.AMOUNT

    def get_holdings(self):
        available_amount = self.state[0]
        close_price = self.state[1]
        shares = self.state[-1]

        holdings = close_price * shares + available_amount
        return holdings

    def generate_state(self, reset=False):
        state = self.arrays[self.index]
        state = torch.concatenate((self.AMOUNT, state))
        if not reset:
            state[-1] = self.state[-1]
            state[0] = self.state[0]
            return state
        return state

    def calculate_reward(self):
        diff = self.AMOUNT - self.HOLDINGS[-1]
        if diff > 0:
            return -diff
        return diff

# load the environment
env = gym.make('Pendulum-v1')

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="gym")'