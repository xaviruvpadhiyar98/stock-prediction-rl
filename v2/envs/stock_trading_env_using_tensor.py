from gymnasium import Env, spaces
import numpy as np
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StockTradingEnv(Env):
    HMAX = 2
    AMOUNT = torch.Tensor([10_000]).to(DEVICE)
    BUY_COST = SELL_COST = 0.01
    SEED = 1337
    BUY = 2
    HOLD = 1
    SELL = 0

    def __init__(self, stock_data, tickers, seed=SEED):
        self.stock_data = stock_data
        self.seed = seed
        self.tickers = tickers
        self.action_space = spaces.Discrete(3)  # buy,hold,sell
        # shape = (
        # available amount + (
        #   close price + past_hours + buy/sell/hold/-shares
        # ) + total_trades + successful_trades
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1 + len(stock_data[0]),),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.index = 0
        self.reward = 0.0
        self.info = {}
        self.HOLDINGS = [self.AMOUNT]
        self.tracking_buy_sell = []
        self.total_trades = 0
        self.successful_trades = 0
        self.state = self.generate_state(reset=True)
        return self.state, self.info

    def step(self, action):
        done = bool(self.index == len(self.stock_data) - 1)
        if done:
            truncated = False
            return (self.state, self.reward, done, truncated, self.info)
        self.index += 1

        if action == self.BUY:
            self.buy()
        elif action == self.SELL:
            self.sell()
        else:
            self.hold()

        holdings = self.get_holdings()
        self.HOLDINGS.append(holdings)
        self.reward += self.calculate_reward(holdings)
        self.state = self.generate_state()
        self.info = {
            "holdings": holdings,
            "available_amount": self.state[0],
            "change_in_holdings": self.HOLDINGS[-2] - self.HOLDINGS[-1],
            "shares_holding": self.state[-1],
            "reward": self.reward,
            "action": action,
            "close_price": self.state[1],
            "total_trades": self.total_trades,
            "successful_trades": self.successful_trades
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

            self.last_buy_price = buy_prices_with_commission
            self.total_trades += 1
        else:
            self.reward += -1

    def sell(self):
        close_price = self.state[1]
        shares = self.state[-1]

        if shares > 0 and hasattr(self, "last_buy_price"):
            shares = min(shares, self.HMAX)
            sell_prices_with_commission = (close_price * (1 + self.SELL_COST)) * shares
            self.state[0] += sell_prices_with_commission
            self.state[-1] -= shares

            self.reward += sell_prices_with_commission - self.last_buy_price
            self.total_trades += 1
            if self.reward > 0:
                self.reward += 1
                self.successful_trades += 1
            else:
                self.reward -= 1
        else:
            self.reward += -1

    def hold(self):
        self.reward = 0.01

    def get_holdings(self):
        available_amount = self.state[0]
        close_price = self.state[1]
        shares = self.state[-1]

        holdings = close_price * shares + available_amount
        return holdings

    def generate_state(self, reset=False):
        state = self.stock_data[self.index]
        state = torch.concatenate((self.AMOUNT, state))
        if not reset:
            state[-1] = self.state[-1]
            state[0] = self.state[0]
            return state
        return state

    def calculate_reward(self, holdings):
        net_difference = holdings - self.AMOUNT

        if net_difference == 0:
            return -1

        if net_difference > 0:
            return +1

        if net_difference < 0:
            return -1
