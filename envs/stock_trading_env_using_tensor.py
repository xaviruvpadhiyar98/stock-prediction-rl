from collections import deque
from gymnasium import Env, spaces
import torch
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StockTradingEnv(Env):
    HMAX = 1
    AMOUNT = torch.Tensor([10_000]).to(DEVICE)
    BUY_COST = SELL_COST = 0.001
    REWARD_SCALING = 2**-11
    GAMMA = 0.99

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

        holdings = self.get_holdings()
        self.HOLDINGS.append(holdings)
        self.reward = self.calculate_reward(holdings)
        self.state = self.generate_state()
        self.info = {
            "holdings": holdings,
            "available_amount": self.state[0],
            "change_in_holdings": self.HOLDINGS[-1] - self.HOLDINGS[-2],
            "shares_holding": self.state[-1],
            "reward": self.reward,
            "action": actions,
            "close_price": self.state[1],
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

            # current_portfolio_value = self.get_holdings()
            # portfolio_change = current_portfolio_value - self.AMOUNT
            # stock_profit = current_portfolio_value - buy_prices_with_commission
            # self.reward = portfolio_change + stock_profit

    def sell(self):
        close_price = self.state[1]
        shares = self.state[-1]

        if shares > 0:
            shares = min(shares, self.HMAX)
            sell_prices_with_commission = (close_price * (1 + self.BUY_COST)) * shares
            self.state[0] += sell_prices_with_commission
            self.state[-1] -= shares

            # current_portfolio_value = self.get_holdings()
            # portfolio_change = current_portfolio_value - self.AMOUNT
            # stock_profit = sell_prices_with_commission - current_portfolio_value
            # self.reward = portfolio_change + stock_profit

    def hold(self):
        ...
        # current_portfolio_value = self.get_holdings()
        # self.reward = current_portfolio_value - self.AMOUNT

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

    def calculate_reward(self, holdings):
        # ...
        # Reward 8 -

        # Reward 7 - Reward Scaling with GAMMA
        # Result - STILL BAD
        # return holdings * self.REWARD_SCALING

        # Reward 6 - Reward Scaling
        # Result - STILL BAD
        # return holdings * self.REWARD_SCALING

        # Reward 5 - End assets - starting assets
        # Result - BAD never crosses initial amount
        # return holdings - self.AMOUNT

        # Reward 4 - Maximize change in holdings reverse
        # Result - Crosses initial amount couple of time
        # but its barely +500 Rs
        # return self.HOLDINGS[-1] - self.HOLDINGS[-2]

        # Reward 3 - Maximize holdings
        # Result - Crosses initial amount frequently
        # but doesnt goes past 1000+
        if holdings > self.AMOUNT:
            return holdings - self.AMOUNT

        return self.AMOUNT - holdings

        # Reward 2 - Maximize change in holdings
        # Result - Crosses initial amount couple of time
        # but its barely +500 Rs
        # return self.HOLDINGS[-2] - self.HOLDINGS[-1]

        # Reward 1 - maximize amount of shares
        # result - never crosses initial amount
        # return self.state[-1]

        # diff = self.AMOUNT - self.HOLDINGS[-1]
        # if diff > 0:
        #     return -diff
        # return diff

        # profit_loss = 0
        # shares_bought_sold = 0
        # for x in self.tracking_buy_sell:
        #     if "buy" in x:
        #         profit_loss += x["buy"]
        #         shares_bought_sold += 1
        #     if "sell" in x:
        #         profit_loss -= x["sell"]
        #         shares_bought_sold -= 1

        # if profit_loss > 0:
        #     return 10
        # return -10
