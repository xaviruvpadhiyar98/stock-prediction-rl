from gymnasium import Env, spaces
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StockTradingEnv(Env):
    HMAX = 5
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
        #   close price + past_hours + buy/sell/hold/shares
        # )
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
        self.unsuccessful_trades = 0
        self.good_buys = 0
        self.good_sells = 0
        self.good_holds = 0
        self.bad_buys = 0
        self.bad_sells = 0
        self.bad_holds = 0
        self.buy_index = []
        self.sell_index = []
        self.cummulative_profit_loss = 0
        self.last_buy_price_with_commission = None
        self.last_buy_price = None

        self.state = self.generate_state(reset=True)
        self.last_action = None
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

        # holdings = self.get_holdings()
        # self.HOLDINGS.append(holdings)
        # self.reward += self.calculate_reward(holdings)

        self.state = self.generate_state()
        self.last_action = action
        self.info = {
            "cummulative_profit_loss": self.cummulative_profit_loss,
            "shares_holding": self.state[-1],
            "reward": self.reward,
            "good_buys": self.good_buys,
            "good_sells": self.good_sells,
            "good_holds": self.good_holds,
            "bad_buys": self.bad_buys,
            "bad_sells": self.bad_sells,
            "bad_holds": self.bad_holds,
            "successful_trades": self.successful_trades,
            "unsuccessful_trades": self.unsuccessful_trades,
            "buy_index": self.buy_index,
            "sell_index": self.sell_index,
            # "available_amount": self.state[0],
            # "holdings": holdings,
            # "change_in_holdings": self.HOLDINGS[-2] - self.HOLDINGS[-1],
            # "action": action,
            # "close_price": self.state[1],
            # "total_trades": self.total_trades,
        }

        truncated = False
        return (self.state, self.reward, done, truncated, self.info)

    def buy(self):
        available_amount = self.state[0]
        close_price = self.state[1]

        shares = min(available_amount // close_price, self.HMAX)
        if shares > 0:
            buy_prices_with_commission = (close_price * (1 + self.BUY_COST)) * shares
            self.last_buy_price_with_commission = buy_prices_with_commission
            self.last_buy_price = close_price
            self.state[0] -= buy_prices_with_commission
            self.state[-1] += shares

            past_hour_mean = (self.state[2:-1]).mean()
            self.reward += close_price - past_hour_mean

            self.good_buys += 1
            self.buy_index.append({self.index: close_price})
        else:
            self.bad_buys += 1
            self.reward -= 1000

    def sell(self):
        close_price = self.state[1]
        shares = self.state[-1]

        if shares > 0 and self.last_buy_price is not None:
            shares = min(shares, self.HMAX)
            sell_prices_with_commission = (close_price * (1 + self.SELL_COST)) * shares
            profit_or_loss = (
                sell_prices_with_commission - self.last_buy_price_with_commission
            )
            self.last_buy_price = None
            self.state[0] += sell_prices_with_commission
            self.state[-1] -= shares

            self.cummulative_profit_loss += profit_or_loss
            self.reward += profit_or_loss
            self.good_sells += 1
            self.sell_index.append({self.index: close_price})
            if profit_or_loss > 0:
                self.successful_trades += 1
                self.reward += 500
            else:
                self.reward -= 500
                self.unsuccessful_trades += 1
        else:
            self.bad_sells += 1
            self.reward -= 1000

    def hold(self):
        if self.last_buy_price is None:
            self.bad_holds += 1
            self.reward -= 1000
        else:
            self.good_holds += 1
            self.reward += self.last_buy_price - self.state[1]

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
        net_difference = (holdings - self.AMOUNT).item()
        if net_difference == 0:
            return -1

        return net_difference
