from gymnasium import Env, spaces
import numpy as np


class StockTradingEnv(Env):
    HMAX = 2
    AMOUNT = 10_000
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

        self.good_buys = 0
        self.good_sells = 0
        self.good_holds = 0

        self.bad_buys = 0
        self.bad_sells = 0
        self.bad_holds = 0

        self.successful_trades = 0
        self.unsuccessful_trades = 0

        self.buy_transactions = []
        self.sell_transactions = []

        self.cummulative_profit_loss = 0

        self.state = self.generate_state(reset=True)
        self.info = self.generate_info()
        return self.state, self.info

    def step(self, action):
        self.index += 1

        done = bool(self.index == len(self.stock_data))
        if done:
            truncated = False
            return (self.state, self.reward, done, truncated, self.info)
        self.state = self.generate_state()

        self.info = {}
        if action == self.BUY:
            self.buy()
        elif action == self.SELL:
            self.sell()
        else:
            self.hold()

        self.info.update(self.generate_info())
        truncated = False
        return (self.state, self.reward, done, truncated, self.info)

    def buy(self):
        self.info["action"] = "BUY"
        available_amount = self.state[0] - self.BUY_COST
        close_price = self.state[1]

        shares = min(int((available_amount // close_price).item()), self.HMAX)

        if shares > 0:
            buy_prices_with_commission = (close_price * shares) + self.BUY_COST
            self.state[0] -= buy_prices_with_commission
            self.state[-1] += shares
            avg_buy_price = buy_prices_with_commission / shares
            self.buy_transactions.extend([avg_buy_price] * shares)

            self.info["shares_bought"] = shares
            self.info["buy_prices_with_commission"] = buy_prices_with_commission
            self.info["avg_buy_price"] = avg_buy_price

            past_hour_mean = (self.state[2:-1]).mean()
            net_difference = past_hour_mean - close_price

            if net_difference > 0:
                self.good_buys += 1
                self.reward += net_difference
            else:
                self.bad_buys += 1
                self.reward += 2 * net_difference

        else:
            self.bad_buys += 1
            self.reward -= available_amount

    def sell(self):
        self.info["action"] = "SELL"
        close_price = self.state[1]
        shares = int(self.state[-1].item())


        # if shares > 0 and self.last_buy_price is not None:
        if shares > 0:
            shares = min(shares, self.HMAX)
            sell_prices_with_commission = (close_price * shares) + self.SELL_COST
            self.state[0] += sell_prices_with_commission
            self.state[-1] -= shares

            starting_n_avg_buy_price = self.buy_transactions[:shares]
            self.buy_transactions = self.buy_transactions[shares:]

            starting_n_avg_buy_price = sum(starting_n_avg_buy_price)

            profit_or_loss = sell_prices_with_commission - starting_n_avg_buy_price

            self.cummulative_profit_loss += profit_or_loss
            if profit_or_loss > 0:
                self.good_sells += 1
                self.reward += profit_or_loss
            else:
                self.bad_sells += 1
                self.reward += 2 * profit_or_loss

            self.info["shares_sold"] = shares
            self.info["profit_or_loss"] = profit_or_loss
            self.info["sell_prices_with_commission"] = sell_prices_with_commission
            self.info["avg_sell_price"] = sell_prices_with_commission / shares

        else:
            self.bad_sells += 1
            self.reward -= 1000


    def hold(self):
        self.info["action"] = "HOLD"
        close_price = self.state[1]
        shares = self.state[-1]

        if shares == 0:
            past_hour_mean = (self.state[2:-1]).mean()
            net_difference = close_price - past_hour_mean

            if net_difference > 0:
                self.bad_holds += 1
                self.reward += 2 * net_difference
            else:
                self.good_holds += 1
                self.reward += net_difference

        else:
            starting_n_avg_buy_price = sum(self.buy_transactions) / len(
                self.buy_transactions
            )

            net_difference = close_price - starting_n_avg_buy_price
            if net_difference > 0:
                self.good_holds += 1
                self.reward += net_difference
            else:
                self.bad_holds += 1
                self.reward += 2 * net_difference



    def generate_state(self, reset=False):
        state = self.stock_data[self.index]
        state = np.append(np.array([self.AMOUNT]), state)
        if not reset:
            state[-1] = self.state[-1]  # shares
            state[0] = self.state[0]  # available amount
            return state
        return state

    def generate_info(self):
        close_price = self.state[1]
        available_amount = self.state[0]
        shares = self.state[-1]

        portfolio_value = shares * close_price + available_amount

        return {
            "index": self.index,
            "close_price": close_price,
            "available_amount": available_amount,
            "shares_holdings": shares,
            "cummulative_profit_loss": self.cummulative_profit_loss,
            "good_buys": self.good_buys,
            "good_sells": self.good_sells,
            "good_holds": self.good_holds,
            "bad_buys": self.bad_buys,
            "bad_sells": self.bad_sells,
            "bad_holds": self.bad_holds,
            "reward": self.reward,
            "portfolio_value": portfolio_value,
        }
    