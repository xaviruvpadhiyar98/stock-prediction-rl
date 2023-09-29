from gymnasium import Env, spaces
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StockTradingEnv(Env):
    """
    Observations =
    available amount + (close price + past_14_hours + buy/sell/hold/shares)
    Actions = Discrete(3)  # buy-2,hold-1,sell-0
    ----
    GOOD_BUY =
        SUCCESSFUL_BUY = IF PAST MEAN PRICES > CLOSE PRICE
        UNSUCCESSFUL_BUY = IF PAST MEAN PRICES < CLOSE PRICE
    BAD_BUY = DONT HAVE MONEY TO BUY ANY 1 SHARES
    ----
    GOOD_SELL =
        SUCCESSFUL_SELL = IF SELL PRICE > AVERAGE BUY PRICE
        UNSUCCESSFUL_SELL = IF SELL PRICE < AVERAGE BUY PRICE
    BAD_SELL = IF DONT HAVE ANY SHARES TO SELL
    ----
    GOOD_HOLD = IF SHARES > 0, and CLOSE PRICE > AVERAGE BUY PRICE
    BAD_HOLD = IF SHARES > 0, and CLOSE PRICE < AVERAGE BUY PRICE
    SUCCESSFUL_HOLD = IF (SHARES = 0), and CLOSE PRICE > PAST MEAN PRICES
    UNSUCCESSFUL_HOLD = IF (SHARES = 0), and CLOSE PRICE < PAST MEAN PRICES
    ----
    """

    HMAX = 5
    AMOUNT = 10_000
    REWARD_SCALING = 2**-11
    BUY_COST = 20
    SELL_COST = 20
    BUY = 2
    HOLD = 1
    SELL = 0

    def __init__(self, stock_data, tickers, use_tensor=False):
        self.stock_data = stock_data
        self.tickers = tickers
        self.action_space = spaces.Discrete(3)  # buy,hold,sell
        self.use_tensor = use_tensor
        if use_tensor:
            self.AMOUNT = torch.Tensor([10_000]).to(DEVICE)
        # shape = (
        # available amount + close price
        # + Stock Indicators + past_hours + buy/sell/hold/shares
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

        self.good_buys = 0
        self.good_sells = 0
        self.good_holds = 0

        self.bad_buys = 0
        self.bad_sells = 0
        self.bad_holds = 0

        self.unsuccessful_buys = 0
        self.unsuccessful_sells = 0
        self.unsuccessful_holds = 0

        self.successful_buys = 0
        self.successful_sells = 0
        self.successful_holds = 0

        self.buy_transactions = []
        self.sell_transactions = []

        self.cummulative_profit_loss = 0
        self.truncated = False

        self.state = self.generate_state(reset=True)
        self.info = self.generate_info()
        self.previous_portfolio_value = self.info["portfolio_value"]
        return self.state, self.info

    def step(self, action):
        self.index += 1
        self.truncated = False

        done = bool(self.index == len(self.stock_data))
        if done:
            return (self.state, self.reward, done, self.truncated, self.info)
        self.state = self.generate_state()

        self.info = {}
        if action == self.BUY:
            self.buy()
        elif action == self.SELL:
            self.sell()
        else:
            self.hold()

        self.info.update(self.generate_info())
        self.truncated = False
        return (self.state, self.reward, done, self.truncated, self.info)

    def buy(self):
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

            # Strategy 1 - Past Hour Buy Indicator/Reward Value
            # past_hour_mean = (self.state[2:-1]).mean()
            # self.info["past_hour_mean"] = past_hour_mean
            # net_difference = past_hour_mean - close_price

            # Strategy 2 - Give reward based on only portfolio value
            available_amount = self.state[0]
            portfolio_value = buy_prices_with_commission + available_amount
            net_difference = portfolio_value - self.previous_portfolio_value
            self.previous_portfolio_value = portfolio_value

            if net_difference > 0:
                self.reward = net_difference
                self.successful_buys += 1
                self.info["action"] = "SUCCESSFUL_BUY"
            else:
                self.reward = net_difference
                self.unsuccessful_buys += 1
                self.info["action"] = "UNSUCCESSFUL_BUY"

        else:
            self.reward += -100_000
            self.truncated = True
            self.bad_buys += 1
            self.info["action"] = "BAD_BUY"

    def sell(self):
        close_price = self.state[1]
        shares = int(self.state[-1].item())

        if shares > 0:
            shares = min(shares, self.HMAX)
            sell_prices_with_commission = (close_price * shares) + self.SELL_COST
            self.state[0] += sell_prices_with_commission
            self.state[-1] -= shares

            available_amount = self.state[0]
            available_shares = self.state[-1]
            portfolio_value = available_shares * close_price + available_amount
            net_difference = portfolio_value - self.previous_portfolio_value
            self.previous_portfolio_value = portfolio_value

            if net_difference > 0:
                self.reward = net_difference
                self.successful_buys += 1
                self.info["action"] = "SUCCESSFUL_SELL"
            else:
                self.reward = net_difference
                self.unsuccessful_buys += 1
                self.info["action"] = "UNSUCCESSFUL_SELL"

            # starting_n_avg_buy_price = self.buy_transactions[:shares]
            # self.buy_transactions = self.buy_transactions[shares:]

            # starting_n_avg_buy_price = sum(starting_n_avg_buy_price)

            # profit_or_loss = sell_prices_with_commission - starting_n_avg_buy_price

            # self.cummulative_profit_loss += profit_or_loss

            # if profit_or_loss > 0:
            #     self.successful_sells += 1
            #     self.reward = 2 * profit_or_loss * self.REWARD_SCALING
            #     self.info["action"] = "SUCCESSFUL_SELL"
            # else:
            #     self.unsuccessful_sells += 1
            #     self.reward = 2 * profit_or_loss * self.REWARD_SCALING
            #     self.info["action"] = "UNSUCCESSFUL_SELL"

            self.info["shares_sold"] = shares
            self.info["profit_or_loss"] = net_difference
            self.info["sell_prices_with_commission"] = sell_prices_with_commission
            self.info["avg_sell_price"] = sell_prices_with_commission / shares

        else:
            self.reward = -100_000
            self.bad_sells += 1
            self.info["action"] = "BAD_SELL"
            self.truncated = True

    def hold(self):
        available_price = self.state[0]
        close_price = self.state[1]
        shares = self.state[-1]

        # Strategy 2: Portfolio Value/Reward
        portfolio_value = shares * close_price + available_price - 20
        net_difference = portfolio_value - self.previous_portfolio_value
        self.previous_portfolio_value = portfolio_value

        if net_difference > 0:
            self.reward = net_difference
            self.successful_holds += 1
            self.info["action"] = "SUCCESSFUL_HOLD"
        else:
            self.reward = net_difference
            self.unsuccessful_holds += 1
            self.info["action"] = "UNSUCCESSFUL_HOLD"

        # Strategy 1: Indepth rewards
        # if shares == 0:
        #     past_hour_mean = (self.state[2:-1]).mean()
        #     self.info["past_hour_mean"] = past_hour_mean
        #     net_difference = close_price - past_hour_mean

        #     if net_difference > 0:
        #         self.unsuccessful_holds += 1
        #         self.reward = 10 * net_difference
        #         self.info["action"] = "SUCCESSFUL_HOLD"
        #     else:
        #         self.successful_holds += 1
        #         self.reward = 10 * net_difference

        #         self.info["action"] = "UNSUCCESSFUL_HOLD"

        # else:
        #     starting_n_avg_buy_price = sum(self.buy_transactions) / len(
        #         self.buy_transactions
        #     )
        #     self.info["past_hour_mean"] = starting_n_avg_buy_price

        #     net_difference = close_price - starting_n_avg_buy_price
        #     if net_difference > 0:
        #         self.good_holds += 1
        #         self.reward = 10 * net_difference
        #         self.info["action"] = "GOOD_HOLD"
        #     else:
        #         self.bad_holds += 1
        #         self.reward = 10 * net_difference
        #         self.info["action"] = "BAD_HOLD"

    def generate_state(self, reset=False):
        state = self.stock_data[self.index]
        if self.use_tensor:
            state = torch.concatenate((self.AMOUNT, state)).to(dtype=torch.float32)
        else:
            state = np.append(np.array([self.AMOUNT]), state).astype('float32')
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
            "portfolio_value": portfolio_value,
            "index": self.index,
            "close_price": close_price,
            "available_amount": available_amount,
            "shares_holdings": shares,
            "cummulative_profit_loss": self.cummulative_profit_loss,
            # "good_buys": self.good_buys,
            # "good_sells": self.good_sells,
            # "good_holds": self.good_holds,
            "bad_buys": self.bad_buys,
            "bad_sells": self.bad_sells,
            "bad_holds": self.bad_holds,
            "reward": self.reward,
            "successful_buys": self.successful_buys,
            "unsuccessful_buys": self.unsuccessful_buys,
            "successful_sells": self.successful_sells,
            "unsuccessful_sells": self.unsuccessful_sells,
            "successful_holds": self.successful_holds,
            "unsuccessful_holds": self.unsuccessful_holds,
        }
