from gymnasium import Env, spaces
import numpy as np


class StockTradingEnv(Env):
    """
    Observations =
    [
        'Close', 'High', 'Low',

        'Past1Hour', 'Past2Hour', 'Past3Hour', 'Past4Hour', 'Past5Hour',
        'Past6Hour', 'Past7Hour', 'Past8Hour', 'Past9Hour', 'Past10Hour',
        'Past11Hour', 'Past12Hour', 'Past13Hour', 'Past14Hour',

        'RSI', 'EMA9', 'EMA21', 'MACD', 'MACD_SIGNAL',
        'BBANDS_UPPER', 'BBANDS_MIDDLE', 'BBANDS_LOWER',
        'ADX', 'STOCH_K', 'STOCH_D', 'ATR', 'CCI', 'MOM',
        'ROC', 'WILLR', 'PPO',

        'Previous1Action', 'Previous2Action', 'Previous3Action', 'Previous4Action', 'Previous5Action', 'Previous6Action', 'Previous7Action', 'Previous8Action', 'Previous9Action', 'Previous10Action', 'Previous11Action', 'Previous12Action', 'Previous13Action', 'Previous14Action', 'Previous15Action', 'Previous16Action', 'Previous17Action', 'Previous18Action', 'Previous19Action', 'Previous20Action',

        'PortfolioValue', 'AvailableAmount', 'SharesHolding', 'CummulativeProfitLoss'
    ]
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
    BUY_COST = 20
    SELL_COST = 20
    SEED = 1337

    def __init__(self, stock_data, mode="train", seed=SEED):
        self.stock_data = stock_data
        self.mode = mode
        self.seed = seed
        self.action_space = spaces.Discrete(3)  # buy,hold,sell

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(stock_data[0]),),
            dtype=np.float32,
        )
        self.close_price_index = 0
        self.portfolio_value_index = -4
        self.available_amount_index = -3
        self.available_shares_index = -2
        self.past_hours_range = range(1, 30)
        self.past_action_range = range(1, 30)
        self.n_technical_indicator = 17
        self.cummulative_profit_loss_index = -1
        self.past_n_hour_index_range = range(3, 3 + max(self.past_hours_range))
        self.technical_indicator_index_range = range(
            max(self.past_n_hour_index_range) + 1,
            max(self.past_n_hour_index_range) + 1 + self.n_technical_indicator,
        )
        self.previous_action_index_range = range(
            max(self.technical_indicator_index_range) + 1,
            max(self.technical_indicator_index_range) + 1 + max(self.past_action_range),
        )

        self.action_mapping = {
            0: ["SELL", self.sell],
            1: ["HOLD", self.hold],
            2: ["BUY", self.buy],
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=self.seed)
        self.seed = self.seed
        self.index = 0
        self.reward = 0
        self.previous_actions = []

        self.unsuccessful_sells = 0
        self.successful_sells = 0
        self.successful_buys = 0
        self.successful_holds = 0

        self.transactions = []

        self.truncated = False

        self.state = self.generate_first_state()

        self.available_amount = 10_000
        self.portfolio_value = 10_000
        self.available_shares = 0
        self.cummulative_profit_loss = 0

        info = self.generate_info()
        return self.state, info

    def step(self, action):
        done = self.index == len(self.stock_data) - 1
        if done or self.truncated:
            return (self.state, self.reward, done, self.truncated, self.info)

        descriptive_action, fn = self.action_mapping[action]
        self.previous_actions.append(action)
        self.info = {"previous_actions": self.previous_actions}
        fn()

        close_price = self.state[self.close_price_index]
        self.portfolio_value = (
            self.available_shares * close_price + self.available_amount
        )
        self.state[self.available_amount_index] = self.available_amount
        self.state[self.available_shares_index] = self.available_shares
        self.state[self.cummulative_profit_loss_index] = self.cummulative_profit_loss
        self.state[self.portfolio_value_index] = self.portfolio_value

        self.index += 1
        info = self.generate_info()
        self.info.update(info)
        self.state = self.generate_next_state(action)
        return (self.state, self.reward, done, self.truncated, self.info)

    def buy(self):
        close_price = self.state[self.close_price_index]

        available_amount_with_commission = self.available_amount - self.BUY_COST
        shares_to_buy = min(
            int((available_amount_with_commission // close_price)), self.HMAX
        )

        if shares_to_buy > 0:
            buy_prices_with_commission = (close_price * shares_to_buy) + self.BUY_COST
            self.available_amount -= buy_prices_with_commission
            self.available_shares += shares_to_buy

            avg_buy_price = buy_prices_with_commission / shares_to_buy
            self.transactions.append(avg_buy_price)

            self.info["shares_bought"] = shares_to_buy
            self.info["buy_prices_with_commission"] = buy_prices_with_commission
            self.info["avg_buy_price"] = avg_buy_price
            self.info["action"] = "BUY"
            self.reward = -0.01
            self.successful_buys += 1

        else:
            self.reward += -100_000
            self.truncated = True
            self.info["action"] = "BAD_BUY"

    def sell(self):
        close_price = self.state[self.close_price_index]

        if self.available_shares > 0:
            shares_to_sell = min(self.available_shares, self.HMAX)
            sell_prices_with_commission = (
                close_price * shares_to_sell
            ) - self.SELL_COST

            self.available_amount += sell_prices_with_commission
            self.available_shares -= shares_to_sell

            avg_sell_price = sell_prices_with_commission / shares_to_sell
            avg_buy_price = self.transactions.pop(0)
            net_difference = avg_sell_price - avg_buy_price
            self.cummulative_profit_loss += net_difference

            self.info["avg_buy_price"] = avg_buy_price
            self.info["avg_sell_price"] = avg_sell_price
            self.info["shares_sold"] = shares_to_sell
            self.info["profit_or_loss"] = net_difference
            self.info["sell_prices_with_commission"] = sell_prices_with_commission

            if net_difference > 0:
                self.reward = net_difference
                self.successful_sells += 1
                self.info["action"] = "SUCCESSFUL_SELL"
            else:
                self.reward = net_difference
                self.unsuccessful_sells += 1
                self.info["action"] = "UNSUCCESSFUL_SELL"

        else:
            self.reward = -100_000
            self.truncated = True
            self.info["action"] = "BAD_SELL"

    def hold(self):
        self.reward = -0.01
        self.successful_holds += 1
        self.info["action"] = "HOLD"

    def generate_first_state(self):
        state = self.stock_data[self.index]
        return state

    def generate_next_state(self, current_action):
        state = self.stock_data[self.index]
        state[-4:] = self.state[-4:]

        state[self.previous_action_index_range] = np.roll(
            self.state[self.previous_action_index_range], 1
        )
        state[min(self.previous_action_index_range)] = current_action

        return state

    def generate_info(self):
        close_price = self.state[self.close_price_index]

        return {
            "portfolio_value": self.portfolio_value,
            "index": self.index,
            "close_price": close_price,
            "available_amount": self.available_amount,
            "shares_holdings": self.available_shares,
            "cummulative_profit_loss": self.cummulative_profit_loss,
            "reward": self.reward,
            "successful_buys": self.successful_buys,
            "successful_holds": self.successful_holds,
            "successful_sells": self.successful_sells,
            "unsuccessful_sells": self.unsuccessful_sells,
            "transactions": self.transactions,
            "seed": self.seed,
        }
