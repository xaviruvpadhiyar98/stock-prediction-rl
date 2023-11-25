from gymnasium import Env, spaces
import numpy as np
import torch


class StockTradingEnv(Env):
    """
    Observations =
    [
        'Close', 'High', 'Low',

        'Past1Hour', 'Past2Hour', 'Past3Hour', 'Past4Hour', 'Past5Hour', 'Past6Hour', 'Past7Hour', 'Past8Hour', 'Past9Hour', 'Past10Hour', 'Past11Hour', 'Past12Hour', 'Past13Hour', 'Past14Hour', 'Past15Hour', 'Past16Hour', 'Past17Hour', 'Past18Hour', 'Past19Hour', 'Past20Hour', 'Past21Hour', 'Past22Hour', 'Past23Hour', 'Past24Hour', 'Past25Hour', 'Past26Hour', 'Past27Hour', 'Past28Hour', 'Past29Hour',

        'RSI', 'EMA9', 'EMA21', 'MACD', 'MACD_SIGNAL', 'BBANDS_UPPER', 'BBANDS_MIDDLE', 'BBANDS_LOWER', 'ADX', 'STOCH_K', 'STOCH_D', 'ATR', 'CCI', 'MOM', 'ROC', 'WILLR', 'PPO',

        'Previous1Action', 'Previous2Action', 'Previous3Action', 'Previous4Action', 'Previous5Action', 'Previous6Action', 'Previous7Action', 'Previous8Action', 'Previous9Action', 'Previous10Action', 'Previous11Action', 'Previous12Action', 'Previous13Action', 'Previous14Action', 'Previous15Action', 'Previous16Action', 'Previous17Action', 'Previous18Action', 'Previous19Action', 'Previous20Action', 'Previous21Action', 'Previous22Action', 'Previous23Action', 'Previous24Action', 'Previous25Action', 'Previous26Action', 'Previous27Action', 'Previous28Action', 'Previous29Action',

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

    HMAX = torch.tensor(5, device="cuda:0")
    BUY_COST = 20
    SELL_COST = 20
    SEED = 1337

    def __init__(self, stock_data, mode="train", seed=SEED):
        super(StockTradingEnv, self).__init__()
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
        self.unsuccessful_buys = 0
        self.unsuccessful_holds = 0

        self.successful_sells = 0
        self.successful_buys = 0
        self.successful_holds = 0
        self.neutral_holds = 0

        self.bad_sells = 0
        self.bad_buys = 0
        self.bad_holds = 0

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
        action = action.item()

        descriptive_action, action_function = self.action_mapping[action]
        # self.previous_actions.append(action)
        # self.info = {"previous_actions": self.previous_actions}
        self.info = {}

        action_function()

        info = self.generate_info()
        self.info.update(info)

        done = self.index == (len(self.stock_data) - 1)

        if done or self.truncated:
            return (self.state, self.reward, done, self.truncated, self.info)

        self.index += 1
        self.state = self.generate_next_state(action)
        return (self.state, self.reward, done, self.truncated, self.info)

    def buy(self):
        close_price = self.state[self.close_price_index]
        if close_price <= self.available_amount:  # Ensure we have enough funds to buy
            # Calculate potential profit: considering simple future prediction using EMA difference
            potential_profit = (
                self.state[self.technical_indicator_index_range[1]]
                - self.state[self.technical_indicator_index_range[2]]
            )

            if (
                potential_profit > 0
            ):  # Encouraging buy when the potential profit is positive
                self.reward = potential_profit
                self.successful_buys += 1
                self.info["action"] = "[SUCCESSFUL BUY]"
            else:
                self.reward = (
                    potential_profit / 2
                )  # Half-penalize when the EMA predicts a loss
                self.unsuccessful_buys += 1
                self.info["action"] = "[UNSUCCESSFUL BUY]"

            self.available_amount -= close_price + self.BUY_COST
            self.available_shares += 1
            self.transactions.append(("BUY", close_price))

        else:  # Penalize if not enough funds to buy
            self.reward = -self.HMAX
            self.bad_buys += 1
            self.truncated = True
            self.info["action"] = "[BAD BUY] NO_MONEY_TO_BUY"

    def sell(self):
        close_price = self.state[self.close_price_index]
        if self.available_shares > 0:  # Ensure we have shares to sell
            profit = (
                close_price - self.transactions[-1][1]
            ) * self.available_shares - self.SELL_COST

            if profit > 0:  # Reward based on profit, dynamically scaled
                self.reward = profit * (
                    len(self.transactions) / (self.successful_sells + 1)
                )
                self.successful_sells += 1
                self.info["action"] = "[SUCCESSFUL SELL]"
            else:  # Increased penalty for selling at a loss after buying
                self.reward = 2 * profit
                self.unsuccessful_sells += 1
                self.info["action"] = "[UNSUCCESSFUL SELL]"

            self.cummulative_profit_loss += profit
            self.available_amount += close_price * self.available_shares
            self.available_shares = 0
            self.transactions.append(("SELL", close_price))
        else:  # Penalize if no shares to sell
            self.reward = -self.HMAX
            self.bad_sells += 1
            self.truncated = True
            self.info["action"] = "[BAD SELL] NO_SHARES_TO_SELL"

    def hold(self):
        # Simple hold reward strategy: Neutral reward for holding unless the stock drops significantly
        close_price = self.state[self.close_price_index]
        past_close = self.state[self.close_price_index + 1]

        if close_price < (0.95 * past_close):  # If stock price dropped by more than 5%
            self.reward = close_price - past_close
            self.bad_holds += 1
            self.info["action"] = "[BAD HOLD] STOCK_DROPPED"
        elif close_price > (
            1.05 * past_close
        ):  # If stock price increased by more than 5%
            self.reward = (
                close_price - past_close
            ) / 2  # Encourage but not as much as selling
            self.successful_holds += 1
            self.info["action"] = "[SUCCESSFUL HOLD] STOCK_INCREASED"
        else:  # Neutral hold
            self.reward = 0
            self.neutral_holds += 1
            self.info["action"] = "[NEUTRAL HOLD]"

    def generate_first_state(self):
        state = self.stock_data[self.index]
        return state

    def generate_next_state(self, current_action):
        state = self.stock_data[self.index]
        state[self.available_amount_index] = self.available_amount
        state[self.available_shares_index] = self.available_shares
        state[self.cummulative_profit_loss_index] = self.cummulative_profit_loss
        state[self.portfolio_value_index] = self.portfolio_value

        state[self.previous_action_index_range] = torch.roll(
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
            "unsuccessful_buys": self.unsuccessful_buys,
            "unsuccessful_holds": self.unsuccessful_holds,
            "neutral_holds": self.neutral_holds,
            "bad_sells": self.bad_sells,
            "bad_buys": self.bad_buys,
            "transactions": self.transactions,
            "seed": self.seed,
        }
