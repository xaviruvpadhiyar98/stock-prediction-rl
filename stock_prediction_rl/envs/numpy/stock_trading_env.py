from gymnasium import Env, spaces
import numpy as np
import polars as pl
from pathlib import Path


correct_actions = (
    pl.read_excel(Path.home() / "Documents/LabelTradeSBI.NS.xlsx")
    .select("Actions")
    .to_series()
    .to_list()
)


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

    HMAX = 5
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

        # If close price is greater than available_amount_plus_commission, early_stop env
        if close_price > (self.available_amount - self.BUY_COST):
            self.info["action"] = "[BAD BUY] NO_MONEY_TO_BUY"
            self.reward = -100_000
            self.truncated = True
            self.bad_buys += 1
            return

        shares_to_buy = min((self.available_amount // close_price), self.HMAX)
        buy_prices_with_commission = (close_price * shares_to_buy) + self.BUY_COST
        self.available_amount -= buy_prices_with_commission
        self.available_shares += shares_to_buy

        avg_buy_price = buy_prices_with_commission / shares_to_buy
        self.transactions.append(avg_buy_price)
        self.portfolio_value = (
            self.available_shares * close_price + self.available_amount
        )

        self.info["shares_bought"] = shares_to_buy
        self.info["buy_prices_with_commission"] = buy_prices_with_commission
        self.info["avg_buy_price"] = avg_buy_price

        past_n_maximum_price = max(self.state[self.past_n_hour_index_range])
        diff = close_price - past_n_maximum_price

        # if close price is greater than past n prices, give good reward
        if diff > 0:
            self.info[
                "action"
            ] = f"[GOOD BUY] CLOSE_PRICE > PAST_N_PRICE == DIFF={diff}"
            self.reward = 1
            self.successful_buys += 1
            return

        # if close price is less than past n prices, give average reward
        self.info[
            "action"
        ] = f"[NOT_A_GOOD BUY] CLOSE_PRICE > PAST_N_PRICE == DIFF={diff}"
        self.reward = -1
        self.unsuccessful_buys += 1
        return

    def sell(self):
        close_price = self.state[self.close_price_index]

        # if we dont have any shares, stop environment early
        if self.available_shares < 1:
            self.info["action"] = "[BAD SELL] NO_SHARES_TO_SELL"
            self.reward = -100_000
            self.truncated = True
            self.bad_sells += 1
            return

        shares_to_sell = min(self.available_shares, self.HMAX)
        sell_prices_with_commission = (close_price * shares_to_sell) - self.SELL_COST

        self.available_amount += sell_prices_with_commission
        self.available_shares -= shares_to_sell

        avg_sell_price = sell_prices_with_commission / shares_to_sell
        avg_buy_price = self.transactions.pop(0)
        net_difference = avg_sell_price - avg_buy_price
        self.cummulative_profit_loss += net_difference

        self.portfolio_value = (
            self.available_shares * close_price + self.available_amount
        )

        self.info["avg_buy_price"] = avg_buy_price
        self.info["avg_sell_price"] = avg_sell_price
        self.info["shares_sold"] = shares_to_sell
        self.info["profit_or_loss"] = net_difference
        self.info["sell_prices_with_commission"] = sell_prices_with_commission

        # if loss in profit/loss goes beyond 200, stop environment early
        if self.cummulative_profit_loss < -200:
            self.reward = -100_000
            self.truncated = True
            self.bad_sells += 1
            self.info[
                "action"
            ] = f"[BAD SELL] LOSS_IS_TOO_MUCH == DIFF={self.cummulative_profit_loss}"
            return

        # if profit is greater than 20 then provide a good reward
        if net_difference > 1:
            self.reward = 1
            self.successful_sells += 1
            self.info["action"] = f"[GOOD SELL] HAD_PROFIT_OF = {net_difference}"
            return

        # if it's a loss, just subtract reward
        self.reward = -1
        self.unsuccessful_sells += 1
        self.info["action"] = f"[NOT_A_GOOD SELL] LOSS OF = {net_difference}"
        return

    def hold(self):
        close_price = self.state[self.close_price_index]
        past_n_minimum_price = min(self.state[self.past_n_hour_index_range])
        past_n_maximum_price = max(self.state[self.past_n_hour_index_range])

        # Agent should have bought but it didnt, penalize
        if (self.available_shares == 0) and (close_price > past_n_minimum_price):
            self.reward = -100_000
            self.bad_holds += 1
            self.info["action"] = (
                f"[NOT_A_GOOD HOLD] CHANCE OF BUYING AT {past_n_minimum_price}"
                f" CURRENT_PRICE {close_price}"
                f" CHANCE OF PROFIT = {close_price - past_n_minimum_price}"
            )
            self.truncated = True
            return

        # If the agent holds while the stock price is increasing, give a small reward.
        if (self.available_shares > 0) and close_price > past_n_maximum_price:
            self.info["action"] = (
                f"[GOOD HOLD] RISING_PRICE"
                f" ESTIMATED PROFIT = {close_price - past_n_maximum_price}"
            )
            self.reward = 1
            self.successful_holds += 1
            return

        # If the price is decreasing and the agent holds without selling, penalize slightly.
        if (self.available_shares > 0) and close_price < past_n_minimum_price:
            self.info["action"] = "[NOT_A_GOOD HOLD] PRICE_FALLING"
            self.reward = -1
            self.unsuccessful_holds += 1
            return

        # tiny reward when neither rise nor fall
        self.reward = 0
        self.info["action"] = "[NEUTRAL_HOLD]"
        self.neutral_holds += 1
        return

        # # Agent should have sold but it didnt, penalize
        # if (self.available_shares > 0) and (close_price > past_n_minimum_price):
        #     self.reward = -1
        #     self.unsuccessful_holds += 1
        #     self.action = (
        #         f"[NOT_A_GOOD HOLD] CHANCE OF BUYING AT {past_n_minimum_price}"
        #         f" CURRENT_PRICE {close_price}"
        #         f" CHANCE OF PROFIT = {close_price - past_n_minimum_price}"
        #     )
        #     return

        # # Scenario 2: Agent holds while price is continuously dropping, penalize
        # if close_price < past_n_minimum_price:
        #     self.reward = -1
        #     self.unsuccessful_holds += 1
        #     self.action = f"PRICE_DROPPING_AGENT_SHOULD_HAVE_BOUGHT"
        #     return

        # # Scenario 3: Agent holds after buying and price is dropping rapidly, penalize
        # if self.available_shares > 0 and (close_price - past_n_maximum_price) < -200:  # Assuming a drop of $10 is significant
        #     self.reward = -1
        #     self.unsuccessful_holds += 1
        #     self.action = f"RECENT_BUY_BUT_PRICE_DROPPING_RAPIDLY"
        #     return

        # # Scenario 4: Agent holds when it should have sold for profit, penalize
        # if self.available_shares > 10 and close_price == past_n_maximum_price:  # Assuming holding more than 10 shares is significant
        #     self.reward = -1
        #     self.unsuccessful_holds += 1
        #     self.action = f"HOLDING_MANY_SHARES_AT_PEAK_PRICE"
        #     return

        # # If the agent holds while the stock price is increasing, give a small reward.
        # if close_price > past_n_maximum_price and self.available_shares > 0:
        #     self.reward = 2
        #     self.info["action"] = "HOLD_ON_RISING_PRICE"
        #     self.successful_holds += 1
        #     return

        # # If the price is decreasing and the agent holds without selling, penalize slightly.
        # elif close_price < past_n_minimum_price and self.available_shares > 0:
        #     self.reward = -1
        #     self.info["action"] = "HOLD_ON_FALLING_PRICE"
        #     self.unsuccessful_holds += 1
        #     return

        # # If the agent holds when it potentially could have bought at a low price.
        # elif self.available_shares == 0 and close_price <= past_n_minimum_price:
        #     self.reward = -1
        #     self.info["action"] = "MISSED_BUY_OPPORTUNITY"
        #     self.unsuccessful_holds += 1
        #     return

        # # Neutral hold: neither a clear rise nor a clear drop. Give a tiny reward to incentivize holding when unsure.
        # else:
        #     self.reward = 0.5
        #     self.info["action"] = "NEUTRAL_HOLD"
        #     self.successful_holds += 1
        #     return

        # if close_price < self.transactions[0]:
        #     self.reward = 1
        #     self.successful_holds += 1
        #     self.info["action"] = "CLOSE_PRICE_>_BUY_PRICE_SUCESSFUL_HOLD"
        #     return

        # if close_price > self.transactions[0] + 50:
        #     self.reward = -1
        #     self.unsuccessful_holds += 1
        #     self.info["action"] = "CLOSE_PRICE_<_BUY_PRICE_UNSUCCESSFUL_HOLD"
        #     return

        # if self.successful_holds > 30:
        #     self.reward -= 100_000
        #     self.truncated = True
        #     self.bad_holds += 1
        #     self.info["action"] = f"HOLDING_FOR_MORE_THAN_{self.successful_holds}_BAD_HOLD"
        #     return

        # self.reward = 1
        # self.successful_holds += 1
        # self.info["action"] = "HOLD"

    def generate_first_state(self):
        state = self.stock_data[self.index]
        return state

    def generate_next_state(self, current_action):
        state = self.stock_data[self.index]
        state[self.available_amount_index] = self.available_amount
        state[self.available_shares_index] = self.available_shares
        state[self.cummulative_profit_loss_index] = self.cummulative_profit_loss
        state[self.portfolio_value_index] = self.portfolio_value

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
            "unsuccessful_buys": self.unsuccessful_buys,
            "unsuccessful_holds": self.unsuccessful_holds,
            "neutral_holds": self.neutral_holds,
            "bad_sells": self.bad_sells,
            "bad_buys": self.bad_buys,
            "transactions": self.transactions,
            "seed": self.seed,
        }
