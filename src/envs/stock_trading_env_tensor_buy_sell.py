from gymnasium import Env, spaces
import numpy as np
import torch
from collections import Counter


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
        self.previous_actions.append(action)
        self.info = {"previous_actions": self.previous_actions}
        self.info = {}

        action_function()

        info = self.generate_info()
        self.info.update(info)



        done = (self.index == (len(self.stock_data) - 1))

        if done or self.truncated:
            return (self.state, self.reward, done, self.truncated, self.info)

        self.index += 1        
        self.state = self.generate_next_state(action)
        return (self.state, self.reward, done, self.truncated, self.info)

    def buy(self):
        close_price = self.state[self.close_price_index]

        available_amount_with_commission = self.available_amount - self.BUY_COST
        shares_to_buy = min(
            int((available_amount_with_commission // close_price)), self.HMAX
        )

        if shares_to_buy < 1:
            self.truncated = True
            self.reward = -100_000
            self.bad_buys += 1
            self.info["action"] = "NO_SHARES_TO_SELL_BAD_BUY"
            return

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
        self.info["action"] = "BUY"
        self.reward = 1
        self.successful_buys += 1


    def sell(self):
        close_price = self.state[self.close_price_index]

        if self.available_shares < 1:
            self.reward -= 100_000
            self.truncated = True
            self.bad_sells += 1
            self.info["action"] = "NO_SHARES_TO_SELL_BAD_SELL"
            return
        
        if self.cummulative_profit_loss < -200:
            self.reward -= 100_000
            self.truncated = True
            self.info["action"] = "LOSS_MORE_THAN_200"
            return


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

        self.portfolio_value = (
            self.available_shares * close_price + self.available_amount
        )

        self.info["avg_buy_price"] = avg_buy_price
        self.info["avg_sell_price"] = avg_sell_price
        self.info["shares_sold"] = shares_to_sell
        self.info["profit_or_loss"] = net_difference
        self.info["sell_prices_with_commission"] = sell_prices_with_commission


        if net_difference > 20:
            self.reward = 100
            # self.reward += net_difference * 2
            self.successful_sells += 1
            self.info["action"] = f"SUCCESSFUL_SELL_{net_difference}"
            return
        
        # self.reward += net_difference * 2
        self.reward = 1
        self.unsuccessful_sells += 1
        self.info["action"] = f"UNSUCCESSFUL_SELL_{net_difference}"
        return


    def hold(self):

        if self.successful_holds > 30:
            self.reward -= 100_000
            self.truncated = True
            self.bad_holds += 1
            self.info["action"] = f"HOLDING_FOR_MORE_THAN_{self.successful_holds}_BAD_HOLD"
            return

        self.reward = 1
        self.successful_holds += 1
        self.info["action"] = "HOLD"


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
            "bad_sells": self.bad_sells,
            "bad_buys": self.bad_buys,
            "transactions": self.transactions,
            "seed": self.seed,
        }
