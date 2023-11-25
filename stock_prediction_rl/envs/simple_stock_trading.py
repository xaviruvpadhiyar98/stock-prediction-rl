from gymnasium import Env, spaces
import numpy as np


class StockTradingEnv(Env):
    """
    Observations
        [Close, Available_Amount, Shares_Holdings]
    Actions
        [HOLD, BUY, SELL]
        [0, 1, 2]
    """

    HMAX = 5
    BUY_COST = 20
    SELL_COST = 20
    AVAILABLE_AMOUNT = 10000
    ACTION_MAP = {0: "HOLD", 1: "BUY", 2: "SELL"}

    def __init__(self, close_prices):
        super().__init__()

        self.close_prices = close_prices
        low = np.min(close_prices)
        high = np.max(close_prices)
        self.length = len(self.close_prices)

        self.observation_space = spaces.Box(
            low=low,
            high=high,
            shape=(3,),
            dtype=np.float32,
        )
        # self.observation_space = spaces.Discrete(start=low, n=low+high)
        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.counter = 0
        self.good_trade = 0
        self.bad_trade = 0
        self.shares_holding = 0
        self.state = self.close_prices[self.counter]
        return self.state, {}

    def step(self, action):
        reward = 0
        truncated = False

        predicted_action = self.ACTION_MAP[action]
        if predicted_action == "BUY":
            self.buy()
        elif predicted_action == "HOLD":
            self.hold()
        elif predicted_action == "SELL":
            self.sell()
        else:
            raise ValueError("Wrong Option")

        done = self.counter == (self.length - 1)
        if done or truncated:
            return self.state, float(reward), done, truncated, info

        self.counter += 1
        self.state = self.close_prices[self.counter]
        return self.state, float(reward), done, truncated, info

        # action_function()

        # info = self.generate_info()
        # self.info.update(info)

        # done = (self.index == (len(self.stock_data) - 1))

        # if done or self.truncated:
        #     return (self.state, self.reward, done, self.truncated, self.info)

        # self.index += 1
        # self.state = self.generate_next_state(action)
        # return (self.state, self.reward, done, self.truncated, self.info)

    # def buy(self):
    #     close_price = self.state

    #     if close_price > (self.AVAILABLE_AMOUNT - self.BUY_COST):

    #         self.info["action"] = "[BAD BUY] NO_MONEY_TO_BUY"
    #         self.reward = -1_000_000
    #         self.truncated = True
    #         self.bad_buys += 1
    #         return

    #     shares = min(int((self.available_amount // close_price)), self.HMAX)

    #     max_buyable_shares = int(self.available_amount / current_price)

    #     # Buy based on a simple mechanism - we're buying one share for the sake of simplicity.
    #     if max_buyable_shares > 0:
    #         self.available_shares += 1
    #         self.available_amount -= current_price
    #         self.transactions.append(
    #             {"type": "BUY", "amount": 1, "price": current_price}
    #         )
    #         self.successful_buys += 1
    #         self.reward = (self.state[self.close_price_index + 1] - current_price) * 1
    #         self.info["action"] = "[SUCCESSFUL BUY]"
    #     else:
    #         self.unsuccessful_buys += 1
    #         self.reward = -1000000  # penalty for failed buying attempt
    #         self.info["action"] = "[BAD BUY] NO_MONEY_TO_BUY"
    #         self.truncated = True

    # def sell(self):
    #     current_price = self.state[self.close_price_index]

    #     # If we have shares, sell one (for simplicity)
    #     if self.available_shares > 0:
    #         self.available_shares -= 1
    #         self.available_amount += current_price
    #         self.transactions.append(
    #             {"type": "SELL", "amount": 1, "price": current_price}
    #         )
    #         self.successful_sells += 1
    #         self.reward = (current_price - self.state[self.close_price_index - 1]) * 1
    #         self.info["action"] = "[SUCCESSFUL SELL]"
    #     else:
    #         self.unsuccessful_sells += 1
    #         self.reward = -1000000  # penalty for failed selling attempt
    #         self.info["action"] = "[BAD SELL] NO_SHARES_TO_SELL"
    #         self.truncated = True

    # def hold(self):
    #     change_in_price = self.state[self.close_price_index] - self.state[self.close_price_index - 1]

    #     if self.available_shares == 0:
    #         if change_in_price > 0:
    #             # price went up, good decision not to sell
    #             self.reward = 1
    #             self.neutral_holds += 1
    #             self.info["action"] = "[NEUTRAL HOLD]"
    #         elif change_in_price < 0:
    #             # price went down, missed opportunity
    #             self.reward = -1000000
    #             self.bad_holds += 1
    #             self.info["action"] = "[BAD HOLD] MISSED_OPPORTUNITY"
    #             self.truncated = True
    #         else:
    #             # price didn't change, neutral
    #             self.reward = 0
    #             self.neutral_holds += 1
    #             self.info["action"] = "[NEUTRAL HOLD]"
    #     else:
    #         # if you have shares and decide to hold
    #         self.reward = change_in_price
    #         if change_in_price > 0:
    #             self.successful_holds += 1
    #             self.info["action"] = "[SUCCESSFUL HOLD]"
    #         elif change_in_price < 0:
    #             self.unsuccessful_holds += 1
    #             self.info["action"] = "[UNSUCCESSFUL HOLD]"
    #         else:
    #             self.neutral_holds += 1
    #             self.info["action"] = "[NEUTRAL HOLD]"

    # def generate_first_state(self):
    #     state = self.stock_data[self.index]
    #     return state

    # def generate_next_state(self, current_action):

    #     state = self.stock_data[self.index]
    #     state[self.available_amount_index] = self.available_amount
    #     state[self.available_shares_index] = self.available_shares
    #     state[self.cummulative_profit_loss_index] = self.cummulative_profit_loss
    #     state[self.portfolio_value_index] = self.portfolio_value

    #     state[self.previous_action_index_range] = np.roll(
    #         self.state[self.previous_action_index_range], 1
    #     )
    #     state[min(self.previous_action_index_range)] = current_action

    #     return state

    # def generate_info(self):
    #     close_price = self.state[self.close_price_index]

    #     return {
    #         "portfolio_value": self.portfolio_value,
    #         "index": self.index,
    #         "close_price": close_price,
    #         "available_amount": self.available_amount,
    #         "shares_holdings": self.available_shares,
    #         "cummulative_profit_loss": self.cummulative_profit_loss,
    #         "reward": self.reward,
    #         "successful_buys": self.successful_buys,
    #         "successful_holds": self.successful_holds,
    #         "successful_sells": self.successful_sells,
    #         "unsuccessful_sells": self.unsuccessful_sells,
    #         "unsuccessful_buys": self.unsuccessful_buys,
    #         "unsuccessful_holds": self.unsuccessful_holds,
    #         "neutral_holds": self.neutral_holds,
    #         "bad_sells": self.bad_sells,
    #         "bad_buys": self.bad_buys,
    #         "transactions": self.transactions,
    #         "seed": self.seed,
    #     }
