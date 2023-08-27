import numpy as np
import polars as pl
import yfinance as yf
from pathlib import Path
from collections import deque
from gymnasium import Env, spaces
from gymnasium.vector import SyncVectorEnv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import random
from time import perf_counter


TICKERS = "SBIN.NS"
INTERVAL = "1h"
PERIOD = "360d"
MODEL_PREFIX = f"{TICKERS}_PPO"
SEED = 1337
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 3e-4
EPS = 1e-5
TOTAL_TIMESTEPS = 10000
NUM_STEPS = 2048
NUM_ENVS = 1
BATCH_SIZE = NUM_ENVS * NUM_STEPS
MINI_BATCH_SIZE = BATCH_SIZE // 32
NUM_UPDATES = 10
GAE_LAMBDA = 0.95
GAMMA = 0.99
UPDATE_EPOCHS = 10
NORM_ADV = True
CLIP_COEF = 0.2
CLIP_VLOSS = True
ENT_COEF = 0.0
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
TARGET_KL = None

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


TRAIN_TEST_SPLIT_PERCENT = 0.15
PAST_HOURS = range(1, 15)
TECHNICAL_INDICATORS = [f"PAST_{hour}_HOUR" for hour in PAST_HOURS]
DATASET = Path("datasets")
TRAINED_MODEL_DIR = Path("trained_models")
TENSORBOARD_LOG_DIR = Path("tensorboard_log")


def load_data():
    """returns following dataframe
    shape: (2_508, 3)
    ┌─────────────────────────┬────────────┬─────────┐
    │ Datetime                ┆ Close      ┆ Ticker  │
    │ ---                     ┆ ---        ┆ ---     │
    │ datetime[ns, UTC]       ┆ f64        ┆ str     │
    ╞═════════════════════════╪════════════╪═════════╡
    │ 2022-03-15 03:45:00 UTC ┆ 485.5      ┆ SBIN.NS │
    │ 2022-03-15 04:45:00 UTC ┆ 486.700012 ┆ SBIN.NS │
    │ 2022-03-15 05:45:00 UTC ┆ 488.549988 ┆ SBIN.NS │
    │ 2022-03-15 06:45:00 UTC ┆ 485.049988 ┆ SBIN.NS │
    │ …                       ┆ …          ┆ …       │
    │ 2023-08-25 06:45:00 UTC ┆ 571.099976 ┆ SBIN.NS │
    │ 2023-08-25 07:45:00 UTC ┆ 570.799988 ┆ SBIN.NS │
    │ 2023-08-25 08:45:00 UTC ┆ 569.799988 ┆ SBIN.NS │
    │ 2023-08-25 09:45:00 UTC ┆ 569.700012 ┆ SBIN.NS │
    └─────────────────────────┴────────────┴─────────┘
    """
    ticker_file = DATASET / TICKERS
    if not ticker_file.exists():
        yf.download(
            tickers=TICKERS,
            period=PERIOD,
            interval=INTERVAL,
            group_by="Ticker",
            auto_adjust=True,
            prepost=True,
        ).reset_index().to_parquet(ticker_file, index=False, engine="fastparquet")
    df = pl.read_parquet(ticker_file).select(["Datetime", "Close"])
    df = df.with_columns(pl.lit(TICKERS).alias("Ticker"))
    df = df.sort("Datetime", descending=False)
    return df


def add_past_hours(df):
    """
    shape: (2_494, 17)
    ┌─────────────┬────────────┬─────────┬─────────────┬───┬─────────────┬─────────────┬─────────────┬────────────┐
    │ Datetime    ┆ Close      ┆ Ticker  ┆ PAST_1_HOUR ┆ … ┆ PAST_11_HOU ┆ PAST_12_HOU ┆ PAST_13_HOU ┆ PAST_14_HO │
    │ ---         ┆ ---        ┆ ---     ┆ ---         ┆   ┆ R           ┆ R           ┆ R           ┆ UR         │
    │ datetime[ns ┆ f64        ┆ str     ┆ f64         ┆   ┆ ---         ┆ ---         ┆ ---         ┆ ---        │
    │ , UTC]      ┆            ┆         ┆             ┆   ┆ f64         ┆ f64         ┆ f64         ┆ f64        │
    ╞═════════════╪════════════╪═════════╪═════════════╪═══╪═════════════╪═════════════╪═════════════╪════════════╡
    │ 2022-03-17  ┆ 500.799988 ┆ SBIN.NS ┆ 491.75      ┆ … ┆ 485.049988  ┆ 488.549988  ┆ 486.700012  ┆ 485.5      │
    │ 03:45:00    ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ UTC         ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ 2022-03-17  ┆ 501.450012 ┆ SBIN.NS ┆ 500.799988  ┆ … ┆ 482.950012  ┆ 485.049988  ┆ 488.549988  ┆ 486.700012 │
    │ 04:45:00    ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ UTC         ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ 2022-03-17  ┆ 502.100006 ┆ SBIN.NS ┆ 501.450012  ┆ … ┆ 486.049988  ┆ 482.950012  ┆ 485.049988  ┆ 488.549988 │
    │ 05:45:00    ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ UTC         ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ 2022-03-17  ┆ 501.799988 ┆ SBIN.NS ┆ 502.100006  ┆ … ┆ 485.100006  ┆ 486.049988  ┆ 482.950012  ┆ 485.049988 │
    │ 06:45:00    ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ UTC         ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ …           ┆ …          ┆ …       ┆ …           ┆ … ┆ …           ┆ …           ┆ …           ┆ …          │
    │ 2023-08-25  ┆ 571.099976 ┆ SBIN.NS ┆ 571.549988  ┆ … ┆ 576.900024  ┆ 576.849976  ┆ 577.75      ┆ 573.700012 │
    │ 06:45:00    ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ UTC         ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ 2023-08-25  ┆ 570.799988 ┆ SBIN.NS ┆ 571.099976  ┆ … ┆ 580.700012  ┆ 576.900024  ┆ 576.849976  ┆ 577.75     │
    │ 07:45:00    ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ UTC         ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ 2023-08-25  ┆ 569.799988 ┆ SBIN.NS ┆ 570.799988  ┆ … ┆ 577.900024  ┆ 580.700012  ┆ 576.900024  ┆ 576.849976 │
    │ 08:45:00    ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ UTC         ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ 2023-08-25  ┆ 569.700012 ┆ SBIN.NS ┆ 569.799988  ┆ … ┆ 576.700012  ┆ 577.900024  ┆ 580.700012  ┆ 576.900024 │
    │ 09:45:00    ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    │ UTC         ┆            ┆         ┆             ┆   ┆             ┆             ┆             ┆            │
    └─────────────┴────────────┴─────────┴─────────────┴───┴─────────────┴─────────────┴─────────────┴────────────┘
    """  # noqa: E501
    df = df.with_columns(
        [pl.col("Close").shift(hour).alias(f"PAST_{hour}_HOUR") for hour in PAST_HOURS]
    )
    df = df.drop_nulls()
    return df


def train_test_split(df):
    """
    train_df ->
    ┌──────────────┬────────────┬─────────┬─────────────┬───┬──────────────┬──────────────┬──────────────┬──────────────┐
    │ Datetime     ┆ Close      ┆ Ticker  ┆ PAST_1_HOUR ┆ … ┆ PAST_12_HOUR ┆ PAST_13_HOUR ┆ PAST_14_HOUR ┆ Buy/Sold/Hol │
    │ ---          ┆ ---        ┆ ---     ┆ ---         ┆   ┆ ---          ┆ ---          ┆ ---          ┆ d            │
    │ datetime[ns, ┆ f64        ┆ str     ┆ f64         ┆   ┆ f64          ┆ f64          ┆ f64          ┆ ---          │
    │ UTC]         ┆            ┆         ┆             ┆   ┆              ┆              ┆              ┆ f64          │
    ╞══════════════╪════════════╪═════════╪═════════════╪═══╪══════════════╪══════════════╪══════════════╪══════════════╡
    │ 2023-06-07   ┆ 588.700012 ┆ SBIN.NS ┆ 589.099976  ┆ … ┆ 583.400024   ┆ 585.5        ┆ 587.0        ┆ 0.0          │
    │ 09:45:00 UTC ┆            ┆         ┆             ┆   ┆              ┆              ┆              ┆              │
    │ 2023-06-08   ┆ 592.700012 ┆ SBIN.NS ┆ 588.700012  ┆ … ┆ 584.099976   ┆ 583.400024   ┆ 585.5        ┆ 0.0          │
    │ 03:45:00 UTC ┆            ┆         ┆             ┆   ┆              ┆              ┆              ┆              │
    │ 2023-06-08   ┆ 593.25     ┆ SBIN.NS ┆ 592.700012  ┆ … ┆ 584.400024   ┆ 584.099976   ┆ 583.400024   ┆ 0.0          │
    │ 04:45:00 UTC ┆            ┆         ┆             ┆   ┆              ┆              ┆              ┆              │
    │ 2023-06-08   ┆ 589.900024 ┆ SBIN.NS ┆ 593.25      ┆ … ┆ 584.549988   ┆ 584.400024   ┆ 584.099976   ┆ 0.0          │
    │ 05:45:00 UTC ┆            ┆         ┆             ┆   ┆              ┆              ┆              ┆              │
    │ 2023-06-08   ┆ 590.900024 ┆ SBIN.NS ┆ 589.900024  ┆ … ┆ 586.099976   ┆ 584.549988   ┆ 584.400024   ┆ 0.0          │
    │ 06:45:00 UTC ┆            ┆         ┆             ┆   ┆              ┆              ┆              ┆              │
    └──────────────┴────────────┴─────────┴─────────────┴───┴──────────────┴──────────────┴──────────────┴──────────────┘


    trade_df ->
    ┌──────────────┬────────────┬─────────┬─────────────┬───┬──────────────┬──────────────┬──────────────┬──────────────┐
    │ Datetime     ┆ Close      ┆ Ticker  ┆ PAST_1_HOUR ┆ … ┆ PAST_12_HOUR ┆ PAST_13_HOUR ┆ PAST_14_HOUR ┆ Buy/Sold/Hol │
    │ ---          ┆ ---        ┆ ---     ┆ ---         ┆   ┆ ---          ┆ ---          ┆ ---          ┆ d            │
    │ datetime[ns, ┆ f64        ┆ str     ┆ f64         ┆   ┆ f64          ┆ f64          ┆ f64          ┆ ---          │
    │ UTC]         ┆            ┆         ┆             ┆   ┆              ┆              ┆              ┆ f64          │
    ╞══════════════╪════════════╪═════════╪═════════════╪═══╪══════════════╪══════════════╪══════════════╪══════════════╡
    │ 2023-06-08   ┆ 593.099976 ┆ SBIN.NS ┆ 590.900024  ┆ … ┆ 585.299988   ┆ 586.099976   ┆ 584.549988   ┆ 0.0          │
    │ 07:45:00 UTC ┆            ┆         ┆             ┆   ┆              ┆              ┆              ┆              │
    │ 2023-06-08   ┆ 588.549988 ┆ SBIN.NS ┆ 593.099976  ┆ … ┆ 587.200012   ┆ 585.299988   ┆ 586.099976   ┆ 0.0          │
    │ 08:45:00 UTC ┆            ┆         ┆             ┆   ┆              ┆              ┆              ┆              │
    │ 2023-06-08   ┆ 588.5      ┆ SBIN.NS ┆ 588.549988  ┆ … ┆ 588.75       ┆ 587.200012   ┆ 585.299988   ┆ 0.0          │
    │ 09:45:00 UTC ┆            ┆         ┆             ┆   ┆              ┆              ┆              ┆              │
    │ 2023-06-09   ┆ 584.700012 ┆ SBIN.NS ┆ 588.5       ┆ … ┆ 588.0        ┆ 588.75       ┆ 587.200012   ┆ 0.0          │
    │ 03:45:00 UTC ┆            ┆         ┆             ┆   ┆              ┆              ┆              ┆              │
    │ 2023-06-09   ┆ 580.900024 ┆ SBIN.NS ┆ 584.700012  ┆ … ┆ 590.0        ┆ 588.0        ┆ 588.75       ┆ 0.0          │
    │ 04:45:00 UTC ┆            ┆         ┆             ┆   ┆              ┆              ┆              ┆              │
    └──────────────┴────────────┴─────────┴─────────────┴───┴──────────────┴──────────────┴──────────────┴──────────────┘
    """  # noqa: E501
    total = df.shape[0]
    train_size = total - int(total * TRAIN_TEST_SPLIT_PERCENT)
    train_df = df.slice(0, train_size)
    trade_df = df.slice(train_size, total)
    return train_df, trade_df


def create_torch_array(df):
    """
    returns array
    [
        [593.09997559 590.90002441 589.90002441 ... 586.09997559 584.54998779
            0.        ]
        [588.54998779 593.09997559 590.90002441 ... 585.29998779 586.09997559
            0.        ]
        [588.5        588.54998779 593.09997559 ... 587.20001221 585.29998779
            0.        ]
        ...
        [570.79998779 571.09997559 571.54998779 ... 576.84997559 577.75
            0.        ]
        [569.79998779 570.79998779 571.09997559 ... 576.90002441 576.84997559
            0.        ]
        [569.70001221 569.79998779 570.79998779 ... 580.70001221 576.90002441
            0.        ]
    ]
    """
    cols = df.columns
    cols.remove("Datetime")
    cols.remove("Ticker")
    arr = []
    for i, (name, data) in enumerate(df.groupby("Datetime")):
        new_arr = data.select(cols).to_numpy().flatten()
        arr.append(new_arr)
    arr = np.asarray(arr).astype(np.float32)
    return torch.from_numpy(arr).to(DEVICE)


class StockTradingEnv(Env):
    HMAX = 1
    AMOUNT = torch.Tensor([10_000]).to(DEVICE)
    BUY_COST = SELL_COST = 0.001

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

            # print(f"Holding {self.state[-1]} Available Amount {self.state[0]:.2f}")

        holdings = self.get_holdings()
        self.HOLDINGS.append(holdings)
        # self.reward = self.calculate_reward()
        self.state = self.generate_state()
        self.info = {
            "holdings": holdings,
            "available_amount": self.state[0],
            "change_in_holdings": self.HOLDINGS[-2] - self.HOLDINGS[-1],
            "shares_holding": self.state[-1],
            "reward": self.reward,
            "action": actions,
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
            # self.tracking_buy_sell.append({"buy": buy_prices_with_commission})
            # print(f"Bought {shares} at {buy_prices_with_commission:.2f}")

            current_portfolio_value = self.get_holdings()
            portfolio_change = current_portfolio_value - self.AMOUNT
            stock_profit = current_portfolio_value - buy_prices_with_commission
            self.reward = portfolio_change + stock_profit

    def sell(self):
        close_price = self.state[1]
        shares = self.state[-1]

        if shares > 0:
            shares = min(shares, self.HMAX)
            sell_prices_with_commission = (close_price * (1 + self.BUY_COST)) * shares
            self.state[0] += sell_prices_with_commission
            self.state[-1] -= shares
            # self.tracking_buy_sell.append({"sell": sell_prices_with_commission})
            # print(f"Sold {shares} at {sell_prices_with_commission:.2f}")

            current_portfolio_value = self.get_holdings()
            portfolio_change = current_portfolio_value - self.AMOUNT
            stock_profit = sell_prices_with_commission - current_portfolio_value
            self.reward = portfolio_change + stock_profit

    def hold(self):
        current_portfolio_value = self.get_holdings()
        self.reward = current_portfolio_value - self.AMOUNT

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

    def calculate_reward(self):
        diff = self.AMOUNT - self.HOLDINGS[-1]
        if diff > 0:
            return -diff
        return diff

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


def test_model(env, model, n_times=1):
    for _ in range(n_times):
        obs, _ = env.reset()
        while True:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            if done:
                break
        # print(info)
    return info


def make_env(env_id, array, tickers):
    def thunk():
        env = env_id(array, [tickers])
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(
                nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01
            ),
        )
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, np.prod(envs.single_action_space.shape))
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


def main():
    df = load_data()
    df = add_past_hours(df)
    df = df.with_columns(pl.lit(0.0).alias("Buy/Sold/Hold"))
    train_df, trade_df = train_test_split(df)

    assert train_df.columns == trade_df.columns

    train_arrays = create_torch_array(train_df)
    trade_arrays = create_torch_array(trade_df)

    train_env = SyncVectorEnv([make_env(StockTradingEnv, train_arrays, [TICKERS])])
    trade_env = SyncVectorEnv([make_env(StockTradingEnv, trade_arrays, [TICKERS])])
    assert isinstance(
        train_env.single_action_space, spaces.Box
    ), "only continuous action space is supported"  # noqa: E501

    writer = SummaryWriter(TENSORBOARD_LOG_DIR / "PPO_CLEAN_RL")
    train_agent = Agent(train_env).to(DEVICE)
    trade_agent = Agent(trade_env).to(DEVICE)

    optimizer = optim.Adam(train_agent.parameters(), lr=LEARNING_RATE, eps=EPS)

    len_of_train_array = len(train_arrays)
    rewards = torch.zeros((len_of_train_array, NUM_ENVS)).to(DEVICE)
    dones = torch.zeros((len_of_train_array, NUM_ENVS)).to(DEVICE)
    values = torch.zeros((len_of_train_array, NUM_ENVS)).to(DEVICE)
    logprobs = torch.zeros((len_of_train_array, NUM_ENVS)).to(DEVICE)
    actions = torch.zeros(
        (len_of_train_array, NUM_ENVS) + train_env.single_action_space.shape
    ).to(DEVICE)
    observations = torch.zeros(
        (len_of_train_array, NUM_ENVS) + train_env.single_observation_space.shape
    ).to(DEVICE)
    global_step = 0

    for update in range(1, NUM_UPDATES + 1):
        start_time = perf_counter()

        # Annealing the rate if instructed to do so.
        frac = 1.0 - (update - 1.0) / NUM_UPDATES
        lrnow = frac * LEARNING_RATE
        optimizer.param_groups[0]["lr"] = lrnow

        step = 0
        obs, _ = train_env.reset(seed=SEED)
        while True:
            with torch.inference_mode():
                action, logprob, entropy, value = train_agent.get_action_and_value(obs)
            obs, reward, terminated, truncated, infos = train_env.step(action)
            done = np.logical_or(terminated, truncated)
            done = torch.Tensor(done).to(DEVICE)
            rewards[step] = torch.tensor(reward).to(DEVICE).view(-1)
            dones[step] = done
            actions[step] = action
            logprobs[step] = logprob
            values[step] = value
            observations[step] = obs
            step += 1
            if done:
                print(infos["final_info"][0])
                break


        with torch.inference_mode():
            next_value = train_agent.get_value(obs)
            advantages = torch.zeros_like(rewards).to(DEVICE)
            lastgaelam = 0
            for t in reversed(range(len_of_train_array)):
                if t == len_of_train_array - 1:
                    nextnonterminal = 1.0 - done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
                )
            returns = advantages + values


        # flatten the batch
        b_obs = observations.reshape((-1,) + train_env.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + train_env.single_action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)


        # Optimizing the policy and value network
        b_inds = torch.arange(BATCH_SIZE).to(DEVICE)
        clipfracs = []



        for start in range(0, BATCH_SIZE, MINI_BATCH_SIZE):
            end = start + MINI_BATCH_SIZE
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = train_agent.get_action_and_value(
                b_obs[mb_inds], b_actions[mb_inds]
            )
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.inference_mode():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [
                    ((ratio - 1.0).abs() > CLIP_COEF).float().mean().item()
                ]

            mb_advantages = b_advantages[mb_inds]
            if NORM_ADV:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(
                ratio, 1 - CLIP_COEF, 1 + CLIP_COEF
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if CLIP_VLOSS:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -CLIP_COEF,
                    CLIP_COEF,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - ENT_COEF * entropy_loss + v_loss * VF_COEF

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(train_agent.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            if TARGET_KL is not None:
                if approx_kl > TARGET_KL:
                    break

        var_y = torch.var(b_returns)
        explained_var = (
            np.nan if var_y == 0 else 1 - torch.var(b_returns - b_values) / var_y
        )

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print(f"SPS: {perf_counter() - start_time:.2f}")
        global_step += 1

    train_env.close()


if __name__ == "__main__":
    main()
