import numpy as np
import polars as pl
from pathlib import Path
from envs.stock_trading_env import StockTradingEnv
import random
import torch
from clean_rl_ppo_agent import Agent
from gymnasium.vector import SyncVectorEnv
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils import (
    load_data,
    add_technical_indicators,
    add_past_hours,
    train_test_split,
    create_torch_array,
    make_env,
)
from tqdm import tqdm
from time import perf_counter

TICKERS = "SBIN.NS"
INTERVAL = "1h"
PERIOD = "360d"
MODEL_PREFIX = f"{TICKERS}_PPO"

TRAIN_TEST_SPLIT_PERCENT = 0.15
PAST_HOURS = range(1, 15)
TECHNICAL_INDICATORS = [f"PAST_{hour}_HOUR" for hour in PAST_HOURS]
DATASET = Path("datasets")
DATASET.mkdir(parents=True, exist_ok=True)
TRAINED_MODEL_DIR = Path("trained_models")
TRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_LOG_DIR = Path("tensorboard_log")
TENSORBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_FILE = TRAINED_MODEL_DIR / "clean_rl_agent_ppo.pt"

SEED = 1337
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 5e-4
EPS = 1e-5
TOTAL_TIMESTEPS = 10_000_000
NUM_STEPS = 1024
NUM_ENVS = 2**5
EVAL_NUM_ENVS = 2 * 4
BATCH_SIZE = NUM_ENVS * NUM_STEPS
NUM_MINIBATCHES = 32
MINIBATCH_SIZE = BATCH_SIZE // NUM_MINIBATCHES
NUM_UPDATES = TOTAL_TIMESTEPS // BATCH_SIZE
GAE_LAMBDA = 0.95
GAMMA = 0.99
UPDATE_EPOCHS = 8
NORM_ADV = True
CLIP_COEF = 0.2
CLIP_VLOSS = True
ENT_COEF = 0.04
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
TARGET_KL = None
CHECKPOINT_FREQUENCY = 1


SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def main():
    df = load_data()
    df = add_past_hours(df)
    df = add_technical_indicators(df)
    df = df.with_columns(pl.lit(0.0).alias("Buy/Sold/Hold"))
    train_df, trade_df = train_test_split(df)

    assert train_df.columns == trade_df.columns

    train_arrays = create_torch_array(train_df, device=DEVICE)
    trade_arrays = create_torch_array(trade_df, device=DEVICE)

    writer = SummaryWriter(TENSORBOARD_LOG_DIR / "clean_rl_ppo_cummulative_reward")
    # writer = SummaryWriter(TENSORBOARD_LOG_DIR / "clean_rl_ppo_each_step_reward")

    # env setup
    train_envs = SyncVectorEnv(
        [
            make_env(StockTradingEnv, train_arrays, TICKERS, True, SEED, i)
            for i in range(NUM_ENVS)
        ]
    )
    trade_envs = SyncVectorEnv(
        [
            make_env(StockTradingEnv, trade_arrays, TICKERS, True, SEED, i)
            for i in range(EVAL_NUM_ENVS)
        ]
    )

    train_agent = Agent(train_envs).to(DEVICE)
    trade_agent = Agent(trade_envs).to(DEVICE)
    # if MODEL_SAVE_FILE.exists():
    #     print(f"Loading existing model from {MODEL_SAVE_FILE}")
    #     train_agent.load_state_dict(torch.load(MODEL_SAVE_FILE, map_location=DEVICE))

    optimizer = optim.Adam(train_agent.parameters(), lr=LEARNING_RATE, eps=EPS)

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (NUM_STEPS, NUM_ENVS) + train_envs.single_observation_space.shape
    ).to(DEVICE)
    actions = torch.zeros(
        (NUM_STEPS, NUM_ENVS) + train_envs.single_action_space.shape
    ).to(DEVICE)
    logprobs = torch.zeros((NUM_STEPS, NUM_ENVS)).to(DEVICE)
    rewards = torch.zeros((NUM_STEPS, NUM_ENVS)).to(DEVICE)
    dones = torch.zeros((NUM_STEPS, NUM_ENVS)).to(DEVICE)
    values = torch.zeros((NUM_STEPS, NUM_ENVS)).to(DEVICE)

    # TRY NOT TO MODIFY: start the game
    global_step = int(Path("global_step").read_text())
    next_obs, _ = train_envs.reset()
    next_done = torch.zeros(NUM_ENVS).to(DEVICE)

    for update in tqdm(range(1, NUM_UPDATES + 1)):
        start_time = perf_counter()
        frac = 1.0 - (update - 1.0) / NUM_UPDATES
        lrnow = frac * LEARNING_RATE
        optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, NUM_STEPS):
            global_step += 1 * NUM_ENVS
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = train_agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, _, info = train_envs.step(action)
            rewards[step] = torch.tensor(reward).to(DEVICE).view(-1)
            next_done = torch.Tensor(done).to(DEVICE)

            # dict_keys(['final_observation', '_final_observation', 'final_info', '_final_info'])
            if "final_info" in info:
                final_info = info["final_info"]
                for i, fi in enumerate(final_info):
                    if fi is None:
                        continue
                    for k, v in fi.items():
                        if k not in ["action"]:
                            writer.add_scalar(f"train/{i}/{k}", v, global_step)
                break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = train_agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(DEVICE)
            lastgaelam = 0
            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1:
                    nextnonterminal = 1.0 - next_done
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
        b_obs = obs.reshape((-1,) + train_envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + train_envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(BATCH_SIZE)
        clipfracs = []
        for epoch in range(UPDATE_EPOCHS):
            for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = train_agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
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

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if update % CHECKPOINT_FREQUENCY == 0:
            infosss = []
            best_cummulative_profit_loss = 0
            best_cummulative_profit_loss_index = 0
            trade_agent.load_state_dict(train_agent.state_dict())
            trade_agent.eval()
            trade_obs, _ = trade_envs.reset()
            with torch.inference_mode():
                while True:
                    t_action, _, _, _ = train_agent.get_action_and_value(trade_obs)
                    trade_obs, _, _, _, t_info = trade_envs.step(t_action)
                    infosss.append(t_info)
                    if "final_info" in t_info:
                        final_info = t_info["final_info"]
                        for i, fi in enumerate(final_info):
                            if fi is None:
                                continue
                            if (
                                fi["cummulative_profit_loss"]
                                > best_cummulative_profit_loss
                            ):
                                best_cummulative_profit_loss_index = i
                                best_cummulative_profit_loss = fi[
                                    "cummulative_profit_loss"
                                ]
                            for k, v in fi.items():
                                if k not in ["action"]:
                                    writer.add_scalar(f"trade/{i}/{k}", v, global_step)
                        print(
                            f"{best_cummulative_profit_loss_index} {final_info[best_cummulative_profit_loss_index]}"
                        )
                        break

            torch.save(train_agent.state_dict(), MODEL_SAVE_FILE)
            # df = pl.DataFrame(infosss)
            # cols = df.columns
            # cols = [col for col in cols if not col.startswith("_")]
            # df.select(cols).write_csv("trade_results.csv")

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar(
            "time/each_run", round(perf_counter() - start_time, 2), global_step
        )

    Path("global_step").write_text(str(global_step))
    train_envs.close()
    writer.close()


if __name__ == "__main__":
    main()
