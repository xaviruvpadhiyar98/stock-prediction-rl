from envs.stock_trading_env_tensor import StockTradingEnv
from envs.stock_trading_env_tensor_buy_sell import StockTradingEnv as OnlyBuySellEnv

from sb.utils import (
    load_data,
    makedirs,
    create_numpy_array,
)
from cleanrl.utils import create_torch_array, create_envs, Agent
from pathlib import Path
import torch
import torch.optim as optim
from tqdm import tqdm
from time import perf_counter
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import json


LEARNING_RATE = 1e-3
EPS = 1e-5
TOTAL_TIMESTEPS = 2_000_000
NUM_STEPS = 1024
NUM_ENVS = 16
BATCH_SIZE = NUM_ENVS * NUM_STEPS
NUM_MINIBATCHES = 64
MINIBATCH_SIZE = BATCH_SIZE // NUM_MINIBATCHES
NUM_UPDATES = TOTAL_TIMESTEPS // BATCH_SIZE
GAE_LAMBDA = 0.95
GAMMA = 0.99
UPDATE_EPOCHS = 10
NORM_ADV = True
CLIP_COEF = 0.2
CLIP_VLOSS = True
ENT_COEF = 0.1
VF_COEF = 0.5
MAX_GRAD_NORM = 0.8
TARGET_KL = None
CHECKPOINT_FREQUENCY = 1


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ticker = "SBIN.NS"
    trained_model_dir = Path("trained_models")
    tensorboard_log = Path("tensorboard_log")
    model_name = "PPO"
    seed = 40
    cols = ["index","shares_holdings","cummulative_profit_loss","portfolio_value","unsuccessful_sells","successful_sells","successful_buys","successful_holds", "bad_buys", "bad_sells"]
    makedirs()
    train_df, trade_df = load_data(ticker)
    train_arrays = create_torch_array(create_numpy_array(train_df), device)
    trade_arrays = create_torch_array(create_numpy_array(trade_df), device)

    tb_log_name = f"{model_name}_{ticker}_{NUM_STEPS}_{NUM_ENVS}_OnlyBuySellEnv_cleanrl"

    writer = SummaryWriter(tensorboard_log / tb_log_name)



    train_envs = create_envs(
        OnlyBuySellEnv, train_arrays, num_envs=NUM_ENVS, mode="train", seed=seed
    )
    trade_envs = create_envs(
        OnlyBuySellEnv, trade_arrays, num_envs=1, mode="trade", seed=seed
    )

    model_filename = trained_model_dir / f"{model_name}_{ticker}_OnlyBuySellEnv_cleanrl"
    train_agent = Agent(train_envs).to(device)
    trade_agent = Agent(trade_envs).to(device)
    if model_filename.exists():
        print(f"Loading existing model from {model_filename}")
        train_agent.load_state_dict(torch.load(model_filename, map_location=device))
    optimizer = optim.Adam(train_agent.parameters(), lr=LEARNING_RATE, eps=EPS)


    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (NUM_STEPS, NUM_ENVS) + train_envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (NUM_STEPS, NUM_ENVS) + train_envs.single_action_space.shape
    ).to(device)

    logprobs = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    rewards = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    values = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    dones = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    next_done = torch.zeros(NUM_ENVS).to(device)
    
    global_step = int(Path("global_step").read_text())
    next_obs, _ = train_envs.reset()


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
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_done = torch.Tensor(done).to(device)

            if "final_info" in info:
                final_info = info["final_info"]
                for i in final_info:
                    if i is not None:
                        for col in cols:
                            writer.add_scalar(f'train/{col}/{i["seed"]}', i[col], global_step)



        # bootstrap value if not done
        with torch.no_grad():
            next_value = train_agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
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


        trade_agent.load_state_dict(train_agent.state_dict())
        trade_agent.eval()
        trade_obs, _ = trade_envs.reset()

        with torch.inference_mode():
            while True:
                t_action, _, _, _ = train_agent.get_action_and_value(trade_obs, deterministic=True)
                trade_obs, _, _, _, t_info = trade_envs.step(t_action)
                if "final_info" in t_info:
                    final_info = t_info["final_info"]
                    for i in final_info:
                        if i is not None:
                            print(json.dumps(i, indent=4, default=str))
                            for col in cols:
                                writer.add_scalar(f'trade/{col}/{i["seed"]}', i[col], global_step)
                    break

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
    # print(global_step)
    
    torch.save(train_agent.state_dict(), model_filename)
    train_envs.close()
    trade_envs.close()
    writer.close()

if __name__ == "__main__":
    main()
