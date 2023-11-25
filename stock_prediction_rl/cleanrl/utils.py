import torch
import numpy as np
from gymnasium.vector import SyncVectorEnv
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.optim as optim
from tqdm import tqdm


def create_torch_array(arr, device="cuda:0"):
    arr = np.asarray(arr).astype(np.float32)
    return torch.from_numpy(arr).to(device)


def make_env(env_id, array, mode, seed, rank):
    def thunk():
        env = env_id(array, mode, seed=seed + rank)
        env.reset(seed=seed + rank)
        return env

    return thunk


def create_envs(env, array, mode, num_envs=1, seed=1337):
    envs = SyncVectorEnv([make_env(env, array, mode, seed, i) for i in range(num_envs)])
    return envs


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
        self.actor = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, deterministic=False):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            if not deterministic:
                action = probs.sample()
            else:
                action = torch.argmax(probs.probs, dim=1)
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def sample_ppo_params(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    num_steps = trial.suggest_categorical("num_steps", [64, 128, 256, 512, 1024, 2048])
    num_minibatches = trial.suggest_categorical(
        "num_minibatches", [8, 16, 32, 64, 128, 256, 512]
    )

    gamma = trial.suggest_categorical(
        "gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
    )

    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    clip_coef = trial.suggest_categorical("clip_coef", [0.1, 0.2, 0.3, 0.4])
    gae_lambda = trial.suggest_categorical(
        "gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0]
    )
    max_grad_norm = trial.suggest_categorical(
        "max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5]
    )
    vf_coef = trial.suggest_float("vf_coef", 0.01, 1, log=True)
    update_epochs = trial.suggest_categorical("update_epochs", [1, 5, 10, 20])

    if num_minibatches > num_steps:
        num_minibatches = num_steps

    return {
        "num_steps": num_steps,
        "gamma": gamma,
        "num_minibatches": num_minibatches,
        "update_epochs": update_epochs,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_coef": clip_coef,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
    }


def train(env, train_arrays, trade_arrays, device, seed, hp):
    eps = 1e-5
    num_envs = 16
    total_timesteps = 1_000_000

    lr = hp["learning_rate"]
    num_steps = hp["num_steps"]
    gamma = hp["gamma"]
    gae_lambda = hp["gae_lambda"]
    num_minibatches = hp["num_minibatches"]
    update_epochs = hp["update_epochs"]
    clip_coef = hp["clip_coef"]
    ent_coef = hp["ent_coef"]
    vf_coef = hp["vf_coef"]
    max_grad_norm = hp["max_grad_norm"]

    batch_size = num_steps * num_envs
    minibatch_size = batch_size // num_minibatches
    num_updates = total_timesteps // batch_size

    train_envs = create_envs(
        env, train_arrays, num_envs=num_envs, mode="train", seed=seed
    )
    trade_envs = create_envs(env, trade_arrays, num_envs=1, mode="trade", seed=seed)
    train_agent = Agent(train_envs).to(device)
    trade_agent = Agent(trade_envs).to(device)

    optimizer = optim.Adam(train_agent.parameters(), lr=lr, eps=eps)

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (num_steps, num_envs) + train_envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (num_steps, num_envs) + train_envs.single_action_space.shape
    ).to(device)

    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    next_done = torch.zeros(num_envs).to(device)

    global_step = 0
    next_obs, _ = train_envs.reset()

    for update in tqdm(range(1, num_updates + 1)):
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * lr
        optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, num_steps):
            global_step += 1 * num_envs
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

        # bootstrap value if not done
        with torch.no_grad():
            next_value = train_agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + gamma * gae_lambda * nextnonterminal * lastgaelam
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
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(update_epochs):
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
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
                        ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - clip_coef, 1 + clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -clip_coef,
                    clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(train_agent.parameters(), max_grad_norm)
                optimizer.step()

    trade_agent.load_state_dict(train_agent.state_dict())
    trade_agent.eval()
    trade_obs, _ = trade_envs.reset()

    cummulative_profit_loss = 0
    with torch.inference_mode():
        while True:
            t_action, _, _, _ = train_agent.get_action_and_value(
                trade_obs, deterministic=True
            )
            trade_obs, _, _, _, t_info = trade_envs.step(t_action)
            if "final_info" in t_info:
                final_info = t_info["final_info"]
                for i in final_info:
                    if i is not None:
                        cummulative_profit_loss = i["cummulative_profit_loss"]
                break

    train_envs.close()
    trade_envs.close()
    return cummulative_profit_loss
