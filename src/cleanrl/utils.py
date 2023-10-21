import torch
import numpy as np
from gymnasium.vector import SyncVectorEnv
import torch.nn as nn
from torch.distributions.categorical import Categorical


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

    envs = SyncVectorEnv(
        [
            make_env(env, array, mode, seed, i)
            for i in range(num_envs)
        ]
    )
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
        # print(probs.probs, action, deterministic)
        if action is None:
            if not deterministic:
                action = probs.sample()
            else:
                action = torch.argmax(probs.probs, dim=1)
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
