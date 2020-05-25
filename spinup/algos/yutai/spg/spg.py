"""
SPG: Stupid/Simple Policy Gradient

The most basic form of policy gradient. Taken from examples.pytorch.pg_math
Thanks for the great explanation, Joshua :) 
"""
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from typing import List, Optional, Callable
from gym.spaces import Discrete, Box

def mlp(sizes: List[int],
        hidden_activation: Optional[Callable] = nn.Tanh,
        output_activation: Optional[Callable] = nn.Identity):
    """
    A nice little feed forward FC NN
    """
    layers = []
    for i in range(len(sizes) - 1): # the in-betweens
        activation_fn = hidden_activation if i < len(sizes) - 2 else output_activation
        layers.extend([nn.Linear(sizes[i], sizes[i+1], activation_fn())])
    return nn.Sequential(*layers)

class Trainer():
    def __init__(self,
                 env_name: Optional[str] = "CartPole-v0",
                 hidden_sizes: Optional[List] = [32],
                 lr: Optional[int] = 1e-2,
                 epochs: Optional[int] = 50,
                 batch_size: Optional[int] = 5000,
                 render: Optional[bool] = False):
        self.env = gym.make(env_name)
        self.obs_dim = self.env.observation_space.shape[0]
        self.num_actions: int = self.env.action_space.n

        self.net = mlp([self.obs_dim] + hidden_sizes + [self.num_actions])
        self.optimizer = Adam(self.net.parameters(), lr=lr)

        self.epochs = epochs
        self.batch_size = batch_size
        self.render = render

    def get_policy(self, obs):
        """
        pass obs into NN, obtain logits, returns a distribution over actions conditioned on obs
        obs: batch_size x obs_dim
        logits: batch_size x num_actions
        """
        logits = self.net(obs)
        return Categorical(logits=logits) 
    
    def get_action(self, obs) -> int:
        """
        Takes one single obs represented as a tensor, returns an action represented by int
        """
        pi = self.get_policy(obs)
        return pi.sample().item()

    def compute_r2g_weights(self, episode_rewards) -> List:
        ep_len = len(episode_rewards)
        weights = np.zeros_like(episode_rewards)
        for i in reversed(range(ep_len)):
            # weights[i] = sum(episode_rewards[i:])

            # slightly more efficient version
            weights[i] = episode_rewards[i] + (weights[i + 1] if i + 1 < ep_len else 0)
        return weights

    def compute_loss(self, obs, action, weights) -> float:
        """
        gradient of this loss wrt NN parameters is the policy gradient
        obs: batch_size x obs_dim
        action: batch_size,
        weights: batch_size,
        """
        pi = self.get_policy(obs)
        log_pi = pi.log_prob(action)
        return -(log_pi * weights).mean()

    def train_one_epoch(self):
        obs_buffer = []
        action_buffer = []
        weight_buffer = []
        return_buffer = []
        len_buffer = []

        obs = self.env.reset()
        done = False
        episode_rewards = []
        finished_rendering_this_epoch = False

        # rollout
        while True:
            if self.render and not finished_rendering_this_epoch:
                self.env.render()
            obs_buffer.append(obs)
            action = self.get_action(torch.as_tensor(obs, dtype=torch.float32))
            action_buffer.append(action)

            obs, reward, done, _ = self.env.step(action)
            episode_rewards.append(reward)

            if done:
                ep_return = sum(episode_rewards)
                ep_len = len(episode_rewards)
                return_buffer.append(ep_return)
                len_buffer.append(ep_len)

                # weight_buffer += [ep_return] * ep_len
                weight_buffer += list(self.compute_r2g_weights(episode_rewards))

                obs, done, episode_rewards = self.env.reset(), False, []
                finished_rendering_this_epoch = True

                if len(obs_buffer) > self.batch_size:
                    # one batch consists of at least batch_size (obs, action, weights) tuple
                    break 
        
        # take a single PG update step
        self.optimizer.zero_grad()
        batch_loss: float = self.compute_loss(obs=torch.as_tensor(obs_buffer,dtype=torch.float32),
                                       action=torch.as_tensor(action_buffer, dtype=torch.int),
                                       weights=torch.as_tensor(weight_buffer, dtype=torch.float32))
        batch_loss.backward()
        self.optimizer.step()
        return batch_loss, return_buffer, len_buffer

    def train(self):
        for epoch in range(self.epochs):
            batch_loss, return_buffer, len_buffer = self.train_one_epoch()
            print(f"epoch: {epoch} \t loss: {batch_loss} \t return: {np.mean(return_buffer)} \t ep_len: {np.mean(len_buffer)}")

if __name__ == '__main__':
    trainer = Trainer(render=False)
    trainer.train()

