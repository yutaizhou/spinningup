import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from typing import Sequence, Tuple, List, Callable, Optional

def combined_shape(length: int, shape = None) -> Tuple:
    if shape is None: 
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)
    
def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def mlp(sizes: List[int],
        hidden_activation: Optional[Callable] = nn.Tanh,
        output_activation: Optional[Callable] = nn.Identity):
    layers = []
    for i in range(len(sizes) - 1): # the in-betweens
        activation_fn = hidden_activation if i < len(sizes) - 2 else output_activation
        layers.extend([nn.Linear(sizes[i], sizes[i+1]), activation_fn()])
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError
    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        log_prob_a = self._log_prob_from_distribution(pi, act) if act is not None else None
        return pi, log_prob_a

class MLPCategoricalActor(Actor):
    def __init__(self,
                 obs_dim: int,
                 num_actions: int,
                 hidden_sizes: List[int],
                 activation: Callable):
        super().__init__()
        self.net = mlp([obs_dim] + hidden_sizes + [num_actions], activation)

    def _distribution(self, obs):
        logits = self.net(obs)
        return Categorical(logits=logits)
    
    def _log_prob_from_distribution(self, pi, act):
       return pi.log_prob(act) 

class MLPGaussianActor(Actor):
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 hidden_sizes: Sequence[int],
                 activation: Callable):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))
        self.net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.net(obs) #long live universal approximation theorem
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1) # perhaps squeeze?

class MLPCritic(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 hidden_sizes: Sequence[int],
                 activation: Callable):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.net(obs), -1)

class MLPActorCritic(nn.Module):
    def __init__(self,
                 obs_space,
                 act_space,
                 hidden_sizes: Sequence[int],
                 activation: Optional[Callable] = nn.Tanh):
        super().__init__()
        obs_dim: int = obs_space.shape[0]

        # build policy 
        if isinstance(act_space, Box): # Continuous action spaces
            act_dim = act_space.shape[0]
            self.actor = MLPGaussianActor(obs_dim, act_dim, hidden_sizes, activation)
        if isinstance(act_space, Discrete): # Discrete action spaces
            num_actions = act_space.n
            self.actor = MLPCategoricalActor(obs_dim, num_actions, hidden_sizes, activation)
        
        # build value function
        self.critic = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.actor._distribution(obs)
            a = pi.sample()
            logprob_a = self.actor._log_prob_from_distribution(pi, a)
            v = self.critic(obs)

        return a.numpy(), v.numpy(), logprob_a.numpy()
    
    def act(self, obs):
        return self.step(obs)[0]