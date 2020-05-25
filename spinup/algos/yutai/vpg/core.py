import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from typing import Tuple, List

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
