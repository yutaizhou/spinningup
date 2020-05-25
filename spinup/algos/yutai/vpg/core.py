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