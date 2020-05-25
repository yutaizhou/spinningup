import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.vpg.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

class Buffer:
    """
    don't confuse this with replay buffer...
    """
    def __init__(self,
                 obs_dim,
                 act_dim,
                 max_size: int,
                 gamma: float = 0.99,
                 lam: float = 0.95):
        self.obs_buffer = np.zeros(core.combined_shape(max_size, obs_dim), dtype=np.float)
        self.act_buffer = np.zeros(core.combined_shape(max_size, act_dim), dtype=np.float)
        self.reward_buffer = np.zeros(max_size, dtype=np.float)
        self.value_buffer = np.zeros(max_size, dtype=np.float)
        self.logprob_buffer = np.zeros(max_size, dtype=np.float)
        self.return_buffer = np.zeros(max_size, dtype=np.float)
        self.advantage_buffer = np.zeros(max_size, dtype=np.float)
        
        self.gamma, self.lam = gamma, lam
        self.ptr, self.episode_start_idx, self.max_size = 0, 0, max_size
    
    def store(self, obs, act, reward, value, logprob):
        assert self.ptr < self.max_size
        self.obs_buffer[self.ptr] = obs
        self.act_buffer[self.ptr] = act
        self.reward_buffer[self.ptr] = reward
        self.value_buffer[self.ptr] = value
        self.logprob_buffer[self.ptr] = logprob
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        call this at the end of an episode or epoch (ie. cutting off an episode)
        to calculate GAE-Lambda for the duration of episode, as well as r2g for each
        state of episode to use as targets for value function learning

        last_val is 0 if this is called at the end of episode. if episode is cut off 
        then it is boostrapped with V(s_t), value of last state before cutoff. 
        """
        # rewards/values for finished/cutoff episode
        episode_slice = slice(self.episode_start_idx, self.ptr)
        rewards = np.append(self.reward_buffer[episode_slice], last_val)
        values = np.append(self.value_buffer[episode_slice], last_val)

        # GAE-Lambda
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.adv_buf[episode_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # r2g
        self.return_buffer[episode_slice] = core.discount_cumsum(rewards, self.gamma)[:-1]

        self.episode_start_idx = self.ptr
    
    def get(self):
        """
        call this at the end of epoch to get all data from buffer, normalized advantage,
        and reset pointers
        """
        assert self.ptr == self.max_size, "Buffer not full"
        self.ptr, self.episode_start_idx = 0, 0

        #advtantage normalization
        adv_mean, adv_std = mpi_statistics_scalar(self.advantage_buffer)
        self.advantage_buffer = (self.advantage_buffer - adv_mean) / adv_std

        data = {obs: self.obs_buffer,
                act: self.act_buffer,
                ret: self.return_buffer,
                adv: self.advantage_buffer,
                logprob: self.logprob_buffer}
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}



