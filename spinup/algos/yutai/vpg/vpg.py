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

class VPG:
    """
    VPG w/ GAE-Lambda
    """
    def __init__(self,
                 env_maker: Callable,
                 ac_maker = core.MLPActorCritic,
                 ac_kwargs = {},
                 seed: int = 0,
                 epochs: int = 50,
                 steps_per_epoch: int = 4000,
                 gamma: float = 0.99,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 1e-3,
                 num_iter_train_critic: int = 80,
                 lam: float = 0.97,
                 max_episode_len: int = 1000,
                 logger_kwargs={},
                 save_freq: int = 10)
        # Special function to avoid certain slowdowns from PyTorch + MPI combo.
        setup_pytorch_for_mpi()
        # Set up logger and save configuration
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())
        # Random seed
        seed += 10000 * proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)

        # make env
        self.env = env_maker()
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape

        # make actor-critic
        self.ac = ac_maker(self.env.observation_space, self.env.action_space, **ac_kwargs)

        # make buffer
        self.local_steps_per_epoch = int(steps_per_epoch/num_procs())
        self.buffer = Buffer(self.obs_dim, self.act_dim, self.local_steps_per_epoch, gamma, lam)
        
        # make optimizers
        self.actor_optimizer = Adam(ac.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(ac.critic.parameters(), lr=critic_lr)


        # Sync params across processes
        sync_params(ac)
        # Count variables
        var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.v])
        logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)
        # Set up model saving
        logger.setup_pytorch_saver(ac)

    def compute_actor_loss(self, data):
        obs, act, adv, logprob_old = data['obs'], data['act'], data['adv'], data['logp']

        # policy loss
        
        # extra info

    def compute_critic_loss(self, data):
        obs, ret = data['obs'], data['ret']
        return ((self.ac.critic(obs) - ret)**2).mean()

