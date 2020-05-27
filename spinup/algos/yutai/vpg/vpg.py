import numpy as np
import torch
from torch.optim import Adam
import gym
import time
from typing import Callable, Optional
import spinup.algos.yutai.vpg.core as core
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
		self.advantage_buffer[episode_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

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

		data = dict(obs = self.obs_buffer,
					act = self.act_buffer,
					ret = self.return_buffer,
					adv = self.advantage_buffer,
					logprob = self.logprob_buffer)
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
				 logger_kwargs=dict(),
				 save_freq: int = 10):
		# Special function to avoid certain slowdowns from PyTorch + MPI combo.
		setup_pytorch_for_mpi()
		# Set up logger and save configuration
		self.logger = EpochLogger(**logger_kwargs)
		self.logger.save_config(locals())
		# Random seed
		seed += 10000 * proc_id()
		torch.manual_seed(seed)
		np.random.seed(seed)

		self.epochs = epochs
		self.steps_per_epoch = steps_per_epoch
		self.num_iter_train_critic = num_iter_train_critic
		self.max_episode_len = max_episode_len
		self.save_freq = save_freq

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
		self.actor_optimizer = Adam(self.ac.actor.parameters(), lr=actor_lr)
		self.critic_optimizer = Adam(self.ac.critic.parameters(), lr=critic_lr)


		# Sync params across processes
		sync_params(self.ac)
		# Count variables
		var_counts = tuple(core.count_vars(module) for module in [self.ac.actor, self.ac.critic])
		self.logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)
		# Set up model saving
		self.logger.setup_pytorch_saver(self.ac)

	def compute_actor_loss(self, data):
		obs, act, adv, logprob_old = data['obs'], data['act'], data['adv'], data['logprob']

		# policy loss
		pi, logprob = self.ac.actor(obs, act)
		loss_actor = -(logprob_old * adv).mean()
		
		# extra info
		approx_kl = (logprob_old - logprob).mean().item()
		entropy = pi.entropy().mean().item()
		pi_info = dict(kl=approx_kl, entropy=entropy)

		return loss_actor, pi_info

	def compute_critic_loss(self, data):
		obs, ret = data['obs'], data['ret']
		return ((self.ac.critic(obs) - ret)**2).mean()

	def update(self):
		data = self.buffer.get()

		actor_loss_old, actor_info_old = self.compute_actor_loss(data)
		actor_loss_old = actor_loss_old.item()
		critic_loss_old = self.compute_critic_loss(data).item()

		# train policy
		self.actor_optimizer.zero_grad()
		actor_loss, actor_info = self.compute_actor_loss(data)
		actor_loss.backward()
		mpi_avg_grads(self.ac.actor)
		self.actor_optimizer.step()

		# train critic
		for i in range(self.num_iter_train_critic):
			self.critic_optimizer.zero_grad()
			critic_loss = self.compute_critic_loss(data)
			critic_loss.backward()
			mpi_avg_grads(self.ac.critic)
			self.critic_optimizer.step()

		#log 
		kl, entropy = actor_info['kl'], actor_info['ent']
		self.logger.store(LossPi=actor_loss_old, LossV=critic_loss_old,
						  KL=kl, Entropy=entropy,
						  DeltaLossV=(critic_loss.item() - critic_loss_old),
						  DeltaLossPi=(actor_loss.item() - actor_loss_old))

						  

	def train(self):
		start_time = time.time()
		obs, episode_ret, episode_len = self.env.reset(), 0, 0

		for epoch in range(self.epochs):
			for t in range(self.local_steps_per_epoch):
				act, v, logprob = self.ac.step(torch.as_tensor(obs, dtype=torch.float32))
				# print(f"act: {act}")
				# print(f"v: {v}")
				# print(f"logprob: {logprob}")
				
				obs_next, reward, done, _ = self.env.step(act)
				episode_ret += reward
				episode_len += 1

				self.buffer.store(obs, act, reward, v, logprob)
				self.logger.store(VVals=v)

				obs = obs_next

				# episode end/timeout logic
				timeout = (episode_len == self.max_episode_len)
				terminal = (done or timeout)
				epoch_ended = (t == self.local_steps_per_epoch - 1)

				if terminal or epoch_ended:
					if epoch_ended and not terminal:
						print(f"Warning: trajectory cut off by epoch at {episode_len} steps")
					if timeout or epoch_ended:
						_, v, _ = self.ac.step(torch.as_tensor(obs, dtype=torch.float32))
					else:
						v = 0
					self.buffer.finish_path(v)
					if terminal:
						self.logger.store(EpRet=episode_ret, EpLen=episode_len)
					obs, episode_ret, episode_len = self.env.reset(), 0, 0

			if (epoch % self.save_freq == 0) or (epoch == self.epochs - 1):
				self.logger.save_state({"env": self.env}, None)

			self.update()

			# Log info about epoch
			self.logger.log_tabular('Epoch', epoch)
			self.logger.log_tabular('EpRet', with_min_and_max=True)
			self.logger.log_tabular('EpLen', average_only=True)
			self.logger.log_tabular('VVals', with_min_and_max=True)
			self.logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
			self.logger.log_tabular('LossPi', average_only=True)
			self.logger.log_tabular('LossV', average_only=True)
			self.logger.log_tabular('DeltaLossPi', average_only=True)
			self.logger.log_tabular('DeltaLossV', average_only=True)
			self.logger.log_tabular('Entropy', average_only=True)
			self.logger.log_tabular('KL', average_only=True)
			self.logger.log_tabular('Time', time.time()-start_time)
			self.logger.dump_tabular()

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--env', type=str, default='CartPole-v0')
	parser.add_argument('--hid', type=int, default=64)
	parser.add_argument('--l', type=int, default=2)
	parser.add_argument('--gamma', type=float, default=0.99)
	parser.add_argument('--seed', '-s', type=int, default=0)
	parser.add_argument('--cpu', type=int, default=4)
	parser.add_argument('--steps', type=int, default=4000)
	parser.add_argument('--epochs', type=int, default=50)
	parser.add_argument('--exp_name', type=str, default='vpg')
	args = parser.parse_args()

	mpi_fork(args.cpu)  # run parallel code with mpi

	from spinup.utils.run_utils import setup_logger_kwargs
	logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

	vpg = VPG(lambda : gym.make(args.env),
			  ac_maker=core.MLPActorCritic,
			  ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
			  gamma=args.gamma,
			  seed=args.seed,
			  steps_per_epoch=args.steps,
			  epochs=args.epochs,
			  logger_kwargs=logger_kwargs)

	vpg.train()