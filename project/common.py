
from collections import deque
import numpy as np
from torch.utils.data.dataset import IterableDataset
from tqdm import tqdm
from torch.optim import AdamW
from model import GradientPolicy
import copy
from torch.utils.data import DataLoader
import torch
from AppSettings import AppSettings
import time

class PVM:
    def __init__(self, capacity):
        """Initializes portfolio vector memory.

        Args:
          capacity: Max capacity of memory.
        """
        # initially, memory will have the same actions
        self.capacity = capacity

        self.setting = AppSettings.get_Train_config()
        self.reset()

    def reset(self):
        self.memory = [np.array(
            [1] + [0] * self.setting['PORTFOLIO_SIZE'], dtype=np.float32)] * (self.capacity + 1)
        self.index = 0  # initial index to retrieve data

    def retrieve(self):
        last_action = self.memory[self.index]
        self.index = 0 if self.index == self.capacity else self.index + 1
        return last_action

    def add(self, action):
        self.memory[self.index] = action


class ReplayBuffer:
    def __init__(self, capacity):
        """Initializes replay buffer.

        Args:
          capacity: Max capacity of buffer.
        """
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        """Represents the size of the buffer

        Returns:
          Size of the buffer.
        """
        return len(self.buffer)

    def append(self, experience):
        """Append experience to buffer. When buffer is full, it pops
           an old experience.

        Args:
          experience: experience to be saved.
        """
        self.buffer.append(experience)

    def sample(self):
        """Sample from replay buffer. All data from replay buffer is
        returned and the buffer is cleared.

        Returns:
          Sample of batch_size size.
        """
        buffer = list(self.buffer)
        self.buffer.clear()
        return buffer


class RLDataset(IterableDataset):
    def __init__(self, buffer):
        """Initializes reinforcement learning dataset.

        Args:
            buffer: replay buffer to become iterable dataset.

        Note:
            It's a subclass of pytorch's IterableDataset,
            check https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
        """
        self.buffer = buffer

    def __iter__(self):
        """Iterates over RLDataset.

        Returns:
          Every experience of a sample from replay buffer.
        """
        for experience in self.buffer.sample():
            yield experience


def polyak_average(net, target_net, tau=0.01):
    """Applies polyak average to incrementally update target net.

    Args:
      net: trained neural network.
      target_net: target neural network.
      tau: update rate.
    """
    for qp, tp in zip(net.parameters(), target_net.parameters()):
        tp.data.copy_(tau * qp.data + (1 - tau) * tp.data)


class PG:
    def __init__(self,
                 env,
                 batch_size=100,
                 lr=1e-3,
                 optimizer=AdamW,
                 tau=0.05):
        """Initializes Policy Gradient for portfolio optimization.

          Args:
            env: environment.
            batch_size: batch size to train neural network.
            lr: policy neural network learning rate.
            optim: Optimizer of neural network.
            tau: update rate in Polyak averaging.
        """
        # environment
        self.env = env

        # neural networks
        self.policy = GradientPolicy()
        self.policy.to(self.policy.device)
        self.target_policy = copy.deepcopy(self.policy)
        self.optimizer = optimizer(self.policy.parameters(), lr=lr)
        self.tau = tau

        # replay buffer and portfolio vector memory
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(capacity=batch_size)
        self.pvm = PVM(self.env.episode_length)

        # dataset and dataloader
        dataset = RLDataset(self.buffer)
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True)

    def train(self, episodes=100):
        """Training sequence

        Args:
            episodes: Number of episodes to simulate
        """
        for i in tqdm(range(1, episodes + 1)):
            obs = self.env.reset()  # observation
            self.pvm.reset()  # reset portfolio vector memory
            done = False

            while not done:
                # define last_action and action and update portfolio vector memory
                last_action = self.pvm.retrieve()
                obs_batch = np.expand_dims(obs, axis=0)# (1, 3, 4, 50) # batch_szie,futrue_size,category_of_market,window
                last_action_batch = np.expand_dims(last_action, axis=0)
                action = self.policy(obs_batch, last_action_batch)# 這裡的last_action_batch 是 wt-1
                self.pvm.add(action)

                # run simulation step
                next_obs, reward, done, info = self.env.step(action)# 這裡的last_action_batch 是 wt

                # add experience to replay buffer
                exp = (obs, last_action,
                       info["price_variation"], info["trf_mu"])

                self.buffer.append(exp)

                # update policy networks
                if len(self.buffer) == self.batch_size:
                    self._gradient_ascent()

                obs = next_obs

            # gradient ascent with episode remaining buffer data
            self._gradient_ascent()

    def _gradient_ascent(self):
        # update target neural network
        polyak_average(self.policy, self.target_policy, tau=self.tau)

        # get batch data from dataloader
        obs, last_actions, price_variations, trf_mu = next(
            iter(self.dataloader))
        obs = obs.to(self.policy.device)
        last_actions = last_actions.to(self.policy.device)
        price_variations = price_variations.to(self.policy.device)
        trf_mu = trf_mu.unsqueeze(1).to(self.policy.device)

        # define policy loss (negative for gradient ascent)
        mu = self.policy.mu(obs, last_actions)
        policy_loss = - \
            torch.mean(
                torch.log(torch.sum(mu * price_variations * trf_mu, dim=1)))

        # update policy network
        self.policy.zero_grad()
        policy_loss.backward()
        self.optimizer.step()