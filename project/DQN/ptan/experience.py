import gym
import torch
import random
import collections
from torch.autograd import Variable
import time
import numpy as np

from collections import namedtuple, deque
import time
from .agent import BaseAgent
from .common import utils

# one single experience step
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'done'])


class ExperienceSource:
    """
    Simple n-step experience source using single or multiple environments

    Every experience contains n list of Experience entries
    """
    def __init__(self, env, agent, steps_count=2, steps_delta=1, vectorized=False):
        """
        Create simple experience source
        :param env: environment or list of environments to be used
        :param agent: callable to convert batch of states into actions to take
        :param steps_count: count of steps to track for every experience chain
        :param steps_delta: how many steps to do between experience items
        :param vectorized: support of vectorized envs from OpenAI universe
        """
        assert isinstance(env, (gym.Env, list, tuple))
        assert isinstance(agent, BaseAgent)
        assert isinstance(steps_count, int)
        assert steps_count >= 1
        assert isinstance(vectorized, bool)
        
        if isinstance(env, (list, tuple)):
            self.pool = env
        else:
            self.pool = [env]
        
        self.agent = agent # <DQN.ptan.agent.DQNAgent object at 0x0000016C63B05D60>
        self.steps_count = steps_count # 3
        self.steps_delta = steps_delta # 1
        self.total_rewards = []
        self.total_steps = []
        self.vectorized = vectorized # False

    
    def __iter__(self):
        print("ExperienceSource iter 測試進入次數:",)
        states, agent_states, histories, cur_rewards, cur_steps = [], [], [], [], []
        env_lens = []
        
        for env in self.pool:
            obs = env.reset()
            # if the environment is vectorized, all it's output is lists of results.
            # Details are here: https://github.com/openai/universe/blob/master/doc/env_semantics.rst
            if self.vectorized:
                obs_len = len(obs)
                states.extend(obs)
            else:
                obs_len = 1
                states.append(obs)
            env_lens.append(obs_len)

            for _ in range(obs_len):
                histories.append(deque(maxlen=self.steps_count)) 
                cur_rewards.append(0.0)
                cur_steps.append(0)
                agent_states.append(self.agent.initial_state()) 

            
            # states # 隨機狀態
            # agent_states # [None]
            # histories # [deque([], maxlen=3)]
            # cur_rewards # [0.0]
            # cur_steps # [0]
            
            
        iter_idx = 0
        while True:
            actions = [None] * len(states) # [None]            
            states_input = []
            states_indices = []
            for idx, state in enumerate(states):
                if state is None:
                    actions[idx] = self.pool[0].action_space.sample()  # assume that all envs are from the same family
                else:
                    states_input.append(state) # 狀態
                    states_indices.append(idx) # 索引
            
            if states_input:
                # 會吐出動作和新狀態[2] [None] # 不過原作者這邊好似沒有使用到agent_states
                states_actions, new_agent_states = self.agent(states_input, agent_states)

                for idx, action in enumerate(states_actions):                    
                    g_idx = states_indices[idx]
                    actions[g_idx] = action
                    agent_states[g_idx] = new_agent_states[idx]
            
            # [[2]]
            grouped_actions = _group_list(actions, env_lens)

            global_ofs = 0
            for env_idx, (env, action_n) in enumerate(zip(self.pool, grouped_actions)):
                # 0 (<TimeLimit<StocksEnv instance>>, [2])                
                if self.vectorized:
                    next_state_n, r_n, is_done_n, _ = env.step(action_n)
                else:
                    next_state, r, is_done, _ = env.step(action_n[0])
                    next_state_n, r_n, is_done_n = [next_state], [r], [is_done]

                for ofs, (action, next_state, r, is_done) in enumerate(zip(action_n, next_state_n, r_n, is_done_n)):                
                    idx = global_ofs + ofs
                    state = states[idx]
                    history = histories[idx]
                    cur_rewards[idx] += r
                    cur_steps[idx] += 1
                    if state is not None:
                        history.append(Experience(state=state, action=action, reward=r, done=is_done))
                    if len(history) == self.steps_count and iter_idx % self.steps_delta == 0:
                        yield tuple(history)
                    states[idx] = next_state
                    # 如果按照這樣來看,如果累積多次的話,會跟原本的reset on close 不一樣
                    if is_done:
                        # generate tail of history
                        while len(history) >= 1:
                            yield tuple(history)
                            history.popleft()
                        self.total_rewards.append(cur_rewards[idx])
                        self.total_steps.append(cur_steps[idx])
                        cur_rewards[idx] = 0.0
                        cur_steps[idx] = 0
                        # vectorized envs are reset automatically
                        states[idx] = env.reset() if not self.vectorized else None
                        agent_states[idx] = self.agent.initial_state()
                        history.clear()
                global_ofs += len(action_n)
            iter_idx += 1

    def pop_total_rewards(self):
        r = self.total_rewards
        if r:
            self.total_rewards = []
            self.total_steps = []
        return r

    def pop_rewards_steps(self):
        res = list(zip(self.total_rewards, self.total_steps))
        if res:
            self.total_rewards, self.total_steps = [], []
        return res


def _group_list(items, lens):
    """
    Unflat the list of items by lens
    :param items: list of items
    :param lens: list of integers
    :return: list of list of items grouped by lengths
    """
    res = []
    cur_ofs = 0
    for g_len in lens:
        res.append(items[cur_ofs:cur_ofs+g_len])
        cur_ofs += g_len
    return res


# those entries are emitted from ExperienceSourceFirstLast. Reward is discounted over the trajectory piece
ExperienceFirstLast = collections.namedtuple('ExperienceFirstLast', ('state', 'action', 'reward', 'last_state'))


class ExperienceSourceFirstLast(ExperienceSource):
    """
    This is a wrapper around ExperienceSource to prevent storing full trajectory in replay buffer when we need
    only first and last states. For every trajectory piece it calculates discounted reward and emits only first
    and last states and action taken in the first state.

    If we have partial trajectory at the end of episode, last_state will be None
    這是一個圍繞ExperienceSource的包裝器（wrapper），
    用於在我們只需要初始和最終狀態時，防止在重播緩衝區（replay buffer）
    中儲存完整的軌跡。對於每一個軌跡片段，它會計算折扣獎勵，並且只輸出第一個和最後一個狀態，以及在初始狀態中採取的行動。

    如果在劇集結束時我們有部分軌跡，那麼last_state將為None。
    
    """
    def __init__(self, env, agent, gamma, steps_count=1, steps_delta=1, vectorized=False):
        assert isinstance(gamma, float)
        super(ExperienceSourceFirstLast, self).__init__(env, agent, steps_count+1, steps_delta, vectorized=vectorized)
        self.gamma = gamma
        self.steps = steps_count

    def __iter__(self):
        for exp in super(ExperienceSourceFirstLast, self).__iter__():            
            if exp[-1].done and len(exp) <= self.steps:
                last_state = None
                elems = exp
            else:
                last_state = exp[-1].state
                elems = exp[:-1]
            total_reward = 0.0
            for e in reversed(elems):
                total_reward *= self.gamma
                total_reward += e.reward
            yield ExperienceFirstLast(state=exp[0].state, action=exp[0].action,
                                      reward=total_reward, last_state=last_state)


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done)
        discounted.append(r)
    return discounted[::-1]


class ExperienceSourceRollouts:
    """
    N-step rollout experience source following A3C rollouts scheme. Have to be used with agent,
    keeping the value in its state (for example, agent.ActorCriticAgent).

    Yields batches of num_envs * n_steps samples with the following arrays:
    1. observations
    2. actions
    3. discounted rewards, with values approximation
    4. values
    """
    def __init__(self, env, agent, gamma, steps_count=5):
        """
        Constructs the rollout experience source
        :param env: environment or list of environments to be used
        :param agent: callable to convert batch of states into actions
        :param steps_count: how many steps to perform rollouts
        """
        assert isinstance(env, (gym.Env, list, tuple))
        assert isinstance(agent, BaseAgent)
        assert isinstance(gamma, float)
        assert isinstance(steps_count, int)
        assert steps_count >= 1

        if isinstance(env, (list, tuple)):
            self.pool = env
        else:
            self.pool = [env]
        self.agent = agent
        self.gamma = gamma
        self.steps_count = steps_count
        self.total_rewards = []
        self.total_steps = []

    def __iter__(self):
        pool_size = len(self.pool)
        states = [np.array(e.reset()) for e in self.pool]
        mb_states = np.zeros((pool_size, self.steps_count) + states[0].shape, dtype=states[0].dtype)
        mb_rewards = np.zeros((pool_size, self.steps_count), dtype=np.float32)
        mb_values = np.zeros((pool_size, self.steps_count), dtype=np.float32)
        mb_actions = np.zeros((pool_size, self.steps_count), dtype=np.int64)
        mb_dones = np.zeros((pool_size, self.steps_count), dtype=np.bool)
        total_rewards = [0.0] * pool_size
        total_steps = [0] * pool_size
        agent_states = None
        step_idx = 0

        while True:
            actions, agent_states = self.agent(states, agent_states)
            rewards = []
            dones = []
            new_states = []
            for env_idx, (e, action) in enumerate(zip(self.pool, actions)):
                o, r, done, _ = e.step(action)
                total_rewards[env_idx] += r
                total_steps[env_idx] += 1
                if done:
                    o = e.reset()
                    self.total_rewards.append(total_rewards[env_idx])
                    self.total_steps.append(total_steps[env_idx])
                    total_rewards[env_idx] = 0.0
                    total_steps[env_idx] = 0
                new_states.append(np.array(o))
                dones.append(done)
                rewards.append(r)
            # we need an extra step to get values approximation for rollouts
            if step_idx == self.steps_count:
                # calculate rollout rewards
                for env_idx, (env_rewards, env_dones, last_value) in enumerate(zip(mb_rewards, mb_dones, agent_states)):
                    env_rewards = env_rewards.tolist()
                    env_dones = env_dones.tolist()
                    if not env_dones[-1]:
                        env_rewards = discount_with_dones(env_rewards + [last_value], env_dones + [False], self.gamma)[:-1]
                    else:
                        env_rewards = discount_with_dones(env_rewards, env_dones, self.gamma)
                    mb_rewards[env_idx] = env_rewards
                yield mb_states.reshape((-1,) + mb_states.shape[2:]), mb_rewards.flatten(), mb_actions.flatten(), mb_values.flatten()
                step_idx = 0
            mb_states[:, step_idx] = states
            mb_rewards[:, step_idx] = rewards
            mb_values[:, step_idx] = agent_states
            mb_actions[:, step_idx] = actions
            mb_dones[:, step_idx] = dones
            step_idx += 1
            states = new_states

    def pop_total_rewards(self):
        r = self.total_rewards
        if r:
            self.total_rewards = []
            self.total_steps = []
        return r

    def pop_rewards_steps(self):
        res = list(zip(self.total_rewards, self.total_steps))
        if res:
            self.total_rewards, self.total_steps = [], []
        return res


class ExperienceSourceBuffer:
    """
    The same as ExperienceSource, but takes episodes from the buffer
    """
    def __init__(self, buffer, steps_count=1):
        """
        Create buffered experience source
        :param buffer: list of episodes, each is a list of Experience object
        :param steps_count: count of steps in every entry
        """
        self.update_buffer(buffer)
        self.steps_count = steps_count

    def update_buffer(self, buffer):
        self.buffer = buffer
        self.lens = list(map(len, buffer))

    def __iter__(self):
        """
        Infinitely sample episode from the buffer and then sample item offset
        """
        while True:
            episode = random.randrange(len(self.buffer))
            ofs = random.randrange(self.lens[episode] - self.steps_count - 1)
            yield self.buffer[episode][ofs:ofs+self.steps_count]


class ExperienceReplayBuffer:
    def __init__(self, experience_source, buffer_size):
        assert isinstance(experience_source, (ExperienceSource, type(None)))
        assert isinstance(buffer_size, int)
        self.experience_source_iter = None if experience_source is None else iter(experience_source)
        self.buffer = []
        self.capacity = buffer_size
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def sample(self, batch_size):
        """
        Get one random batch from experience replay
        TODO: implement(實施) sampling order policy
        :param batch_size:
        :return:
        """
        if len(self.buffer) <= batch_size:
            return self.buffer
        # Warning: replace=False makes random.choice O(n)
        keys = np.random.choice(len(self.buffer), batch_size, replace=True)
        return [self.buffer[key] for key in keys]

    def _add(self, sample):
        """
        
            將跌代的資料帶入
            萬一超過就覆寫
        Args:
            sample (_type_): _description_
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.pos] = sample
            self.pos = (self.pos + 1) % self.capacity

    def populate(self, samples):
        """
        將樣本填入緩衝區中
        Populates samples into the buffer
        :param samples: how many samples to populate
        
        <class 'ptan.experience.ExperienceFirstLast'>
        entry: ExperienceFirstLast(state=array([ 0.00773994, -0.01083591,  0.00773994,  0.00456621, -0.01065449,
        0.00456621,  0.00607903, -0.00455927,  0.00455927,  0.        ,
       -0.01783061, -0.00148588,  0.00437956, -0.01021898, -0.00291971,
        0.00442478, -0.02359882, -0.02359882,  0.01226994, -0.00153374,
        0.00306748,  0.01076923, -0.00615385,  0.00153846,  0.00310559,
       -0.01086957, -0.00465839,  0.02503912, -0.00312989,  0.02190923,
        0.        ,  0.        ], dtype=float32), action=1, reward=-2.7099031710120034, last_state=array([ 0.00607903, -0.00455927,  0.00455927,  0.        , -0.01783061,
       -0.00148588,  0.00437956, -0.01021898, -0.00291971,  0.00442478,
       -0.02359882, -0.02359882,  0.01226994, -0.00153374,  0.00306748,
        0.01076923, -0.00615385,  0.00153846,  0.00310559, -0.01086957,
       -0.00465839,  0.02503912, -0.00312989,  0.02190923,  0.00311042,
       -0.00777605, -0.00311042,  0.00944882,  0.        ,  0.0015748 ,
        1.        , -0.02603369], dtype=float32))
        """
        for _ in range(samples):
            entry = next(self.experience_source_iter)
            self._add(entry)


class PrioReplayBufferNaive:
    def __init__(self, exp_source, buf_size, prob_alpha=0.6):
        self.exp_source_iter = iter(exp_source)
        self.prob_alpha = prob_alpha
        self.capacity = buf_size
        self.pos = 0
        self.buffer = []
        self.priorities = np.zeros((buf_size, ), dtype=np.float32)

    def __len__(self):
        return len(self.buffer)

    def populate(self, count):
        max_prio = self.priorities.max() if self.buffer else 1.0
        for _ in range(count):
            sample = next(self.exp_source_iter)
            if len(self.buffer) < self.capacity:
                self.buffer.append(sample)
            else:
                self.buffer[self.pos] = sample
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = np.array(prios, dtype=np.float32) ** self.prob_alpha

        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=True)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio


class PrioritizedReplayBuffer(ExperienceReplayBuffer):
    def __init__(self, experience_source, buffer_size, alpha):
        super(PrioritizedReplayBuffer, self).__init__(experience_source, buffer_size)
        assert alpha > 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2

        self._it_sum = utils.SumSegmentTree(it_capacity)
        self._it_min = utils.MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def _add(self, *args, **kwargs):
        idx = self.pos
        super()._add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights, dtype=np.float32)
        samples = [self.buffer[idx] for idx in idxes]
        return samples, idxes, weights

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


class BatchPreprocessor:
    """
    Abstract preprocessor class descendants to which converts experience
    batch to form suitable to learning.
    """
    def preprocess(self, batch):
        raise NotImplementedError

