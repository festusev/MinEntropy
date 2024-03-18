import numpy as np
from typing import Any, Dict, Optional, List

from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
from ray.rllib.utils.typing import SampleBatchType
from ray.rllib.utils.annotations import override

class ActionReplayBuffer:
    def __init__(self, num_actions: int):
        self.replayBufferMap = [SimpleBuffer() for _ in range(num_actions)]
        self.numActions = num_actions

    def add(self, batch: SampleBatchType):
        for action in range(self.numActions):
            self.addActionFromBatch(batch, action)

    def addActionFromBatch(self, batch: SampleBatch, action: int):
        action_buffer = self.replayBufferMap[action]
        matching_indices = np.argwhere(batch["actions"] == action)[:, 0]

        matching_obs = batch["obs"][matching_indices]
        matching_actions = batch["actions"][matching_indices]
        matching_new_obs = batch["new_obs"][matching_indices]

        action_buffer.update(matching_obs, matching_actions, matching_new_obs)

    def sample(self, num_items: int, action: int) -> Optional[SampleBatchType]:
        action_buffer = self.replayBufferMap[action]
        samples = action_buffer.sample(num_items)

        return samples

class SimpleBuffer:
    def __init__(self, obs=None, actions=None, new_obs=None, max_size=10000):
        if obs is None:
            self.initialized = False
        else:
            self.initialized = True
            self.obs = obs
            self.actions = actions.astype(int)
            self.new_obs = new_obs

        self.max_size = max_size

    def initialize(self, obs_shape):
        self.obs = np.zeros((0, *obs_shape))
        self.actions = np.zeros((0))
        self.new_obs = np.zeros((0, *obs_shape))

        self.initialized = True

    def add(self, batch: SampleBatchType):
        self.update(batch["obs"], batch["actions"], batch["new_obs"])

    def update(self, obs: np.ndarray, actions: np.ndarray, new_obs: np.ndarray):
        if not self.initialized:
            self.initialize(obs.shape[1:])

        self.obs = np.concatenate([self.obs, obs], axis=0)
        self.actions = np.concatenate([self.actions, actions], axis=0)
        self.new_obs = np.concatenate([self.new_obs, new_obs], axis=0)

        if self.obs.shape[0] > self.max_size:
            self.evict()

    def evict(self):
        if self.max_size == -1:
            return

        num_over = self.obs.shape[0] - self.max_size

        self.obs = self.obs[num_over:]
        self.actions = self.actions[num_over:]
        self.new_obs = self.new_obs[num_over:]

    def sample(self, num_items: int):
        idxes = [np.random.randint(0, len(self)) for _ in range(num_items)]

        sampled_obs = self.obs[idxes]
        sampled_actions = self.actions[idxes]
        sampled_new_obs = self.new_obs[idxes]

        samples = SimpleBuffer(sampled_obs, sampled_actions, sampled_new_obs)
        return samples

    def __len__(self):
        if self.max_size == -1:
            return self.obs.shape[0]

        return min(self.obs.shape[0], self.max_size)
