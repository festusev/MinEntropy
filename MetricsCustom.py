from ray.rllib.utils.typing import SampleBatchType
from ray.util.iter import LocalIterator
from typing import Dict
from abc import ABC, abstractmethod

class MetricsABC(ABC):
    @abstractmethod
    def update_results(self, sample_batch: SampleBatchType, result: Dict) -> Dict:
        raise NotImplementedError()

class MetricsCallback:
    def __init__(self, metrics_computer: MetricsABC):
        self.rollouts = []
        self.metrics_computer = metrics_computer

    def add_rollout(self, sample_batch: SampleBatchType) -> SampleBatchType:
        self.rollouts.append(sample_batch.policy_batches)
        return sample_batch

    def clear_rollouts(self, sample_batch: SampleBatchType) -> SampleBatchType:
        self.rollouts = []
        return sample_batch

    def __call__(self, result: Dict) -> Dict:
        for sample_batch in self.rollouts:
            self.metrics_computer.update_results(sample_batch, result)
        return result

