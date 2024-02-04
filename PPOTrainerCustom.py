import logging
from typing import Type, List

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents import with_common_config
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.agents.trainer import Trainer
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.rollout_ops import ParallelRollouts, ConcatBatches, \
    StandardizeFields, SelectExperiences
from ray.rllib.execution.train_ops import TrainOneStep, MultiGPUTrainOneStep
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import DEPRECATED_VALUE
from ray.rllib.utils.metrics.learner_info import LEARNER_INFO, \
    LEARNER_STATS_KEY
from ray.rllib.utils.typing import TrainerConfigDict, SampleBatchType
from ray.util.iter import LocalIterator
from ray.rllib.policy.sample_batch import MultiAgentBatch

## DO NOT MODIFY
class UpdateKL:
    """Callback to update the KL based on optimization info.

    This is used inside the execution_plan function. The Policy must define
    a `update_kl` method for this to work. This is achieved for PPO via a
    Policy mixin class (which adds the `update_kl` method),
    defined in ppo_[tf|torch]_policy.py.
    """

    def __init__(self, workers):
        self.workers = workers

    def __call__(self, fetches):
        def update(pi, pi_id):
            assert LEARNER_STATS_KEY not in fetches, \
                ("{} should be nested under policy id key".format(
                    LEARNER_STATS_KEY), fetches)
            if pi_id in fetches:
                kl = fetches[pi_id][LEARNER_STATS_KEY].get("kl")
                assert kl is not None, (fetches, pi_id)
                # Make the actual `Policy.update_kl()` call.
                pi.update_kl(kl)
            else:
                logger.warning("No data for {}, not updating kl".format(pi_id))

        # Update KL on all trainable policies within the local (trainer)
        # Worker.
        self.workers.local_worker().foreach_trainable_policy(update)

## DO NOT MODIFY
def warn_about_bad_reward_scales(config, result):
    if result["policy_reward_mean"]:
        return result  # Punt on handling multiagent case.

    # Warn about excessively high VF loss.
    learner_info = result["info"][LEARNER_INFO]
    if DEFAULT_POLICY_ID in learner_info:
        scaled_vf_loss = config["vf_loss_coeff"] * \
            learner_info[DEFAULT_POLICY_ID][LEARNER_STATS_KEY]["vf_loss"]

        policy_loss = learner_info[DEFAULT_POLICY_ID][LEARNER_STATS_KEY][
            "policy_loss"]
        if config.get("model", {}).get("vf_share_layers") and \
                scaled_vf_loss > 100:
            logger.warning(
                "The magnitude of your value function loss is extremely large "
                "({}) compared to the policy loss ({}). This can prevent the "
                "policy from learning. Consider scaling down the VF loss by "
                "reducing vf_loss_coeff, or disabling vf_share_layers.".format(
                    scaled_vf_loss, policy_loss))

    # Warn about bad clipping configs
    if config["vf_clip_param"] <= 0:
        rew_scale = float("inf")
    else:
        rew_scale = round(
            abs(result["episode_reward_mean"]) / config["vf_clip_param"], 0)
    if rew_scale > 200:
        logger.warning(
            "The magnitude of your environment rewards are more than "
            "{}x the scale of `vf_clip_param`. ".format(rew_scale) +
            "This means that it will take more than "
            "{} iterations for your value ".format(rew_scale) +
            "function to converge. If this is not intended, consider "
            "increasing `vf_clip_param`.")

    return result


class TrainCustomOneStep:
    def __init__(self, train, workers: WorkerSet = None, callback=None):
        self.train = train
        self.workers = workers
        self.callback = callback

    def __call__(self, batch: SampleBatchType) -> SampleBatchType:
        sample_batches = []
        # if isinstance(batch, MultiAgentBatch):
        #     for b in batch.policy_batches.values():
        #         sample_batches.append(b)
        # else:
        #     sample_batches.append(batch)

        self.train(batch.policy_batches)

        if self.callback is not None:
            self.workers.foreach_env(self.callback(self.train))

        return batch

class Hook:
    def __init__(self, trainer):
        self.trainer = trainer

    def __call__(self, batch: SampleBatchType) -> SampleBatchType:
        import pdb; pdb.set_trace()
        return self.trainer(batch)

class PPOTrainerCustom(PPOTrainer):
    @classmethod
    @override(PPOTrainer)
    def get_default_config(cls) -> TrainerConfigDict:
        default_config = super().get_default_config()
        default_config["execution_plan"] = {"train_extras": []}
        return default_config

    @override(PPOTrainer)
    def _kwargs_for_execution_plan(self):
        kwargs = super()._kwargs_for_execution_plan()
        kwargs.update(self.config["execution_plan"])
        return kwargs

    @staticmethod
    @override(Trainer)
    def execution_plan(workers: WorkerSet, config: TrainerConfigDict,
                       **kwargs) -> LocalIterator[dict]:
        rollouts = ParallelRollouts(workers, mode="bulk_sync")

        # Collect batches for the trainable policies.
        rollouts = rollouts.for_each(
            SelectExperiences(workers.trainable_policies()))
        # Concatenate the SampleBatches into one.
        rollouts = rollouts.combine(
            ConcatBatches(
                min_batch_size=config["train_batch_size"],
                count_steps_by=config["multiagent"]["count_steps_by"],
            ))

        # To add functionality, allow customisable functions to read the batch before training.
        # These must return a Batch datatype
        for extra in kwargs["train_extras"]:
            rollouts = rollouts.for_each(
                TrainCustomOneStep(workers=workers, **extra))

        # Standardize advantages.
        rollouts = rollouts.for_each(StandardizeFields(["advantages"]))

        # Perform one training step on the combined + standardized batch.
        if config["simple_optimizer"]:
            train_op = rollouts.for_each(
                TrainOneStep(
                    workers,
                    num_sgd_iter=config["num_sgd_iter"],
                    sgd_minibatch_size=config["sgd_minibatch_size"]))
        else:
            train_op = rollouts.for_each(
                MultiGPUTrainOneStep(
                    workers=workers,
                    sgd_minibatch_size=config["sgd_minibatch_size"],
                    num_sgd_iter=config["num_sgd_iter"],
                    num_gpus=config["num_gpus"],
                    _fake_gpus=config["_fake_gpus"]))

        # Update KL after each round of training.
        train_op = train_op.for_each(lambda t: t[1]).for_each(
            UpdateKL(workers))

        # Warn about bad reward scales and return training metrics.
        return StandardMetricsReporting(train_op, workers, config) \
            .for_each(lambda result: warn_about_bad_reward_scales(
            config, result))
