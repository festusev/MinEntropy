import argparse
import os
from typing import Callable, List, cast

import ray
import torch
from bpd.envs.overcooked import (
    OvercookedCallbacks,
    OvercookedMultiAgent,
    get_littered_start_state_fn,
)
from bpd.training_utils import (
    build_logger_creator,
    load_policies_from_checkpoint,
    load_trainer_config,
)
from gym import spaces
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.trainer import COMMON_CONFIG
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.typing import MultiAgentPolicyConfigDict, TrainerConfigDict
from typing_extensions import Literal

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layout_name",
        type=str,
        required=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_training_iters", type=int, default=200)
    parser.add_argument("--log_dir", type=str, default="data/logs")
    parser.add_argument("--experiment_tag", type=str, required=False)
    parser.add_argument("--human_model_checkpoint", type=str, required=False)
    parser.add_argument("--num_littered_objects", type=int, default=0)
    parser.add_argument("--no_smirl", action="store_true")
    args = parser.parse_args()

    # Environment
    layout_name = args.layout_name
    rew_shaping_params = {
        "PLACEMENT_IN_POT_REW": 3,
        "DISH_PICKUP_REWARD": 3,
        "SOUP_PICKUP_REWARD": 5,
        "DISH_DISP_DISTANCE_REW": 0,
        "POT_DISTANCE_REW": 0,
        "SOUP_DISTANCE_REW": 0,
    }
    dispense_reward = 0
    horizon = 400
    no_regular_reward = False
    action_rewards = [0] * Action.NUM_ACTIONS
    num_littered_objects = args.num_littered_objects
    start_state_fn = get_littered_start_state_fn(
        num_littered_objects, OvercookedGridworld.from_layout_name(layout_name)
    )

    # Training
    num_workers = min(5, os.cpu_count() or 1)
    seed = args.seed
    num_gpus = 1 if torch.cuda.is_available() else 0
    num_gpus_per_worker = 0
    train_batch_size = 10000
    sgd_minibatch_size = 2000
    rollout_fragment_length = horizon
    num_training_iters = args.num_training_iters
    lr = 1e-3
    grad_clip = 0.1
    gamma = 0.99
    gae_lambda = 0.98
    vf_share_layers = False
    vf_loss_coeff = 1e-4
    entropy_coeff_start = 1e-3
    entropy_coeff_end = 1e-3
    entropy_coeff_horizon = num_training_iters * train_batch_size // 4
    kl_coeff = 0.2
    clip_param = 0.05
    num_sgd_iter = 3

    # Model
    num_hidden_layers = 3
    size_hidden_layers = 64
    num_filters = 25
    num_conv_layers = 3
    split_backbone = False
    use_lstm = False
    use_attention = False
    use_sequence_model = False
    lstm_cell_size = 256  # LSTM memory cell size (only used if use_lstm=True)
    max_seq_len = horizon
    custom_model_config = {
        "num_hidden_layers": num_hidden_layers,
        "size_hidden_layers": size_hidden_layers,
        "num_filters": num_filters,
        "num_conv_layers": num_conv_layers,
        "split_backbone": split_backbone,
    }
    custom_model = "overcooked_ppo_model"

    model_config = {
        "custom_model": custom_model,
        "max_seq_len": max_seq_len,
        "custom_model_config": custom_model_config,
        "vf_share_layers": vf_share_layers,
        "use_lstm": use_lstm,
        "lstm_cell_size": lstm_cell_size,
        "use_attention": use_attention,
    }

    # Reward shaping
    use_phi = False  # Whether dense reward should come from potential function or not
    # Constant by which shaped rewards are multiplied by when calculating total reward
    reward_shaping_factor = 1.0
    # Linearly anneal the reward shaping factor such that it reaches zero after this
    # number of timesteps
    reward_shaping_horizon = num_training_iters * train_batch_size // 2
    # Whether the agents should both get all dense rewards.
    share_dense_reward = False

    checkpoint_to_load_policies = args.human_model_checkpoint
    if checkpoint_to_load_policies is not None:
        checkpoint_to_load_policies_config: TrainerConfigDict = load_trainer_config(
            checkpoint_to_load_policies
        )

    # Multiagent
    multiagent_mode: Literal["self_play", "cross_play"] = (
        "self_play" if checkpoint_to_load_policies is None else "cross_play"
    )
    policy_ids: List[str]
    policy_mapping_fn: Callable[..., str]
    if multiagent_mode == "self_play" or checkpoint_to_load_policies is not None:
        policy_ids = ["ppo"]
        policy_mapping_fn = lambda agent_id, *args, **kwargs: "ppo"
    elif multiagent_mode == "cross_play":
        policy_ids = ["ppo_0", "ppo_1"]
        policy_mapping_fn = lambda agent_id, *args, **kwargs: cast(str, agent_id)
    policies_to_train = policy_ids

    # Logging
    save_freq = 25  # noqa: F841
    log_dir = args.log_dir
    experiment_tag = args.experiment_tag
    experiment_name_parts = [multiagent_mode, "PPO", layout_name]
    if experiment_tag is not None:
        experiment_name_parts.append(experiment_tag)
    experiment_name = os.path.join(*experiment_name_parts)  # noqa: F841
    checkpoint_path = None  # noqa: F841

    env_id = "overcooked_multi_agent"
    env_config = {
        # To be passed into OvercookedGridWorld constructor
        "mdp_params": {
            "layout_name": layout_name,
            "rew_shaping_params": rew_shaping_params,
            "smirl": not args.no_smirl
        },
        # To be passed into OvercookedEnv constructor
        "env_params": {
            "horizon": horizon,
            "start_state_fn": start_state_fn,
            "num_mdp": 1,
        },
        # To be passed into OvercookedMultiAgent constructor
        "multi_agent_params": {
            "reward_shaping_factor": reward_shaping_factor,
            "reward_shaping_horizon": reward_shaping_horizon,
            "use_phi": use_phi,
            "share_dense_reward": share_dense_reward,
            "bc_schedule": OvercookedMultiAgent.self_play_bc_schedule,
            "extra_rew_shaping": {
                "onion_dispense": dispense_reward,
                "dish_dispense": dispense_reward,
            },
            "no_regular_reward": no_regular_reward,
            "action_rewards": action_rewards,
        },
    }

    overcooked_env_config = env_config
    env = OvercookedMultiAgent.from_config(overcooked_env_config)

    policies: MultiAgentPolicyConfigDict = {}

    ppo_observation_space = env.ppo_observation_space

    for policy_id in policy_ids:
        policies[policy_id] = PolicySpec(
            None,
            ppo_observation_space,
            env.action_space,
            {"model": model_config},
        )

    if multiagent_mode == "cross_play" and checkpoint_to_load_policies is not None:
        # In the case where we want to train a policy via cross-play with an
        # existing policy from a checkpoint. The loaded policy does not
        # get trained.

        bc_features = False
        if "multiagent" in checkpoint_to_load_policies_config:
            loaded_policy_dict: MultiAgentPolicyConfigDict = (
                checkpoint_to_load_policies_config["multiagent"]["policies"]
            )
            loaded_policy_ids = list(loaded_policy_dict.keys())
            assert len(loaded_policy_ids) == 1
            (loaded_policy_id,) = loaded_policy_ids
            loaded_policy_obs_space: spaces.Box = loaded_policy_dict[loaded_policy_id][
                1
            ]
            bc_features = (
                loaded_policy_obs_space.shape == env.bc_observation_space.shape
            )
        else:
            bc_features = True
            loaded_policy_id = DEFAULT_POLICY_ID
            loaded_policy_dict = {}

        if not bc_features:
            (
                loaded_policy_cls,
                loaded_policy_obs_space,
                loaded_policy_action_space,
                loaded_policy_config,
            ) = loaded_policy_dict[loaded_policy_id]
            policies[loaded_policy_id] = PolicySpec(
                None,
                loaded_policy_obs_space,
                loaded_policy_action_space,
                loaded_policy_config,
            )
            policy_mapping_fn = (
                lambda agent_id, *args, loaded_policy_id=loaded_policy_id, **kwargs: "ppo"
                if agent_id == "ppo_0"
                else loaded_policy_id
            )

            checkpoint_env_config = checkpoint_to_load_policies_config["env_config"]
        else:
            # We're doing cross play with a BC agent.
            assert loaded_policy_id == DEFAULT_POLICY_ID
            policies[DEFAULT_POLICY_ID] = PolicySpec(
                None,
                env.bc_observation_space,
                env.action_space,
                {
                    "model": {
                        **checkpoint_to_load_policies_config.get("model", {}),
                        **loaded_policy_dict.get(
                            DEFAULT_POLICY_ID, (None, None, None, {})
                        )[3].get("model", {}),
                        "vf_share_layers": True,
                    }
                },
            )
            env_config["multi_agent_params"]["bc_schedule"] = [
                (0, 1),
                (float("inf"), 1),
            ]
            policy_mapping_fn = (
                lambda agent_id, *args, **kwargs: DEFAULT_POLICY_ID
                if agent_id.startswith("bc")
                else "ppo"
            )

    config: TrainerConfigDict = {  # noqa: F841
        "env": env_id,
        "env_config": env_config,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train": policies_to_train,
        },
        "callbacks": OvercookedCallbacks,
        "num_workers": num_workers,
        "train_batch_size": train_batch_size,
        "sgd_minibatch_size": sgd_minibatch_size,
        "rollout_fragment_length": rollout_fragment_length,
        "num_sgd_iter": num_sgd_iter,
        "lr": lr,
        "grad_clip": grad_clip,
        "gamma": gamma,
        "lambda": gae_lambda,
        "vf_loss_coeff": vf_loss_coeff,
        "kl_coeff": kl_coeff,
        "clip_param": clip_param,
        "num_gpus": num_gpus,
        "num_gpus_per_worker": num_gpus_per_worker,
        "seed": seed,
        "entropy_coeff_schedule": [
            (0, entropy_coeff_start),
            (entropy_coeff_horizon, entropy_coeff_end),
        ],
        "framework": "torch",
    }

    if "disable_env_checking" in COMMON_CONFIG:
        config["disable_env_checking"] = True

    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
    )

    trainer = PPOTrainer(
        config,
        logger_creator=build_logger_creator(
            log_dir,
            experiment_name,
        ),
    )

    if checkpoint_to_load_policies is not None:
        print(f"Initializing human policy from {checkpoint_to_load_policies}")
        load_policies_from_checkpoint(checkpoint_to_load_policies, trainer)

    if checkpoint_path is not None:
        print(f"Restoring checkpoint at {checkpoint_path}")
        trainer.restore(checkpoint_path)

    result = None
    for _ in range(num_training_iters):
        print(f"Starting training iteration {trainer.iteration}")
        result = trainer.train()

        if trainer.iteration % save_freq == 0:
            checkpoint = trainer.save()
            print(f"Saved checkpoint to {checkpoint}")

    checkpoint = trainer.save()
    print(f"Saved final checkpoint to {checkpoint}")
