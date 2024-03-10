import argparse
import os
from typing import Callable, List, cast, Dict, Tuple, Optional

import numpy as np
import ray
import torch
import attr
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
from overcooked_ai_py.utils import read_layout_dict

# from ray.rllib.agents.ppo import PPOTrainer
from PPOTrainerCustom import PPOTrainerCustom
from ray.rllib.agents.trainer import COMMON_CONFIG
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.typing import MultiAgentPolicyConfigDict, TrainerConfigDict
from typing_extensions import Literal
from Empowerment import Empowerment, ClassifierEmpowerment, TwoHeadedEmpowerment, MIMIEmpowerment, \
    ContrastiveEmpowerment
from PPOTrainerCustom import TrainCustomOneStep
import ray
import wandb

HORIZON = 400
TRAIN_BATCH_SIZE = 10_000
MODEL_TYPE_REGISTRY = {
    "ppo": "overcooked_ppo_model",
    "smirl": "overcooked_smirl_model",
    "smirl_e": "overcooked_smirl_model",
    "contrastive_e": "overcooked_smirl_model"
}
ENV_ID = "overcooked_multi_agent"


@attr.s(auto_attribs=True)
class WandbTrainWrapper:
    empowerment_model: Empowerment

    def __call__(self, sample_batches: Dict):
        self.sample_batches = sample_batches


def get_overcooked_env_config(layout_name: str, num_training_iters, policy_ids: List[str], no_anneal: bool = False,
                              empowerment_model: Empowerment = None, empowerment_weight: bool = 1,
                              compute_smirl: bool = False, yell_action: bool = False) -> Dict:
    start_state_fn = get_littered_start_state_fn(0, OvercookedGridworld.from_layout_name(layout_name))
    # TODO: Make the start_state_fn just a string so it is properly serializable

    rew_shaping_params = {
        "PLACEMENT_IN_POT_REW": 3,
        "DISH_PICKUP_REWARD": 3,
        "SOUP_PICKUP_REWARD": 5,
        "DISH_DISP_DISTANCE_REW": 0,
        "POT_DISTANCE_REW": 0,
        "SOUP_DISTANCE_REW": 0,
    }
    dispense_reward = 0
    no_regular_reward = False
    action_rewards = [0] * Action.NUM_ACTIONS

    # Whether dense reward should come from potential function or not
    use_phi = False

    # Constant by which shaped rewards are multiplied by when calculating total reward
    reward_shaping_factor = 1.0

    # Linearly anneal the reward shaping factor such that it reaches zero after this number of timesteps
    reward_shaping_horizon = num_training_iters * TRAIN_BATCH_SIZE // 2

    if no_anneal:
        reward_shaping_horizon = float('inf')

    # Whether the agents should both get all dense rewards.
    share_dense_reward = False

    env_config = {
        # To be passed into OvercookedGridWorld constructor
        "mdp_params": {
            "layout_name": layout_name,
            "rew_shaping_params": rew_shaping_params,
            "empowerment_model": empowerment_model,
            "smirl": compute_smirl,
            "yell": yell_action
        },
        # To be passed into OvercookedEnv constructor
        "env_params": {
            "horizon": HORIZON,
            "start_state_fn": start_state_fn,
            "num_mdp": 1,
        },
        # To be passed into OvercookedMultiAgent constructor
        "multi_agent_params": {
            "reward_shaping_factor": reward_shaping_factor,
            "reward_shaping_horizon": reward_shaping_horizon,
            "empowerment_weight": empowerment_weight,
            "use_phi": use_phi,
            "share_dense_reward": share_dense_reward,
            "bc_schedule": OvercookedMultiAgent.self_play_bc_schedule,
            "extra_rew_shaping": {
                "onion_dispense": dispense_reward,
                "dish_dispense": dispense_reward,
            },
            "no_regular_reward": no_regular_reward,
            "action_rewards": action_rewards,
            "agents": policy_ids
        },
    }

    return env_config


def get_obs_shape_from_layout(layout_name):
    layout_dict = read_layout_dict(layout_name)
    grid = layout_dict["grid"]
    grid = [layout_row.strip() for layout_row in grid.split("\n")]

    return len(grid[0]), len(grid)


def get_policy_ids(model_0: str, model_1: str = None) -> List[str]:
    policy_ids = [model_0 + "_0"]

    if model_1 is not None:
        if model_0 != model_1:
            policy_ids.append(model_1 + "_0")
        else:
            policy_ids.append(model_1 + "_1")

    return policy_ids


def get_policy_specs(policy_ids: List[str], env: OvercookedMultiAgent,
                     model_configs: List[Dict]) -> MultiAgentPolicyConfigDict:
    policies: MultiAgentPolicyConfigDict = {}

    for i in range(len(policy_ids)):
        policies[policy_ids[i]] = PolicySpec(
            None,
            env.ppo_observation_space,
            env.action_space,
            model_configs[i]
        )

    return policies


def load_policy_configs_from_checkpoint(checkpoint_path: str) -> Tuple[List[str], List[Dict]]:
    checkpoint = load_trainer_config(checkpoint_path)
    loaded_policy_dict: MultiAgentPolicyConfigDict = (checkpoint["multiagent"]["policies"])

    model_configs: List[Dict] = []

    loaded_policy_ids = list(loaded_policy_dict.keys())
    for loaded_policy_id in loaded_policy_ids:
        (
            loaded_policy_cls,
            loaded_policy_obs_space,
            loaded_policy_action_space,
            loaded_policy_config,
        ) = loaded_policy_dict[loaded_policy_id]

        model_configs.append(loaded_policy_config)

    return loaded_policy_ids, model_configs


def get_model_configs(policy_ids: List[str]) -> List[Dict]:
    custom_model_config = {
        "num_hidden_layers": 3,
        "size_hidden_layers": 64,
        "num_filters": 25,
        "num_conv_layers": 3,
        "split_backbone": False
    }

    model_configs = []
    for i in range(len(policy_ids)):
        policy_name: str = policy_ids[i][:-2]
        model_type = MODEL_TYPE_REGISTRY[policy_name]
        model_configs.append({"model": {
            "custom_model": model_type,
            "max_seq_len": HORIZON,
            "custom_model_config": custom_model_config,
            "vf_share_layers": False,
            "use_lstm": False,
            "lstm_cell_size": 256,
            "use_attention": False,
        }})

    return model_configs


def get_empowerment_train_extras(empowerment_model: Empowerment) -> List[Dict]:
    def get_empowerment_setter(empowerment_model):
        def setter(env):
            env.base_env.update_empowerment(empowerment_model)

        return setter

    train_extras = [{"train": empowerment_model, "callback": get_empowerment_setter},
                    {"train": WandbTrainWrapper(empowerment_model)}]

    return train_extras


def get_trainer(multiagent_mode: bool, experiment_name: str, log_dir: str, num_training_iters: int,
                policies_to_train: List[str], overcooked_env_config: Dict,
                policies: MultiAgentPolicyConfigDict, seed: int, train_extras: List[Dict]) -> PPOTrainerCustom:
    if multiagent_mode:
        policy_mapping_fn = lambda agent_id, *args, **kwargs: cast(str, agent_id)
    else:  # We are doing self-play
        policy_mapping_fn = lambda agent_id, *args, **kwargs: policies_to_train[0]

    config: TrainerConfigDict = {  # noqa: F841
        "env": ENV_ID,
        "env_config": overcooked_env_config,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train": policies_to_train,
        },
        "callbacks": OvercookedCallbacks,
        "num_workers": min(5, os.cpu_count() or 1),
        "train_batch_size": TRAIN_BATCH_SIZE,
        "sgd_minibatch_size": 2_000,
        "rollout_fragment_length": HORIZON,
        "num_sgd_iter": 3,
        "lr": 1e-3,
        "grad_clip": 0.1,
        "gamma": 0.99,
        "lambda": 0.98,
        "vf_loss_coeff": 1e-4,
        "kl_coeff": 0.2,
        "clip_param": 0.05,
        "num_gpus": 1 if torch.cuda.is_available() else 0,
        "num_gpus_per_worker": 0,
        "seed": seed,
        "entropy_coeff_schedule": [
            (0, 1e-3),
            (num_training_iters * TRAIN_BATCH_SIZE // 4, 1e-3),
        ],
        "framework": "torch",
        "execution_plan": {
            "train_extras": train_extras
        },
        "simple_optimizer": True
    }

    if "disable_env_checking" in COMMON_CONFIG:
        config["disable_env_checking"] = True

    trainer = PPOTrainerCustom(
        config,
        logger_creator=build_logger_creator(
            log_dir,
            experiment_name,
        )
    )

    return trainer


def get_experiment_name(policy_ids: List[str], multiagent_mode: bool, layout_name: str, experiment_tag: str,
                        no_anneal: bool) -> str:
    if no_anneal:
        experiment_tag += "no_anneal"

    subdir = policy_ids[0][:-2].upper()
    if multiagent_mode:
        subdir += "_" + policy_ids[1][:-2].upper()
    experiment_name_parts = ["cross_play" if multiagent_mode else "self_play", subdir, layout_name]

    experiment_name_parts.append(experiment_tag)

    experiment_name: str = os.path.join(*experiment_name_parts)
    return experiment_name


def setup_experiment(args: argparse.Namespace) -> Tuple[PPOTrainerCustom, Empowerment]:
    # Environment
    layout_name = args.layout_name
    obs_shape = get_obs_shape_from_layout(layout_name)

    # Training
    seed = args.seed
    num_training_iters = args.num_training_iters

    multiagent_mode = args.model_1 is not None

    # If model_0 is not in the MODEL_TYPE_REGISTRY, load from checkpoint
    if args.model_0 not in MODEL_TYPE_REGISTRY:
        policy_ids, model_configs = load_policy_configs_from_checkpoint(args.model_0)
    else:
        policy_ids = get_policy_ids(args.model_0, args.model_1)
        model_configs = get_model_configs(policy_ids)

    device = torch.device("cpu")  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train an agent with empowerment + SMiRL rewards. First, add a callback to the PPO train loop that updates
    # the empowerment networks
    train_extras: List[Dict] = []

    empowerment_model: Optional[Empowerment] = None
    compute_empowerment: bool = any(["smirl_e" in name or "contrastive_e" in name for name in policy_ids])
    if compute_empowerment:
        empowerment_model = ContrastiveEmpowerment(num_actions=Action.NUM_ACTIONS, in_channels=26, obs_shape=obs_shape,
                                                   device=device,
                                                   prob=args.goal_prob)
        train_extras.extend(get_empowerment_train_extras(empowerment_model))

    overcooked_env_config: Dict = get_overcooked_env_config(layout_name, num_training_iters, policy_ids,
                                                            no_anneal=args.no_anneal,
                                                            empowerment_model=empowerment_model,
                                                            empowerment_weight=args.empowerment_weight,
                                                            compute_smirl=False, yell_action=args.yell)
    env = OvercookedMultiAgent.from_config(overcooked_env_config)
    env.agents = policy_ids[:]

    policies = get_policy_specs(policy_ids, env, model_configs)

    if not multiagent_mode:  # We are doing self-play
        policy_ids.append(policy_ids[0][:-2] + "_1")

    policies_to_train = [policy_ids[0]]
    if args.train_both:
        policies_to_train.append(policy_ids[1])

    # Logging
    experiment_name: str = get_experiment_name(policy_ids, multiagent_mode, layout_name,
                                               args.experiment_tag, args.no_anneal)
    log_dir = args.log_dir

    ray.init(
        num_cpus=5,
        ignore_reinit_error=True,
        include_dashboard=False,
    )

    trainer = get_trainer(multiagent_mode, experiment_name, log_dir, num_training_iters, policies_to_train,
                          overcooked_env_config, policies, seed, train_extras)

    return trainer, empowerment_model


def train(trainer: PPOTrainerCustom, num_training_iters: int, save_freq: int,
          empowerment_model: Optional[Empowerment]) -> None:
    result = None
    for _ in range(num_training_iters):
        print(f"Starting training iteration {trainer.iteration}")
        result = trainer.train()

        if trainer.iteration % save_freq == 0:
            checkpoint = trainer.save()

            if empowerment_model is not None:
                checkpoint_dir = os.path.dirname(checkpoint)
                empowerment_model.save_to_folder(os.path.join(checkpoint_dir, "empowerment_model"))

            print(f"Saved checkpoint to {checkpoint}")

    checkpoint = trainer.save()
    print(f"Saved final checkpoint to {checkpoint}")


def main(args: argparse.Namespace) -> None:
    wandb_mode = "disabled" if args.no_wandb else "enabled"
    run = wandb.init(project="Entropy", config={}, mode=wandb_mode)  # TODO: Add config

    checkpoint_save_freq = 25

    trainer, empowerment_model = setup_experiment(args)
    train(trainer, args.num_training_iters, checkpoint_save_freq, empowerment_model)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layout_name",
        type=str,
        required=True,
    )
    parser.add_argument("--seed", type=int, default=np.random.randint(1000, high=100000))
    parser.add_argument("--num_training_iters", type=int, default=200)
    parser.add_argument("--log_dir", type=str, default="data/logs")
    parser.add_argument("--experiment_tag", type=str, required=False, default="")
    parser.add_argument("--human_model_checkpoint", type=str, required=False)
    parser.add_argument("--num_littered_objects", type=int, default=0)
    # parser.add_argument("--smirl", action="store_true")
    parser.add_argument("--model_0", type=str,
                        required=True)  # "ppo", "smirl", "smirl_e", "contrastive_e" or a model checkpoint
    parser.add_argument("--model_1", type=str)
    parser.add_argument("--train_both",
                        action="store_true")  # We always train model 0, but this is if we want to train model 1
    parser.add_argument("--no_anneal", action="store_true")
    parser.add_argument("--empowerment_weight", type=float, default=1)
    parser.add_argument("--yell", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--goal_prob", type=float, default=0.2)
    args = parser.parse_args()

    wandb.login()
    main(args)
