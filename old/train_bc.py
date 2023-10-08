import argparse
import glob
import os

import ray
import torch
from bpd.agents.bc import BCTrainer
from bpd.envs.overcooked import (
    OvercookedMultiAgent,
    build_overcooked_eval_function,
    load_human_trajectories_as_sample_batch,
)
from bpd.training_utils import build_logger_creator
from ray.rllib.agents.trainer import COMMON_CONFIG, Trainer
from ray.rllib.offline.json_writer import JsonWriter
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.typing import ModelConfigDict, TrainerConfigDict
from typing_extensions import Literal


def get_human_offline_data(
    human_data_fname: str,
    layout_name: str,
    *,
    use_bc_features=True,
    num_trajectories: int = 8,
) -> str:
    offline_data_fname = (
        ".".join(human_data_fname.split(".")[:-1])
        + f"_{layout_name}_{num_trajectories}_traj"
    )
    if not use_bc_features:
        offline_data_fname += "_ppo_features"
    if len(glob.glob(os.path.join(offline_data_fname, "*.json"))) == 0:
        human_data_sample_batch = load_human_trajectories_as_sample_batch(
            human_data_fname,
            layout_name,
            featurize_fn_id="bc" if use_bc_features else "ppo",
            traj_indices=set(range(num_trajectories)),
        )
        offline_writer = JsonWriter(offline_data_fname)
        offline_writer.write(human_data_sample_batch)
    return offline_data_fname


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_training_iters", type=int, default=20)
    parser.add_argument("--use_raw_features", action="store_true", default=False)
    parser.add_argument("--log_dir", type=str, default="data/logs")
    parser.add_argument(
        "--data_split", type=str, default="train", choices=["train", "test"]
    )
    parser.add_argument("--num_trajectories", type=int, default=8)
    args = parser.parse_args()

    # Environment
    layout_name = args.layout_name

    # Training
    num_workers = 0
    seed = 0
    num_gpus = 1 if torch.cuda.is_available() else 0
    sgd_minibatch_size = 64
    num_training_iters = args.num_training_iters
    lr = 1e-3
    use_bc_features = not args.use_raw_features
    data_split: Literal["train", "test"] = args.data_split
    num_trajectories = args.num_trajectories

    # Model
    model_config: ModelConfigDict
    if use_bc_features:
        num_hidden_layers = 2
        size_hidden_layers = 64
        model_config = {
            "fcnet_hiddens": [size_hidden_layers] * num_hidden_layers,
            "fcnet_activation": "relu",
        }
    else:
        num_hidden_layers = 3
        size_hidden_layers = 64
        num_filters = 25
        num_conv_layers = 3
        model_config = {
            "custom_model": "overcooked_ppo_model",
            "vf_share_layers": True,
            "custom_model_config": {
                "num_hidden_layers": num_hidden_layers,
                "size_hidden_layers": size_hidden_layers,
                "num_filters": num_filters,
                "num_conv_layers": num_conv_layers,
            },
        }

    # Logging
    save_freq = 5
    log_dir = args.log_dir
    experiment_tag = None
    experiment_name_parts = ["bc", layout_name, data_split, f"{num_trajectories}_traj"]
    if experiment_tag is not None:
        experiment_name_parts.append(experiment_tag)
    experiment_name = os.path.join(*experiment_name_parts)  # noqa: F841

    # Human data
    human_data_fname = f"data/human_data/human_data_state_dict_and_action_by_traj_{data_split}_inserted_fixed.pkl"

    environment_params = {
        "mdp_params": {
            "layout_name": layout_name,
        },
        "env_params": {"horizon": 1},
        "multi_agent_params": {},
    }

    env = OvercookedMultiAgent.from_config(environment_params)

    # Validation
    validation_prop = 0

    # Evaluation
    evaluation_interval = 1000
    evaluation_ep_length = 400
    evaluation_num_games = 10
    evaluation_display = False

    config: TrainerConfigDict = {  # noqa: F841
        "env": "overcooked_multi_agent",
        "env_config": environment_params,
        "num_workers": num_workers,
        "sgd_minibatch_size": sgd_minibatch_size,
        "lr": lr,
        "lr_drop_patience": 5,
        "validation_prop": validation_prop,
        "num_gpus": num_gpus,
        "seed": seed,
        "framework": "torch",
        "evaluation_interval": evaluation_interval,
        "custom_eval_function": build_overcooked_eval_function(
            eval_params={
                "ep_length": evaluation_ep_length,
                "num_games": evaluation_num_games,
                "display": evaluation_display,
            },
            eval_mdp_params=environment_params["mdp_params"],
            env_params=environment_params["env_params"],
            outer_shape=None,
            agent_0_policy_str=DEFAULT_POLICY_ID,
            agent_1_policy_str=DEFAULT_POLICY_ID,
            use_bc_featurize_fn=use_bc_features,
        ),
        "multiagent": {
            "policies": {
                DEFAULT_POLICY_ID: (
                    None,
                    env.bc_observation_space
                    if use_bc_features
                    else env.ppo_observation_space,
                    env.action_space,
                    {
                        "model": model_config,
                    },
                )
            },
        },
    }

    if "disable_env_checking" in COMMON_CONFIG:
        config["disable_env_checking"] = True

    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
    )

    config["input"] = get_human_offline_data(
        human_data_fname,
        layout_name,
        use_bc_features=use_bc_features,
        num_trajectories=num_trajectories,
    )
    trainer: Trainer = BCTrainer(
        config,
        logger_creator=build_logger_creator(
            log_dir,
            experiment_name,
        ),
    )

    result = None
    for train_iter in range(num_training_iters):
        print(f"Starting training iteration {train_iter}")
        result = trainer.train()

        if trainer.iteration % save_freq == 0:
            checkpoint = trainer.save()
            print(f"Saved checkpoint to {checkpoint}")

    checkpoint = trainer.save()
    print(f"Saved final checkpoint to {checkpoint}")
