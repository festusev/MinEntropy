import argparse

from tqdm import tqdm
from bpd.envs.overcooked import evaluate
from bpd.training_utils import load_trainer
from bpd.envs.overcooked import get_littered_start_state_fn
from PPOTrainerCustom import PPOTrainerCustom
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from ray.rllib.policy.torch_policy import TorchPolicy
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from Empowerment import ClassifierEmpowerment, TwoHeadedEmpowerment, MIMIEmpowerment, ContrastiveEmpowerment
from overcooked_ai_py.mdp.actions import Action
import torch
from train import get_obs_shape_from_layout
from typing import Dict, List
import numpy as np
import wandb
import os
from datetime import datetime

def get_experiment_log_dir(log_prefix, experiment_name) -> str:
    experiment_log_dir: str = os.path.join(
        log_prefix,
        experiment_name,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    os.makedirs(experiment_log_dir, exist_ok=True)
    return experiment_log_dir


def pretrain_empowerment(checkpoint_path: str, layout_name: str, num_games: int, goal_prob: float,
                         batch_size: int, epochs: int, log_prefix: str) -> None:
    wandb_run = wandb.run
    log_dir = get_experiment_log_dir(log_prefix, wandb_run.name)
    print(f"Saving checkpoints to {log_dir}")

    wandb_run.config["log_dir"] = log_dir

    trainer = load_trainer(
        checkpoint_path, PPOTrainerCustom,
        config_overrides={"input": "sampler", "execution_plan": {"train_extras": []}}
    )

    trainer_policies = list(trainer.workers.local_worker().policy_map.policy_specs.keys())

    policy_ids = []

    policy_0 = trainer.get_policy(trainer_policies[0])
    policy_1 = trainer.get_policy(trainer_policies[1])

    if "smirl" in trainer_policies[1] or "contrastive" in trainer_policies[1]:
        policy_0, policy_1 = policy_1, policy_0
        policy_ids = [trainer_policies[1], trainer_policies[0]]
    else:
        policy_ids = trainer_policies[1]

    assert isinstance(policy_0, TorchPolicy) and isinstance(policy_1, TorchPolicy)

    horizon = 400
    rew_shaping_params = {
        "PLACEMENT_IN_POT_REW": 0,
        "DISH_PICKUP_REWARD": 0,
        "SOUP_PICKUP_REWARD": 0,
        "DISH_DISP_DISTANCE_REW": 0,
        "POT_DISTANCE_REW": 0,
        "SOUP_DISTANCE_REW": 0,
    }

    start_state_fn = get_littered_start_state_fn(
        0, OvercookedGridworld.from_layout_name(layout_name)
    )
    env_params = {  # noqa: F841
        "horizon": horizon,
        "start_state_fn": start_state_fn,
        "num_mdp": 1
    }
    mdp_params = {
        "layout_name": layout_name,
        "rew_shaping_params": rew_shaping_params,
        "smirl": False,
        "yell": False
    }

    eval_params = {
        "ep_length": horizon,
        "num_games": num_games,
        "display": False,
    }

    evaluator = AgentEvaluator.from_layout_name(
        mdp_params=mdp_params,
        env_params=env_params,
    )
    env: OvercookedEnv = evaluator.env

    bc_obs_shape = env.featurize_state_mdp(env.mdp.get_standard_start_state())[0].shape

    def get_featurize_fn(policy: TorchPolicy):
        if policy.observation_space.shape == bc_obs_shape:
            return lambda state: env.featurize_state_mdp(state)
        else:
            return env.lossless_state_encoding_mdp

    print("Constructing Empowerment Model")
    NUM_ACTIONS = Action.NUM_ACTIONS
    device = torch.device("cpu")
    obs_shape = get_obs_shape_from_layout(layout_name)
    empowerment_model = ContrastiveEmpowerment(num_actions=NUM_ACTIONS, in_channels=26, obs_shape=obs_shape,
                                               device=device,
                                               prob=goal_prob, batch_size=batch_size, buffer_max_size=400)

    print("Generating Rollouts")
    results: Dict = evaluate(
        eval_params=dict(eval_params),
        mdp_params=mdp_params,
        outer_shape=None,
        agent_0_policy=policy_0,
        agent_1_policy=policy_1,
        agent_0_featurize_fn=get_featurize_fn(policy_0),
        agent_1_featurize_fn=get_featurize_fn(policy_1),
    )

    empowerment_model.train()

    ep_states = []
    ep_actions = []

    for i in range(num_games):
        ep_states.append([env.lossless_state_encoding_mdp(state)[0] for state in results["ep_states"][i]])
        ep_actions.append([Action.ACTION_TO_INDEX[action_pair[0]] for action_pair in
                           results["ep_actions"][i]])  # Todo: Make this [0] random, to randomly select an expert
    ep_states = np.array(ep_states)
    ep_actions = np.array(ep_actions)

    batches: List[Dict] = [{"ppo": {
        "obs": ep_states[i][:-1],
        "actions": ep_actions[i][:-1],
        "new_obs": ep_states[i][1:]
    }} for i in range(num_games)]

    import pdb; pdb.set_trace()

    pbar = tqdm(range(epochs))
    for epoch in pbar:
        if epoch % 5 == 0:
            empowerment_checkpoint_dir = os.path.join(log_dir, f"{epoch:03}")
            os.makedirs(empowerment_checkpoint_dir, exist_ok=True)

            empowerment_model.save_to_folder(empowerment_checkpoint_dir)

        batch_index = np.random.randint(0, len(batches))
        empowerment_model.modelUpdate(batches[batch_index])
        pbar.set_description(
            f"{epoch}: empowerment_classifier_loss: {empowerment_model.info['empowerment_classifier_loss']}")

        wandb_run.log(
            {"empowerment_classifier_loss": empowerment_model.info['empowerment_classifier_loss'], "epoch": epoch})


if __name__ == "__main__":
    wandb.login()

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument(
        "--layout_name",
        type=str,
        help="name of the Overcooked layout to evaluate on",
        required=True,
    )
    parser.add_argument(
        "--num_games", "-n", type=int, default=20
    )
    parser.add_argument("--goal_prob", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--log_prefix", type=str, default="data/logs/pretrain_empowerment")
    parser.add_argument("--batch_size", "-B", type=int, default=128)
    args = parser.parse_args()

    wandb_mode = "disabled" if args.no_wandb else "online"
    run = wandb.init(project="Entropy",
                     name=f"{args.layout_name}_n{args.num_games}_e{args.epochs}_g{args.goal_prob}_B{args.batch_size}",
                     group="pretrain_empowerment", config=args, mode=wandb_mode)  # TODO: Add config

    pretrain_empowerment(args.checkpoint_path, args.layout_name, args.num_games, args.goal_prob, args.batch_size,
                         args.epochs, args.log_prefix)
