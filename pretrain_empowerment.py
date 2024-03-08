import argparse

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

if __name__ == "__main__":
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
    parser.add_argument("--no_smirl", action="store_true")
    parser.add_argument("--yell", action="store_true")
    parser.add_argument("--goal_prob", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    checkpoint_path: str = args.checkpoint_path

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
        0, OvercookedGridworld.from_layout_name(args.layout_name)
    )
    env_params = {  # noqa: F841
        "horizon": horizon,
        "start_state_fn": start_state_fn,
        "num_mdp": 1
    }
    mdp_params = {
        "layout_name": args.layout_name,
        "rew_shaping_params": rew_shaping_params,
        "smirl": not args.no_smirl,
        "yell": args.yell
    }

    num_games = args.num_games
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
    obs_shape = get_obs_shape_from_layout(args.layout_name)
    empowerment_model = ContrastiveEmpowerment(num_actions=NUM_ACTIONS, in_channels=26, obs_shape=obs_shape,
                                               device=device,
                                               prob=args.goal_prob)

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

    batches: List[Dict] = [{"ppo": {
        "obs" : results["ep_states"][i],
        "actions" : results["ep_actions"][i]
    }} for i in range(num_games)]

    for epoch in range(args.num_epochs):
        batch_index = np.random.randint(0, len(batches))
        empowerment_model.modelUpdate(batches[batch_index])


    import pdb; pdb.set_trace()
