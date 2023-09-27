import argparse
import json
from typing import Dict, Optional, Type

import numpy as np
import ray
from bpd.agents.bc import BCTrainer
from bpd.envs.overcooked import (
    EpisodeInformation,
    evaluate,
    get_littered_start_state_fn,
)
from bpd.training_utils import load_trainer
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.trainer import Trainer
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.utils.typing import PolicyID
from tqdm import tqdm
from typing_extensions import Literal

RunStr = Literal["ppo", "bc"]
trainer_classes: Dict[RunStr, Type[Trainer]] = {
    "ppo": PPOTrainer,
    "bc": BCTrainer,
}

default_policy_ids: Dict[RunStr, PolicyID] = {
    "ppo": "ppo",
    "bc": DEFAULT_POLICY_ID,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_0",
        type=str,
        choices=list(map(str, trainer_classes.keys())),
        help="algorithm used to train the first agent",
        required=True,
    )
    parser.add_argument(
        "--checkpoint_path_0",
        type=str,
        help="path to checkpoint of the first agent",
        required=True,
    )
    parser.add_argument(
        "--run_1",
        type=str,
        choices=list(map(str, trainer_classes.keys())),
        help="algorithm used to train the second agent",
        required=True,
    )
    parser.add_argument(
        "--checkpoint_path_1",
        type=str,
        help="path to checkpoint of the second agent",
        required=True,
    )
    parser.add_argument(
        "--layout_name",
        type=str,
        help="name of the Overcooked layout to evaluate on",
        required=True,
    )
    parser.add_argument("--num_littered_objects", type=int, default=0)
    parser.add_argument(
        "--num_games",
        type=int,
        default=1,
        help="number of games to evaluate on",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=False,
        help="if specified, output the evaluation results to this path",
    )
    parser.add_argument(
        "--render_path",
        type=str,
        required=False,
        help="if specified, render the games to this path",
    )
    args = parser.parse_args()

    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
    )

    run_0: RunStr = args.run_0
    run_1: RunStr = args.run_1
    policy_id_0 = default_policy_ids[run_0]
    policy_id_1 = default_policy_ids[run_1]
    checkpoint_path_0: str = args.checkpoint_path_0
    checkpoint_path_1: str = args.checkpoint_path_1
    layout_name: str = args.layout_name
    num_littered_objects: int = args.num_littered_objects
    num_games: int = args.num_games
    out_path: Optional[str] = args.output_path
    render_path: Optional[str] = args.render_path

    trainer_0 = load_trainer(
        checkpoint_path_0, trainer_classes[run_0], config_overrides={"input": "sampler"}
    )
    policy_0 = trainer_0.get_policy(policy_id_0)
    trainer_1 = load_trainer(
        checkpoint_path_1, trainer_classes[run_1], config_overrides={"input": "sampler"}
    )
    policy_1 = trainer_1.get_policy(policy_id_1)
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
        num_littered_objects, OvercookedGridworld.from_layout_name(layout_name)
    )
    env_params = {  # noqa: F841
        "horizon": horizon,
        "start_state_fn": start_state_fn,
        "num_mdp": 1
    }
    mdp_params = {
        "layout_name": layout_name,
        "rew_shaping_params": rew_shaping_params,
        "smirl": True
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

    results = evaluate(
        eval_params=dict(eval_params),
        mdp_params=mdp_params,
        outer_shape=None,
        agent_0_policy=policy_0,
        agent_1_policy=policy_1,
        agent_0_featurize_fn=get_featurize_fn(policy_0),
        agent_1_featurize_fn=get_featurize_fn(policy_1),
    )

    ep_returns = [int(ep_return) for ep_return in results["ep_returns"]]
    simple_results = {
        "ep_returns": ep_returns,
        "mean_return": float(np.mean(ep_returns)),
    }
    all_results = {
        "results": results,
    }
    ep_returns_all = list(ep_returns)

    results_flipped = evaluate(
        eval_params=dict(eval_params),
        mdp_params=mdp_params,
        outer_shape=None,
        agent_0_policy=policy_1,
        agent_1_policy=policy_0,
        agent_0_featurize_fn=get_featurize_fn(policy_1),
        agent_1_featurize_fn=get_featurize_fn(policy_0),
    )
    ep_returns_flipped = [int(ep_return) for ep_return in results_flipped["ep_returns"]]
    simple_results.update(
        {
            "ep_returns_flipped": list(ep_returns_flipped),
            "mean_return_flipped": float(np.mean(ep_returns_flipped)),
        }
    )
    ep_returns_all.extend(ep_returns_flipped)
    all_results["results_flippped"] = results_flipped

    simple_results.update(
        {
            "ep_returns_all": ep_returns_all,
            "mean_return_all": float(np.mean(ep_returns_all)),
        }
    )

    if render_path:
        import pygame
        import skvideo.io
        from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

        for episode_index, (episode_states, episode_infos) in enumerate(
            zip(results["ep_states"], results["ep_infos"])
        ):
            video_writer = skvideo.io.FFmpegWriter(
                f"{render_path}_{episode_index}.mp4",
                outputdict={
                    "-filter:v": "setpts=5*PTS",
                    "-pix_fmt": "yuv420p",
                },
            )
            state: OvercookedState
            info: EpisodeInformation
            for state, info in tqdm(
                zip(episode_states, episode_infos),
                desc=f"Rendering episode {episode_index}",
            ):
                state_frame = pygame.surfarray.array3d(
                    StateVisualizer(tile_size=60).render_state(
                        state,
                        grid=evaluator.env.mdp.terrain_mtx,
                        action_probs=None,
                    )
                ).transpose((1, 0, 2))
                video_writer.writeFrame(state_frame)
            video_writer.close()

    if out_path is not None:
        with open(out_path, "w") as simple_results_file:
            print(f"Saving results to {out_path}")
            json.dump(simple_results, simple_results_file)
