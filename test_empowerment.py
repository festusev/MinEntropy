from Empowerment import Empowerment
from bpd.envs.overcooked import (
    OvercookedCallbacks,
    OvercookedMultiAgent,
    get_littered_start_state_fn,
)
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
import torch
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
empowerment_model = Empowerment(in_channels=26, device=device)

num_training_iters = 3000
train_batch_size = 10000

layout_name = "coordination_ring"
rew_shaping_params = {
    "PLACEMENT_IN_POT_REW": 3,
    "DISH_PICKUP_REWARD": 3,
    "SOUP_PICKUP_REWARD": 5,
    "DISH_DISP_DISTANCE_REW": 0,
    "POT_DISTANCE_REW": 0,
    "SOUP_DISTANCE_REW": 0,
}
compute_smirl = True
horizon = 400
num_littered_objects = 0
start_state_fn = get_littered_start_state_fn(
        num_littered_objects, OvercookedGridworld.from_layout_name(layout_name)
    )

# Reward shaping
use_phi = False  # Whether dense reward should come from potential function or not
# Constant by which shaped rewards are multiplied by when calculating total reward
reward_shaping_factor = 1.0
# Linearly anneal the reward shaping factor such that it reaches zero after this
# number of timesteps
reward_shaping_horizon = num_training_iters * train_batch_size // 2
# Whether the agents should both get all dense rewards.
share_dense_reward = False
dispense_reward = 0
no_regular_reward = False
action_rewards = [0] * Action.NUM_ACTIONS

policy_ids = ["smirl_e_0", "ppo_0"]

env_config = {
    # To be passed into OvercookedGridWorld constructor
    "mdp_params": {
        "layout_name": layout_name,
        "rew_shaping_params": rew_shaping_params,
        "empowerment_model": empowerment_model,
        "smirl": compute_smirl
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
        "agents": policy_ids
    },
}

overcooked_env_config = env_config
env = OvercookedMultiAgent.from_config(overcooked_env_config)

def rollout(n):
    def update(lst, val):
        lst["smirl_e_0"].append(val["smirl_e_0"])
        lst["ppo_0"].append(val["ppo_0"])

    all_obs = []
    all_actions = []
    all_rewards = []
    all_infos = []
    for i in range(n):
        for lst in [all_obs, all_actions, all_rewards, all_infos]:
            lst.append({"smirl_e_0": [], "ppo_0": []})

        obs = env.reset()

        all_obs[-1]["smirl_e_0"].append(obs["smirl_e_0"])
        all_obs[-1]["ppo_0"].append(obs["ppo_0"])


        done = False
        while not done:
            action = {"ppo_0": env.action_space.sample(), "smirl_e_0": env.action_space.sample()}
            obs, rewards, dones, infos = env.step(action)

            update(all_obs[-1], obs)
            update(all_actions[-1], action)
            update(all_rewards[-1], rewards)
            update(all_infos[-1], infos)

            done = dones["__all__"]
    return all_obs, all_actions, all_rewards, all_infos

def get_multiagent_batch():
    all_obs, all_actions, all_rewards, all_infos = rollout(1)

    batch = []
    for policy_id in policy_ids:
        batch.append({"obs": np.array(all_obs[0][policy_id])[:-1], "new_obs": np.array(all_obs[0][policy_id][1:]),
                 "actions": np.array(all_actions[0][policy_id])})

    return batch

validation = get_multiagent_batch()

train_losses = []
val_losses = []

pbar = tqdm(range(1000))
for epoch in pbar:
    batch = get_multiagent_batch()

    empowerment_model.train()
    sm_loss, tr_loss = empowerment_model(batch)

    empowerment_model.eval()
    with torch.no_grad():
        val_sm_loss, val_tr_loss = empowerment_model.get_loss(validation)

    train_losses.append([sm_loss, tr_loss])
    val_losses.append([val_sm_loss, val_tr_loss])

    pbar.set_description(f"Epoch {epoch} sm_loss: {sm_loss} tr_loss: {tr_loss} val_sm_loss {val_sm_loss} val_tr_loss {val_tr_loss}")

import pdb; pdb.set_trace()
