import ray.cloudpickle as cloudpickle

checkpoint_fname = "data/logs/cross_play/SMIRL_PPO/coordination_ring_nook_adversarial/2023-11-12_20-56-18/checkpoint_002000/checkpoint-2000"
with open(checkpoint_fname, "rb") as checkpoint_file:
    checkpoint_data = cloudpickle.load(checkpoint_file)
worker_data = cloudpickle.loads(checkpoint_data["worker"])
