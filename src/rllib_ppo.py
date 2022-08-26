from src.vizdoom_gym.envs.VizDoomEnv import VizdoomEnv
from ray.tune.registry import register_env
import gym
import os
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

import ray.rllib.agents.ppo as ppo
import shutil

# https://github.com/DerwenAI/gym_example/blob/main/train.py


def env_creator():
    import gym
    return gym.make('VizDoomVeryDenseReward-v0',
                    config_file="custom\\very_dense_reward.cfg",
                    scenario_file="custom\\very_dense_reward.wad")


if __name__ == "__main__":

    ray.init(ignore_reinit_error=True, local_mode=True)

    tune.register_env('my_env', env_creator)


    # configure the environment and create agent
    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    config["num_workers"] = 1
    agent = ppo.PPOTrainer(config, env="my_env")

    while True:
        print(agent.train())

