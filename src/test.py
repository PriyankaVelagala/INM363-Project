from src.vizdoom_gym.envs.VizDoomEnv import VizdoomEnv
from src.vizdoom_gym.envs.VizDoomEnv_def import VizDoomVeryDenseReward

from ray.tune.registry import register_env
import gym
import os
import ray
import ray.rllib.agents.ppo as ppo
import shutil

if __name__ == "__main__":

    print("Hello world")

    chkpt_root = "tmp/exa"
    shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)

    print("here 1")

    ray_results = "{}/ray_results/".format(os.getenv("HOME"))
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

    print("here 2")

    ray.shutdown()
    print("Shutdown ray")
    # start Ray -- add `local_mode=True` here for debugging
    ray.init(ignore_reinit_error=True, local_mode=True)
    print("here 3: initialized ray")

    # register the custom environment
    select_env = "VizDoomVeryDenseReward-v0"

    register_env(select_env, lambda config: VizDoomVeryDenseReward())
    #register_env(select_env, lambda config: VizdoomEnv())

    print("registered environment")

    # configure the environment and create agent
    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    agent = ppo.PPOTrainer(config, env=select_env)

    print("created agent")

    status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
    n_iter = 3

    print("started trainign loop")

    # train a policy with RLlib using PPO
    for n in range(n_iter):
        result = agent.train()
        chkpt_file = agent.save(chkpt_root)

        print(status.format(
            n + 1,
            result["episode_reward_min"],
            result["episode_reward_mean"],
            result["episode_reward_max"],
            result["episode_len_mean"],
            chkpt_file
        ))

    print("ending training loop")

    ray.shutdown()
    print("shutdown ray")


