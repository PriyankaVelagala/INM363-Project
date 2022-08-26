import gym
import random
import time

import matplotlib.pyplot as plt

import vizdoom_gym.envs.VizDoomEnv
import PIL.Image as Image

import helper_fuctions as helper


if __name__ == "__main__":

    sectors, health_pos, armor_pos = helper.get_env_layout()

    env = gym.make('VizDoomVeryDenseReward-v0',
                   config_file="custom\\very_dense_reward.cfg",
                   scenario_file="custom\\very_dense_reward.wad")

    obs = env.reset()

    j = 0
    num_episodes = 1
    path = []
    episode = 0

    while True:
        action = helper.random_action(env)
        ob, reward, done, info = env.step(action)
        # print("Total Reward:", reward)

        """
        if info["StateNum"] == 82:
            plt.imshow(ob)
            plt.show()
            time.sleep(10)
            break
        """

        if len(info) > 0:
            path.append((info["X"], info["Y"]))

        if j % 10 == 0:
            if len(info) > 0:
                helper.print_metadata(info)

        # env.render()
        time.sleep(0.01)

        if done:
            episode += 1
            print("Resetting the environment.")
            env.reset()

        if episode == num_episodes:
            print("Exiting env")
            break

        j += 1

    env.close()

    helper.plot_layout(sectors, health_pos, armor_pos, path)
