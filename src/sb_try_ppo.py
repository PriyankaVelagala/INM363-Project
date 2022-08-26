from stable_baselines3.common.env_checker import check_env
import gym
import random
import time
import matplotlib.pyplot as plt
import vizdoom_gym.envs.VizDoomEnv
import PIL.Image as Image
import helper_fuctions as helper
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import helper_fuctions as helper


#https://towardsdatascience.com/beginners-guide-to-custom-environments-in-openai-s-gym-989371673952

if __name__ == "__main__":

    env = gym.make('VizDoomVeryDenseReward-v0',
                   config_file="custom\\very_dense_reward.cfg",
                   scenario_file="custom\\very_dense_reward.wad")

    # checks if env compatible with stable baselines
    # check_env(env)
    # env.close()

    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=20)
    obs = env.reset()

    for i in range(1):
        action, states = model.predict(obs)
        print(action)
        obs, rewards, dones, info = env.step(action)
        helper.print_metadata(info)


    env.close()
