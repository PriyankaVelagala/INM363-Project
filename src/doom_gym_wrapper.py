import gym
from gym import spaces
import vizdoom as vzd
from pathlib import Path
from typing import List
import numpy as np

"""
sources:
https://github.com/pathak22/noreward-rl/blob/master/doomFiles/doom_env.py
https://github.com/shakenes/vizdoomgym/blob/master/vizdoomgym/envs/vizdoomenv.py
"""
# TODO: render method
# TODO: check if reset method needs a return value
# TODO: add data structures to store dector/armor/health kit position
# TODO: add data structure to store agent path traversal for one episode? - not sure how this would work?
# TODO: frame stacking?
# TODO: registering config files in gym

CONFIG_DIR = "C:\\Users\\priya\\OneDrive - City, University of London\\Documents\\GitHub\\INM363-Project\\scenarios\\"

CONFIGS = [
    ["my_way_home.cfg", 5],
]


class VizdoomEnv(gym.Env):

    def __init__(self, level, **kwargs):
        # get keyword args to initialize env
        self.mode = kwargs.get("mode", "SPECTATOR")
        self.config_file = kwargs.get("config_file", False)
        self.scenario_file = kwargs.get("scenario_file", False)
        self.training = kwargs.get("training", False)

        # store last shaping reward
        self.last_shaping_reward = 0

        # initialize game instance
        self.game = vzd.DoomGame()
        self.game.load_config(Path(CONFIG_DIR, self.config_file).as_posix())
        self.game.set_doom_scenario_path(Path(CONFIG_DIR, self.scenario_file).as_posix())
        self.game.set_mode(vzd.Mode.SPECTATOR)
        self.game.init()

        if self.training:
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)

        self.state = None
        self.viewer = None

        # only 3 actions available: move left/right/forward
        self.action_space = spaces.Discrete(3)

        # Observation spaces
        # screen size = 640 x 480 - height = 480, width = 640
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.game.get_screen_height(),
                                                                    self.game.get_screen_width(),
                                                                    self.game.get_screen_channels()))

    def step(self, action):
        # convert action to vizdoom action space (one hot)
        act = np.zeros(self.action_space.n)
        act[action] = 1
        act = np.uint8(act)
        act = act.tolist()

        reward = self.game.make_action(act)
        self.state = self.game.get_state()
        done = self.game.is_episode_finished()

        if done:
            obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        else:
            # not sure if transpose is needed
            obs = np.tranpose(self.state.screen_buffer, (1, 2, 0))

        # get shaping reward
        shaping_reward = vzd.doom_fixed_to_double(self.game.get_game_variable(vzd.GameVariable.USER1))
        shaping_reward = shaping_reward - self.last_shaping_reward
        self.last_shaping_reward += shaping_reward

        # add to total reward
        reward += shaping_reward

        # get other game variables to info
        info = {"StateNum": self.state.number,
                "X": self.state.game_variables[0],
                "Y": self.state.game_variables[1],
                "ItemCount": self.state.game_variables[2],
                "Health": self.state.game_variables[3],
                "ShapingReward": shaping_reward,
                "TotalReward": self.game.get_total_reward() + self.last_shaping_reward}

        return obs, reward, done, info

    # not sure if reset needs a return value
    def reset(self):
        self.game.new_episode()
        # might not need to do this
        self.last_shaping_reward = 0
        self.state = self.game.get_state()
        return

    def render(self):
        raise NotImplementedError





    """
    def __collect_observations(self):
        observation = []
        # not a terminal state
        if self.state is not None:
            observation.append(np.tranpose(self.state.screen_buffer, (1, 2, 0))) # check this transpose based on config
        # if terminal state
        else:
            if isinstance(self.observation_space, gym.spaces.box.Box):
                obs_space = [self.observation_space]
            else:
                obs_space = self.observation_space

            for space in obs_space:
                observation.append(np.zeros(space.shape, dtype=space.dtype))

        if len(observation) == 1:
            observation = observation[0]

        return observation
    """

