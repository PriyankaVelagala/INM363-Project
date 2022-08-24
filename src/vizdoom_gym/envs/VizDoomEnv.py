import gym
from gym import spaces
# from gym.envs.classic_control import rendering
import vizdoom as vzd
from pathlib import Path
import numpy as np
import multiprocessing

"""
sources:
- Using PyDoom
ICM: https://github.com/pathak22/noreward-rl/blob/master/doomFiles/doom_env.py
- Using VizDoom 
VizDoom Gym: https://github.com/shakenes/vizdoomgym/blob/master/vizdoomgym/envs/vizdoomenv.py
Gym Doom: https://github.com/MarvineGothic/gym_doom/tree/5cda5bbb52c0e82e9fd2ed355d2c7e84c767603b
"""
# use synchronous modes: PLAYER(agent)/SPECTATOR(human) (similar to ICM paper)



# TODO: print observations
# TODO: render method
# TODO: low priority: record screen
# TODO: check if lock needed for multiprocessing
# TODO: frame stacking?

# TODO: add data structures to store sector/armor/health kit position - need to do this in a seperate helper class? -- DONE
# TODO: add data structure to store agent path traversal for one episode? - not sure how this would work? --DONE
# TODO: check if reset method needs a return value -- DONE


CONFIG_DIR = "C:\\Users\\priya\\OneDrive - City, University of London\\Documents\\GitHub\\INM363-Project\\scenarios\\"

CONFIGS = [
    ["default\\my_way_home.cfg", 5],  # level 0
    ["custom\\very_dense_reward.cfg", 3]  # level 1
]

DEBUG = True


class VizdoomEnv(gym.Env):
    def __init__(self, level, **kwargs):
        # get keyword args to initialize env
        self.mode = kwargs.get("mode", "SPECTATOR")
        self.config_file = kwargs.get("config_file", False)
        self.scenario_file = kwargs.get("scenario_file", False)
        # self.training = kwargs.get("training", False)

        # store last shaping reward
        self.last_shaping_reward = 0
        # frame repeat
        self.frame_repeat = 4
        # force mode to algo for now
        self.mode = 'human'

        # initialize game instance
        self.game = vzd.DoomGame()
        if DEBUG:
            print(f"config file: {Path(CONFIG_DIR, self.config_file).as_posix()}")
            print(f"scenario file: {Path(CONFIG_DIR, self.scenario_file).as_posix()}")
        self.game.load_config(Path(CONFIG_DIR, self.config_file).as_posix())
        self.game.set_doom_scenario_path(Path(CONFIG_DIR, self.scenario_file).as_posix())
        # set mode
        if self.mode == 'algo':
            self.game.set_mode(vzd.Mode.PLAYER)
        elif self.mode == 'human':
            self.game.set_mode(vzd.Mode.SPECTATOR)
        else:
            raise NotImplementedError

        # change this for training/testing
        self.game.set_window_visible(True)
        self.game.init()

        self.state = None
        self.viewer = None

        # only 3 actions available: move left/right/forward
        # self.action_space = spaces.Discrete(3)
        # controlled by actions spaces in CONFIGS
        self.action_space = spaces.Discrete(CONFIGS[level][1])

        # Observation space
        # screen size = 640 x 480 - height = 480, width = 640
        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=(self.game.get_screen_height(),
                                                   self.game.get_screen_width(),
                                                   self.game.get_screen_channels()))

    def step(self, action):
        # convert action to vizdoom action space (one hot)
        act = np.zeros(self.action_space.n)
        act[action] = 1
        act = np.uint8(act)
        act = act.tolist()

        # if frame skipping enabled
        if self.frame_repeat:
            reward = self.game.make_action(act, self.frame_repeat)
        else:
            reward = self.game.make_action(act)

        self.state = self.game.get_state()
        done = self.game.is_episode_finished()

        if done:
            obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        else:
            # not sure if transpose is needed
            obs = np.transpose(self.state.screen_buffer, (1, 2, 0))

        # get shaping reward
        shaping_reward = vzd.doom_fixed_to_double(self.game.get_game_variable(vzd.GameVariable.USER1))
        shaping_reward = shaping_reward - self.last_shaping_reward
        self.last_shaping_reward += shaping_reward

        # add to total reward
        reward += shaping_reward

        # get other game variables to info
        if self.state is not None:
            info = {"StateNum": self.state.number,
                    "X": self.state.game_variables[0],
                    "Y": self.state.game_variables[1],
                    "ItemCount": self.state.game_variables[2],
                    "Health": self.state.game_variables[3],
                    "ShapingReward": shaping_reward,
                    "TotalReward": self.game.get_total_reward() + self.last_shaping_reward}
        else:
            info = {}

        return obs, reward, done, info

    # Resets episode
    def reset(self):
        self.game.new_episode()
        # might not need to do this
        self.last_shaping_reward = 0
        self.state = self.game.get_state()
        return self.state

    def render(self, mode="algo"):
        img = self.game.get_state().screen_buffer.copy()
        if mode == "algo":
            return img
        elif mode == "human":
            return img

    def close(self):
        # close viewer
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        # game cleanup
        self.game.close()
        return
