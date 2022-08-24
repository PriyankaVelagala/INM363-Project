from src.vizdoom_gym.envs.VizDoomEnv import VizdoomEnv


class VizdoomMyWayHome(VizdoomEnv):
    def __init__(self, **kwargs):
        super(VizdoomMyWayHome, self).__init__(0, **kwargs)


class VizDoomVeryDenseReward(VizdoomEnv):
    def __init__(self, **kwargs):
        super(VizDoomVeryDenseReward, self).__init__(1, **kwargs)
