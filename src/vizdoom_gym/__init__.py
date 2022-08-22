from gym.envs.registration import register

register(
    id='VizdoomMyWayHome-v0',
    entry_point='vizdoom_gym.envs:VizdoomMyWayHome'
)

register(
    id='VizDoomVeryDenseReward-v0',
    entry_point='vizdoom_gym.envs:VizDoomVeryDenseReward'
)
