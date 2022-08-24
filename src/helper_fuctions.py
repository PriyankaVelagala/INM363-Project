import random
import gym
import vizdoom as vzd
from pathlib import Path
from matplotlib import pyplot as plt

CONFIG_DIR = "C:\\Users\\priya\\OneDrive - City, University of London\\Documents\\GitHub\\INM363-Project\\scenarios\\"
CONFIG_FILE = "custom\\very_dense_reward.cfg"
SCENARIO_FILE = "custom\\very_dense_reward.wad"

DEBUG = True


# generate random int between 0 and n available actions
def random_action(env):
    actions = env.action_space.n - 1
    return random.randint(0, actions)


# print agent metadata
def print_metadata(info):
    print(f"StateNum: {info['StateNum']}, "
          f"X: {info['X']}, "
          f"Y: {info['Y']}, "
          f"ItemCount: {info['ItemCount']}, "
          f"Health: {info['Health']}, "
          f"Shaping Reward: {info['ShapingReward']}, "
          f"Total Reward: {info['TotalReward']}"
          )


def get_env_layout(config=CONFIG_FILE, scenario=SCENARIO_FILE):
    config_path = Path(CONFIG_DIR, config)
    scenario_path = Path(CONFIG_DIR, scenario)
    game = vzd.DoomGame()
    game.load_config(config_path.as_posix())
    game.set_doom_scenario_path(scenario_path.as_posix())
    game.set_mode(vzd.Mode.SPECTATOR)
    game.set_window_visible(False)
    game.init()

    health_pos = []
    armor_pos = []
    sectors = []

    game.new_episode()
    state = game.get_state()

    for o in state.objects:
        if o.name == "HealthBonus":
            health_pos.append((o.position_x, o.position_y))
        elif o.name == "GreenArmor":
            armor_pos.append((o.position_x, o.position_y))

    if DEBUG:
        print(f"Health Bonus locations: {health_pos}")
        print(f"Armor location: {armor_pos}")

    for s in state.sectors:
        for l in s.lines:
            if l.is_blocking:
                sectors.append((l.x1, l.x2, l.y1, l.y2))
    if DEBUG:
        print(f"Sector locations: {sectors}")

    game.close()

    return sectors, health_pos, armor_pos


def plot_layout(sectors, health_kit, armor, path=[]):
    # plot rooms:
    for sector in sectors:
        plt.plot([sector[0], sector[1]], [sector[2], sector[3]], color='black', linewidth=2)

    # plot path taken
    for pos in path:
        plt.plot(pos[0], pos[1], color='green', marker='o')

    # plot positions of health bonus and armor:
    for pos in health_kit:
        plt.plot(pos[0], pos[1], color='blue', marker='x')
    for pos in armor:
        plt.plot(pos[0], pos[1], color='red', marker='x')

    plt.axis("off")
    plt.show()

    return
