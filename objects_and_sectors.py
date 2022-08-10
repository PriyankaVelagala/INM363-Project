#modified from objects_and_sectors.py
#draws agent's path traversal at end of episode

#####################################################################
# This script presents how to access the information
# about the actors and objects present in the current scenario
# and map sectors (geometry).
#
# This information is not available if "+viz_nocheat" flag is enabled.
#####################################################################

from random import choice
import vizdoom as vzd
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import os
from pathlib import Path

#DEFAULT_CONFIG = os.path.join(vzd.scenarios_path, "my_way_home.cfg")

CONFIG_DIR = "C:\\Users\\priya\\OneDrive - City, University of London\\Documents\\GitHub\\INM363-Project\\scenarios\\"
CONFIG_FILE = "default\\my_way_home.cfg"
SCENARIO_FILE = "default\\my_way_home.wad"

if __name__ == "__main__":
    # Set up game configuration
    config_path = Path(CONFIG_DIR, CONFIG_FILE)
    scenario_path = Path(CONFIG_DIR, SCENARIO_FILE)

    parser = ArgumentParser()
    parser.add_argument(dest="config",
                        default=config_path.as_posix(),
                        nargs="?")
    args = parser.parse_args()

    # Initialize game instance with configuration
    game = vzd.DoomGame()
    game.load_config(args.config)
    game.set_doom_scenario_path(scenario_path.as_posix())
    game.set_mode(vzd.Mode.SPECTATOR)

    # Enables information about all objects present in the current episode/level.
    game.set_objects_info_enabled(True)

    # Enables information about all sectors (map layout).
    game.set_sectors_info_enabled(True)

    game.clear_available_game_variables()
    game.add_available_game_variable(vzd.GameVariable.POSITION_X)
    game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
    game.add_available_game_variable(vzd.GameVariable.POSITION_Z)

    game.init()

    episodes = 1
    sleep_time = 2.0 #1.0 / vzd.DEFAULT_TICRATE  # = 0.028

    pos_x = []
    pos_y = []
    pos_z = []
    arm_pos_x = None
    arm_pos_y = None

    for i in range(episodes):
        print("Episode #" + str(i + 1))

        # Not needed for the first episode but the loop is nicer.
        game.new_episode()
        while not game.is_episode_finished():

            # Gets the state
            state = game.get_state()
            game.advance_action()
            last_action = game.get_last_action()
            reward = game.get_last_reward()

            for o in state.objects:
                if o.name == "DoomPlayer":
                    pos_x.append(o.position_x)
                    pos_y.append(o.position_y)
                else:
                    arm_pos_x = o.position_x
                    arm_pos_y = o.position_y

            if state.number % 10 == 0:
                print("State #" + str(state.number))
                print("Player position: x:", state.game_variables[0], ", y:", state.game_variables[1], ", z:",
                      state.game_variables[2])
                print("Objects:")

        # Print information about objects present in the episode.
        #for o in state.objects:
            #print("Object id:", o.id, "object name:", o.name)
            #print("Object position: x:", o.position_x, ", y:", o.position_y, ", z:", o.position_z)

            # Other available fields:
            # print("Object rotation angle", o.angle, "pitch:", o.pitch, "roll:", o.roll)
            # print("Object velocity x:", o.velocity_x, "y:", o.velocity_y, "z:", o.velocity_z)

            # Plot object on map
            #print("Object name:", o.name)
            #if o.name == "DoomPlayer":
        for coords in zip(pos_x, pos_y):
            plt.plot(coords[0], coords[1], color='green', marker='o')
        #else:

        plt.plot(arm_pos_x, arm_pos_y, color='red', marker='x')

        print("=====================")

        print("Sectors:")

        # Print information about sectors.
        for s in state.sectors:
            #print("Sector floor height:", s.floor_height, ", ceiling height:", s.ceiling_height)
            #print("Sector lines:", [(l.x1, l.y1, l.x2, l.y2, l.is_blocking) for l in s.lines])

            # Plot sector on map
            for l in s.lines:
                if l.is_blocking:
                    plt.plot([l.x1, l.x2], [l.y1, l.y2], color='black', linewidth=2)

        # Show map
        plt.show()

        print("Episode finished!")

    # It will be done automatically anyway but sometimes you need to do it in the middle of the program...
    game.close()