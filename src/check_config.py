from argparse import ArgumentParser
from pathlib import Path
from time import sleep
import vizdoom as vzd

CONFIG_DIR = "C:\\Users\\priya\\OneDrive - City, University of London\\Documents\\GitHub\\INM363-Project\\scenarios\\"
CONFIG_FILE = "custom\\config.cfg"
SCENARIO_FILE = "default\\my_way_home.wad"

if __name__ == "__main__":

    #Set up game configuration
    config_path = Path(CONFIG_DIR, CONFIG_FILE)
    scenario_path = Path(CONFIG_DIR, SCENARIO_FILE)
    print(config_path)
    print(scenario_path)

    parser = ArgumentParser()
    parser.add_argument(dest="config",
                        default=CONFIG_FILE)
    args = parser.parse_args()
    print(args)

    #Initialize game instance with configuration
    game = vzd.DoomGame()
    game.load_config(args.config)
    game.set_doom_scenario_path(scenario_path)
    game.set_mode(vzd.Mode.SPECTATOR)
    game.init()

    episodes = 2

    for i in range(episodes):
        print("Episode #" + str(i + 1))
        game.new_episode()
        j = 0
        while not game.is_episode_finished():
            state = game.get_state()
            game.advance_action()
            last_action = game.get_last_action()
            reward = game.get_last_reward()

            if j % 5 == 0:
                print("State #" + str(state.number))
                print("Game variables: ", state.game_variables)
                print("Action:", last_action)
                print("Reward:", reward)
                print("=====================")
            j += 1

        print("Episode finished!")
        print("Total reward:", game.get_total_reward())
        print("************************")
        sleep(2.0)

    game.close()







