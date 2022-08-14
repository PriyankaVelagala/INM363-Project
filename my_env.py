from argparse import ArgumentParser
from pathlib import Path
from time import sleep
import vizdoom as vzd


CONFIG_DIR = "C:\\Users\\priya\\OneDrive - City, University of London\\Documents\\GitHub\\INM363-Project\\scenarios\\"
CONFIG_FILE = "custom\\very_dense_reward.cfg"
SCENARIO_FILE = "custom\\very_dense_reward.wad"
DEBUG = False

if __name__ == "__main__":

    #Set up game configuration
    config_path = Path(CONFIG_DIR, CONFIG_FILE)
    scenario_path = Path(CONFIG_DIR, SCENARIO_FILE)
    if DEBUG:
        print(config_path)
        print(scenario_path)

    parser = ArgumentParser()
    parser.add_argument(dest="config",
                        default=config_path.as_posix(),
                        nargs="?")
    args = parser.parse_args()
    print(args)

    #Initialize game instance with configuration
    game = vzd.DoomGame()
    game.load_config(args.config)
    game.set_doom_scenario_path(scenario_path.as_posix())
    game.set_mode(vzd.Mode.SPECTATOR)
    game.init()

    episodes = 1

    for i in range(episodes):
        print("Episode #" + str(i + 1))
        game.new_episode()
        last_total_shaping_reward = 0
        j = 0
        while not game.is_episode_finished():
            state = game.get_state()
            game.advance_action()
            last_action = game.get_last_action()
            reward = game.get_last_reward()

            # Retrieve the shaping reward
            fixed_shaping_reward = game.get_game_variable(vzd.GameVariable.USER1)  # Get value of scripted variable
            shaping_reward = vzd.doom_fixed_to_double(fixed_shaping_reward)  # If value is in DoomFixed format project it to double
            shaping_reward = shaping_reward - last_total_shaping_reward
            last_total_shaping_reward += shaping_reward

            if j % 10 == 0:
                print("State #" + str(state.number))
                #print(f"X Position: {state.game_variables[0]}, Y Position: {state.game_variables[1]}")
                print(f"Item Count: {state.game_variables[2]}")
                print(f"Health: {state.game_variables[3]}")
                print("Action:", last_action)
                print("Reward:", reward)
                print("Shaping Reward:", shaping_reward )
                print("=====================")
            j += 1

        print("Episode finished!")
        print("Total reward:", game.get_total_reward()+last_total_shaping_reward)
        print("************************")
        sleep(2.0)

    game.close()







