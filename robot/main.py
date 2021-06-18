"""
Module main.py

This is the main file to initialize and generate the simulations. There is two ways to use it.
1)  Call it directly with a config file as option. For example :
    `python main.py --config Experience1.json` which will run the experiment 1 with its configs.
2)  Call initialize_env function from another python script with specific input. Used for testing
    multiple simulation at the same time, for example with mapping.py script


Attributes
----------
parser : ArgumentParser
    Attribute that handles the argument given when called in case 1)
args : Parser argument
    Contains the argument from command line. If called in case 2), will be None

Methods
-------
initialize_env(sequence='BBBB', phase=0,, reverse=False)
    Create a simulation environement with different parameters comming from the arguments or config
    files.

"""
import argparse
import json
from pathlib import Path

from simulation import Simulation
from utils import Utils

parser = argparse.ArgumentParser(description='')
parser.add_argument('--config',
                    type=str,
                    help='Choose which file from config folder to load')
args = parser.parse_args()


def initialize_env(sequence='BBBB', phase=0, reverse=False):
    """
    Entry point of the simulation, allows us to initialize a robot and a simulation
    environnement with a config file of with some basics parameters. 

    The function will use two configuration files. The first one is selected by the user.
    The second file is the default.json which contains all the necessary parameters that
    generally don't change such as the shape of the robot, the position of the legs etc...
    If a specific parameter is needed for a scenario you can add the key/value to your
    json file and it will override the default value.

    Parameters
    ----------
    sequence : str, optional
        string of 4 characters representing a specific sequence. Will be processed only if
         this module is not called directly but instead by another python program, for example
         mapping.py
    phase : int, optional
        represent the phase difference of the actuators. As for sequence, this parameter will
         only be processed if function called by another script.
    reverse : bool, optional
        represent a specific mode of the simulation where the actuation are reversed. It will
        be used only if there is no config file.

    Returns
    -------
    Simulation
        return a initialized simulation with the configurations
    """
    if args.config is not None:
        with open(f'{Path(__file__).resolve().parent}/config/{args.config}') as param_file:
            params = json.load(param_file)
    else:
        params = {
            "simulation": {
                "camera_robot_ref": True,
                "actuation": {
                    "steps": 50,  # TODO CHANGE
                    "cycles": 1,
                    "phase": phase,
                    "reverse": reverse
                },
                "draw": False
            },
            "robot": {
                "J1": {
                    "sequence": sequence[0]
                },
                "J2": {
                    "sequence": sequence[1]
                },
                "J3": {
                    "sequence": sequence[2]
                },
                "J4": {
                    "sequence": sequence[3]
                }
            }
        }
    # Load defaults params
    with open(f'{Path(__file__).resolve().parent}/config/default.json') as param_file:
        default = json.load(param_file)

    Utils.dict_merge(params, default)

    return Simulation(params)


if __name__ == "__main__":
    sim = initialize_env()
    sim.simulate()
