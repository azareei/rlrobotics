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
    # execute only if run as a script
    sim = initialize_env()
    sim.simulate()
