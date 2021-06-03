from simulation import Simulation
import argparse
import json
from pathlib import Path

from utils import Utils

sim = None

parser = argparse.ArgumentParser(description='')
parser.add_argument('--config',
                    type=str,
                    default='config1.json',
                    help='Choose which file from config folder to load')
args = parser.parse_args()


def initialize_env():
    with open(f'{Path(__file__).resolve().parent}/config/{args.config}') as param_file:
        params = json.load(param_file)
    
    # Load defaults params
    with open(f'{Path(__file__).resolve().parent}/config/default.json') as param_file:
        default = json.load(param_file)

    Utils.dict_merge(params, default)

    global sim
    sim = Simulation(params)


def simulate():
    global sim
    sim.simulate()


if __name__ == "__main__":
    # execute only if run as a script
    initialize_env()
    simulate()
