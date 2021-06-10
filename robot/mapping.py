"""
This script is used to produce all the possible sequences of the robot motion and add it to a table
"""

from main import initialize_env
import time
import pandas as pd
from pathlib import Path

sequences = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
theoretical_sequences = ['K', 'L', 'M', 'N', 'O']  # not used

actuation = [0, 180]

start = time.time()

results = []

for act in actuation:
    for s1 in sequences:
        for s2 in sequences:
            for s3 in sequences:
                for s4 in sequences:
                    seq = f'{s1}{s2}{s3}{s4}'
                    sim = initialize_env(seq, act)
                    sim.mapping = True

                    x, y, yaw = sim.simulate()
                    results.append([seq, act, x, y, yaw])

print(f'Simulation time : {(time.time() - start) / 60:.0f} minutes {(time.time() - start) % 60:.0f} secondes')


df = pd.DataFrame(results)

df.to_pickle('{0}/results/_all_sequences.pkl'.format(
    Path(__file__).resolve().parent
))

df.to_csv('{0}/results/_all_sequences.csv'.format(
    Path(__file__).resolve().parent
))
