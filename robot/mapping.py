"""
Module mapping.py
This script creates a table containing different sequences and their respective motion in x, y, heading
with respect to a specific actuation.

By default it will use only the realistic sequences but a boolean can make use also of the theoretical
sequences if needed.

At the end of the simulation, results are produced and especially 4 plots are saved representing the
repartition of the points.

Attributes
----------
REALISTIC_SEQUENCES : list
    represent our 10 realistic sequences that are reachable.
THEORETICAL_SEQUENCES : list
    represent 5 different sequences that should not be possible in reality even though we have observed in
    Vladimir work that it happens sometimes. Maily they involve to have both blocks moving at the same time.
ACTUATION_PHASE: list
    Correspond to the list of phase difference we want to play with. Actually supports only 0 and 180 degrees.
REVERSE_ACTUATION : list
    Correspond to the list of boolean to reverse the actuation. Needed if we want to have symmetric results.
    Disable it if speed is important.
results : list
    Will store the results of the simulation. The results contains the sequence,
    the actuation, the reverse actuation used and the final position of the robot
    in x, y and yaw.
SIMULATE : bool
    Tells if the simulation will take place, if not the case, the script will use the pre-generated data to create
    the 3D plots
SIMULATE_THEORY_SEQ : bool
    If true, not only the realistic sequences will be tested but also the theoretical ones.
STEPS : int
    Represent the number of step used for a simulation.

"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from main import initialize_env

SIMULATE = True
SIMULATE_THEORY_SEQ = False
REALISTIC_SEQUENCES = ['A', 'C', 'E', 'G', 'I', 'B', 'D', 'F', 'H', 'J']
THEORETICAL_SEQUENCES = ['K', 'L', 'M', 'N', 'O']   # not used
ACTUATION_PHASE = [0, 180]
REVERSE_ACTUATION = [False, True]
STEPS = 10

results = []

if SIMULATE:
    if SIMULATE_THEORY_SEQ:
        sequences = REALISTIC_SEQUENCES + THEORETICAL_SEQUENCES
    else:
        sequences = REALISTIC_SEQUENCES

    start = time.time()
    for s1 in sequences:
        for s2 in sequences:
            for s3 in sequences:
                for s4 in sequences:
                    for act in ACTUATION_PHASE:
                        for rev in REVERSE_ACTUATION:
                            seq = f'{s1}{s2}{s3}{s4}'
                            sim = initialize_env(seq, act, rev, STEPS)
                            sim.mapping = True

                            x, y, yaw = sim.simulate()
                            results.append([seq, act, rev, x, y, yaw])

    print(f'Simulation time : {(time.time() - start) / 60:.0f} minutes {(time.time() - start) % 60:.0f} secondes')

    df = pd.DataFrame(results, columns=['sequence', 'actuation', 'reverse', 'x', 'y', 'yaw'])

    df.to_pickle('{0}/results/_all_sequences.pkl'.format(
        Path(__file__).resolve().parent
    ))

    df.to_csv('{0}/results/_all_sequences.csv'.format(
        Path(__file__).resolve().parent
    ))

df = pd.read_pickle('{0}/results/_all_sequences.pkl'.format(
        Path(__file__).resolve().parent
    ))

x = df['x']
y = df['y']
yaw = df['yaw']
act = df['actuation']

# 3D Plot scatter
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection='3d')


ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
ax.set_zlabel('yaw[deg]')

scatter = ax.scatter3D(x, y, np.degrees(yaw), c=act, cmap='rainbow', s=1.5)

ax.legend(*scatter.legend_elements(),
          loc="upper left", title="Phase")

plt.savefig('{0}/results/_3D.png'.format(
    Path(__file__).resolve().parent,))

# 2D Plots
fig = plt.figure()
ax = plt.axes()
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')

scatter = ax.scatter(x, y, c=act, cmap='rainbow', s=1.5)
ax.legend(*scatter.legend_elements(),
          loc='upper left', title='Phase')
ax.title.set_text('Displacement x/y')
plt.savefig('{0}/results/_x_y.png'.format(
    Path(__file__).resolve().parent,))

fig = plt.figure()
ax = plt.axes()
ax.set_xlabel('x[m]')
ax.set_ylabel('yaw[deg]')

scatter = ax.scatter(x, np.degrees(yaw), c=act, cmap='rainbow', s=1.5)
ax.legend(*scatter.legend_elements(),
          loc='upper left', title='Phase')
ax.title.set_text('Displacement x/yaw')
plt.savefig('{0}/results/_x_yaw.png'.format(
    Path(__file__).resolve().parent,))

fig = plt.figure()
ax = plt.axes()
ax.set_xlabel('y[m]')
ax.set_ylabel('yaw[deg]')

scatter = ax.scatter(y, np.degrees(yaw), c=act, cmap='rainbow', s=1.5)
ax.legend(*scatter.legend_elements(),
          loc='upper left', title='Phase')
ax.title.set_text('Displacement y/yaw')
plt.savefig('{0}/results/_y_yaw.png'.format(
    Path(__file__).resolve().parent,))
