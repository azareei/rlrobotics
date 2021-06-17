"""
This script is used to produce all the possible sequences of the robot motion and add it to a table
"""

from main import initialize_env
import time
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


SIMULATE = True

if SIMULATE:
    sequences = ['A', 'B']#, 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    theoretical_sequences = ['K', 'L', 'M', 'N', 'O']   # not used

    actuation = [0, 180]
    reverse_actuation = [False, True]

    start = time.time()

    results = []

    for act in actuation:
        for s1 in sequences:
            for s2 in sequences:
                for s3 in sequences:
                    for s4 in sequences:
                        for rev in reverse_actuation:
                            seq = f'{s1}{s2}{s3}{s4}'
                            sim = initialize_env(seq, act, rev)
                            sim.mapping = True

                            x, y, yaw = sim.simulate()
                            results.append([seq, act, x, y, yaw])

    print(f'Simulation time : {(time.time() - start) / 60:.0f} minutes {(time.time() - start) % 60:.0f} secondes')

    df = pd.DataFrame(results, columns=['sequence', 'actuation', 'x', 'y', 'yaw'])

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
fig = plt.figure()
ax = plt.axes(projection='3d')


ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
ax.set_zlabel('yaw[rad]')

scatter = ax.scatter3D(x, y, yaw, c=act, cmap='rainbow')

ax.legend(*scatter.legend_elements(),
          loc="upper left", title="Phase")

plt.savefig('{0}/results/_3D.png'.format(
    Path(__file__).resolve().parent,))

# 2D Plots
fig = plt.figure()
ax = plt.axes()
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')

scatter = ax.scatter(x, y, c=act, cmap='rainbow')
ax.legend(*scatter.legend_elements(),
          loc='upper left', title='Phase')
ax.title.set_text('Displacement x/y')
plt.savefig('{0}/results/_x_y.png'.format(
    Path(__file__).resolve().parent,))

fig = plt.figure()
ax = plt.axes()
ax.set_xlabel('x[m]')
ax.set_ylabel('yaw[m]')

scatter = ax.scatter(x, yaw, c=act, cmap='rainbow')
ax.legend(*scatter.legend_elements(),
          loc='upper left', title='Phase')
ax.title.set_text('Displacement x/yaw')
plt.savefig('{0}/results/_x_yaw.png'.format(
    Path(__file__).resolve().parent,))

fig = plt.figure()
ax = plt.axes()
ax.set_xlabel('y[m]')
ax.set_ylabel('yaw[m]')

scatter = ax.scatter(y, yaw, c=act, cmap='rainbow')
ax.legend(*scatter.legend_elements(),
          loc='upper left', title='Phase')
ax.title.set_text('Displacement y/yaw')
plt.savefig('{0}/results/_y_yaw.png'.format(
    Path(__file__).resolve().parent,))





