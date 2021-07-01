import matplotlib.pyplot as plt
import numpy as np

k = 1.
l0 = 1.
ri = 1.5
P = [
    [0., -1.],
    [-0.2, -1.],
    [-1, -1.]
]

theta = np.radians(range(-45, 46))


def energy(angle, position):
    s = ri * np.sin(angle) - position[0]
    c = ri * np.cos(angle) - position[1]
    sq = np.sqrt(s**2 + c**2)
    return k / 2 * (sq - l0**2)


energies = []
for p in P:
    subset = []
    for t in theta:
        subset.append(energy(t, p))
    energies.append(subset)

plt.figure()
plt.plot(np.degrees(theta), energies[0], label='center')
plt.plot(np.degrees(theta), energies[1], label='slightly left')
plt.plot(np.degrees(theta), energies[2], label='left')
plt.legend()
plt.grid()
plt.xlabel('Theta angle [deg]')
plt.ylabel('Energy normalized')
plt.title('Energy by anchor position')
plt.savefig('energy.png')
