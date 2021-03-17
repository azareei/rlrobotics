from models.joint import Joint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Simulation:
    fig = None

    def __init__(self):
        self.joint = Joint()
        # input movement
        self.x = np.linspace(0, 3/100, num=5)

    def simulate(self):
        self.draw()
        prev = self.x[0]
        for x_i in self.x:
            self.joint.update_position(x_i-prev)
            prev = x_i
            self.draw()

    def draw(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(-0.25, 0.25)
        self.ax.set_ylim(-0.25, 0.25)
        self.ax.add_patch(self.joint.block_bot.draw())
        self.ax.add_patch(self.joint.block_top.draw())
        self.ax.plot((self.joint.P.x, self.joint.Q.x), (self.joint.P.y, self.joint.Q.y), 'r')
        self.ax.grid()
        plt.show()
