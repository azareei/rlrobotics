from models.joint import Joint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Simulation:
    fig = None

    def __init__(self):
        self.joint = Joint()
        # input movement
        self.x = np.linspace(-3.46 / 100, 3.46 / 100, num=10)

    def simulate(self):
        self.draw()
        # prev = self.x[0]
        for x_i in self.x:
            self.joint.update_position(x_i)
            # prev = x_i
            self.draw()

    def draw(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(-0.10, 0.10)
        self.ax.set_ylim(-0.10, 0.10)

        # Draw blocks
        self.joint.block_bot.draw(self.ax)
        self.joint.block_mid.draw(self.ax)
        self.joint.block_top.draw(self.ax)

        # Draw bars
        self.joint.bars_bot.draw(self.ax)
        self.joint.bars_top.draw(self.ax)

        # Draw spring
        self.joint.spring_bot.draw(self.ax)
        self.joint.spring_top.draw(self.ax)

        self.ax.grid()
        plt.show()
