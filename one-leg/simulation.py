from models.joint import Joint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from coordinates import Coordinate

class Simulation:
    fig = None
    t = 0
    A = []
    B = []
    C = []

    def __init__(self):
        self.joint = Joint()
        # input movement

    def simulate(self):
        steps = 100
        self.x = np.linspace(0, 3.46 / 100 * 4, num=steps)
        for x_i in self.x:
            self.joint.update_position(x_i, True)
            offset = 4 / 100
            _A = Coordinate(x=self.joint.block_top.center.x - offset, y=0)
            _B = Coordinate(x=self.joint.block_mid.center.x, y=0)
            self.A.append(_A)
            self.B.append(_B)
            self.C.append(self.joint.compute_leg_height(_A, _B))
            self.draw()

        self.x = np.linspace(3.46 / 100 * 4, 0, num=steps)
        for x_i in self.x:
            self.joint.update_position(x_i, False)
            offset = 4 / 100
            _A = Coordinate(x=self.joint.block_top.center.x - offset, y=0)
            _B = Coordinate(x=self.joint.block_mid.center.x, y=0)
            self.A.append(_A)
            self.B.append(_B)
            self.C.append(self.joint.compute_leg_height(_A, _B))
            self.draw()

        self.gen_leg_animation()

    def draw(self):
        
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(-0.20, 0.20)
        self.ax.set_ylim(-0.20, 0.20)

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
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.savefig('out_{}.png'.format(self.t))
        self.t += 1
        plt.close('all')
        #plt.show()


    def gen_leg_animation(self):
        
        self.t = 0
        for A, B, C in zip(self.A, self.B, self.C):
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.ax.set_xlim(-0.20, 0.20)
            self.ax.set_ylim(-0.20, 0.20)
            self.ax.invert_yaxis()
            self.ax.grid()
            _x = (A.x, C.x)
            _y = (A.y, C.y)
            self.ax.plot(_x, _y, 'g')
            _x = (C.x, B.x)
            _y = (C.y, B.y)
            self.ax.plot(_x, _y, 'b')
            plt.xlabel('x [m]')
            plt.ylabel('z [m]')
            plt.savefig('leg_{}.png'.format(self.t))
            plt.close('all')
            self.t += 1