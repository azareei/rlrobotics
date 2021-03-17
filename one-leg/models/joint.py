from models.block import Block
from coordinates import Coordinate
import numpy as np


class Joint:
    """
    Represent a joint constituted by two blocs linked with two arms and a spring.
    """
    def __init__(self):
        self.r = 3/100  # need to check that r is bigger than 2*self.d
        self.k = 1 / 0.05
        self.d = 1/100  # distance between ancher of joint and the side of the block
        self.block_top = Block(5/100, 5/100, _center=Coordinate(x=0, y=self.r-self.d+(2.5/100)))
        self.block_bot = Block(5/100, 5/100, _center=Coordinate(x=0, y=self.d-(2.5/100)))
        self.theta_s = np.arccos((self.d*2) / self.r)
        
        self.theta_i = 0 
        self.P = Coordinate(x=0, y=0)
        self.Q = Coordinate(x=0, y=self.r)
        #self.block_top.update_position(self.r*np.sin(self.theta_i), self.r*np.cos(self.theta_i))
        

    def update_position(self, x_i):
        """
        This apply only to the top block, as the bot block is fixed
        x_i is a one dimension displacement along x axis.
        """
        self.theta_i = np.arcsin(x_i)

