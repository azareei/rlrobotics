from models.block import Block
from coordinates import Coordinate
import numpy as np


class Joint:
    """
    Represent a joint constituted by two blocs linked with two arms and a spring.
    """
    def __init__(self):
        self.r = 3/100  # need to check that r is bigger than 2*self.d
        self.P = Coordinate(x=0, y=0)
        self.k = 1 / 0.05
        self.d = 1/100  # distance between ancher of joint and the side of the block
        self.block_top = Block(5/100, 5/100, _center=Coordinate(x=0, y=self.r-self.d+(2.5/100)))
        self.block_bot = Block(5/100, 5/100, _center=Coordinate(x=0, y=self.d-(2.5/100)))
        self.theta_s = np.sqrt((self.d*2)**2 / self.r**2)
        self.theta_i = -self.theta_s  # starting condition from the left

    def update_position(self, x_i):
        """
        This apply only to the top block, as the bot block is fixed
        """

