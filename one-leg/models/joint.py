from models.block import Block
from models.bar import Bar
from coordinates import Coordinate
import numpy as np


class Joint:
    """
    Represent a joint constituted by two blocs linked with two arms and a spring.
    """
    def __init__(self):
        self.r = 10/100  # need to check that r is bigger than 2*self.d
        self.d = 1/100  # distance between ancher of joint and the side of the block
        # We define the following for now:
        #   anchor distance to side of the block is 1cm
        #   length of the bars are 4cm
        #   width is 6cm
        #   height is 6 cm

        # Create first block
        _l = 4/100
        _d = 1/100
        _w = 6/100
        _h = 6/100
        _center = Coordinate(x=0, y=_d - (_h / 2))
        self.block_bot = Block(_w, _h, _center, _d)

        # Create top block
        _center = Coordinate(x=0, y=_l - _d + (_h / 2))
        self.block_top = Block(_w, _h, _center, _d)

        # Create the bars
        self.bars = Bar(
            self.block_bot.get_anchor(type="t"),
            self.block_top.get_anchor(type="b"),
            _l,
            self.block_bot.get_anchor_distance())

        # Compute Theta_s - limits of the angle for the bar.
        self.theta_s = np.arccos(2 * self.block_bot.anchor_d / self.bars.length)

        self.theta_i = 0

    def update_position(self, x_i):
        """
        This apply only to the top block, as the bot block is fixed
        x_i is a one dimension displacement along x axis.
        """
        self.theta_i = np.arcsin(x_i/self.r)
        v_diff = self.r-(self.r * np.cos(self.theta_i))
        self.block_top.update_position(x_i, v_diff)
        self.Q = self.Q + Coordinate(x=x_i, y=v_diff)
        print((self.P-self.Q).norm(order=2))
