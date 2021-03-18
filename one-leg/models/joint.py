from models.block import Block
from models.bar import Bar
from models.spring import Spring
from coordinates import Coordinate
import numpy as np


class Joint:
    """
    Represent a joint constituted by two blocs linked with two arms and a spring.
    """
    def __init__(self):
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

        # Create the spring
        self.spring = Spring(
            _P=Coordinate(x=0, y=0),
            _Q=Coordinate(x=0, y=_l)
        )

        # Compute Theta_s - limits of the angle for the bar.
        self.theta_s = np.arccos(2 * self.block_bot.anchor_d / self.bars.length)

        self.theta_i = 0

    def update_position(self, h_diff):
        """
        This apply only to the top block, as the bot block is fixed
        x_i is a one dimension displacement along x axis.
        """
        length = self.bars.length
        theta_i = np.arcsin(x_i / length)
        d_theta = theta_i - self.theta_i
        v_diff = length * np.cos(theta_i)
        self.block_top.set_position(x_i, v_height)
        self.bars.low_anchor = self.block_bot.get_anchor(type="t")
        self.bars.high_anchor = self.block_top.get_anchor(type="b")
        self.theta_i = theta_i
