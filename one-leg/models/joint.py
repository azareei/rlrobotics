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

    def update_position(self, u_i):
        """
        This apply only to the top block, as the bot block is fixed
        x_i is a one dimension displacement along x axis.
        """
        length = self.bars.length
        if u_i > length:
            return
        theta_i = np.arcsin(u_i / length)
        if np.abs(theta_i) > self.theta_s:
            return
        v_diff = length * np.cos(theta_i)
        self.block_top.set_position(u_i, v_diff + (self.block_top.height / 2) - self.block_top.anchor_d)
        self.bars.low_anchor = self.block_bot.get_anchor(type="t")
        self.bars.high_anchor = self.block_top.get_anchor(type="b")
        self.spring.Q.x = self.block_top.center.x
        self.spring.Q.y = self.bars.high_anchor.y
        self.theta_i = theta_i
