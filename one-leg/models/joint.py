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
        #   length of the bars_bot are 4cm
        #   width is 6cm
        #   height is 6 cm

        # Create first block
        _l = 4/100
        _d = 1/100
        _w = 6/100
        _h = 6/100
        _center = Coordinate(x=0, y=_d - (_h / 2))
        self.block_bot = Block(_w, _h, _center, _d)

        # Create mid block
        _center = Coordinate(x=0, y=_l - _d + (_h / 2))
        self.block_mid = Block(_w, _h, _center, _d)

        # Create top block
        _center = Coordinate(x=0, y=self.block_mid.get_anchor(type="t").y + _l - _d + (_h/2))
        self.block_top = Block(_w, _h, _center, _d)

        # Create the bars_bot
        self.bars_bot = Bar(
            self.block_bot.get_anchor(type="t"),
            self.block_mid.get_anchor(type="b"),
            _l,
            self.block_bot.get_anchor_distance()
        )

        # Create the bars_top
        self.bars_top = Bar(
            self.block_mid.get_anchor(type='t'),
            self.block_top.get_anchor(type='b'),
            _l,
            self.block_mid.get_anchor_distance()
        )

        # Create the spring_bot
        self.spring_bot = Spring(
            _P=Coordinate(x=0, y=self.block_bot.get_anchor(type='t').y),
            _Q=Coordinate(x=0, y=self.block_mid.get_anchor(type='b').y)
        )

        # Create the spring_top
        self.spring_top = Spring(
            _P=Coordinate(x=0, y=self.block_mid.get_anchor(type='t').y),
            _Q=Coordinate(x=0, y=self.block_top.get_anchor(type='b').y)
        )

        # Compute Theta_s - limits of the angle for the bar.
        self.theta_s_bot = np.arccos(2 * self.block_bot.anchor_d / self.bars_bot.length)
        self.theta_s_top = np.arccos(2 * self.block_mid.anchor_d / self.bars_top.length)

        self.theta_i_bot = 0
        self.theta_i_top = 0

        # Compute Legs endpoint
        self.compute_leg_height()

        self.prev_ui = None
        self.init_position()

    def compute_leg_height(self):
        legs_length = 5 / 100
        tmp = legs_length**2 - ((self.block_mid.center - self.block_top.center).norm(order=2) / 2)**2
        self.leg_position = np.sqrt(tmp)

    def init_position(self):
        self.theta_i_top = -self.theta_s_top
        self.theta_i_bot = -self.theta_s_bot

        # First move mid block
        dh = np.sin(self.theta_i_bot) * self.bars_bot.length
        dv = np.cos(self.theta_i_bot) * self.bars_bot.length

        self.block_mid.set_position(
            _x=dh,
            _y=dv + (self.block_mid.height/2) - self.block_mid.anchor_d
        )
        self.bars_bot.low_anchor = self.block_bot.get_anchor(type='t')
        self.bars_bot.high_anchor = self.block_mid.get_anchor(type='b')
        self.spring_bot.Q.x = self.block_mid.center.x
        self.spring_bot.Q.y = self.bars_bot.high_anchor.y

        # Move top block 
        dh = np.sin(self.theta_i_top) * self.bars_top.length
        dv = np.cos(self.theta_i_top) * self.bars_top.length

        self.block_top.set_position(
            _x=self.block_mid.center.x + dh,
            _y=self.block_mid.get_anchor(type='t').y + (self.block_top.height/2) - self.block_top.anchor_d + dv
        )
        self.bars_top.low_anchor = self.block_mid.get_anchor(type='t')
        self.bars_top.high_anchor = self.block_top.get_anchor(type='b')
        self.spring_top.Q.x = self.block_top.center.x
        self.spring_top.Q.y = self.bars_top.high_anchor.y
        self.spring_top.P.x = self.block_mid.center.x
        self.spring_top.P.y = self.block_mid.get_anchor(type='t').y


    def update_position(self, u_i):
        """
        u_i is the delta between the previous position et the one now.
        """
        length = self.bars_bot.length
        if u_i > 2 * length:
            return
        theta_i = np.arcsin(u_i / length)
        if np.abs(theta_i) > self.theta_s:
            return

        top_block_first = True
        if top_block_first: # use the formula i wrote

        self.theta_i_top = np.arcsin(u_i)
        self.theta_i_bot = -self.theta_s_bot

        # First move mid block
        dh = np.sin(self.theta_i_bot) * self.bars_bot.length
        dv = np.cos(self.theta_i_bot) * self.bars_bot.length

        self.block_mid.set_position(
            _x=dh,
            _y=dv + (self.block_mid.height/2) - self.block_mid.anchor_d
        )
        self.bars_bot.low_anchor = self.block_bot.get_anchor(type='t')
        self.bars_bot.high_anchor = self.block_mid.get_anchor(type='b')
        self.spring_bot.Q.x = self.block_mid.center.x
        self.spring_bot.Q.y = self.bars_bot.high_anchor.y

        # Move top block 
        dh = np.sin(self.theta_i_top) * self.bars_top.length
        dv = np.cos(self.theta_i_top) * self.bars_top.length

        self.block_top.set_position(
            _x=self.block_mid.center.x + dh,
            _y=self.block_mid.get_anchor(type='t').y + (self.block_top.height/2) - self.block_top.anchor_d + dv
        )
        self.bars_top.low_anchor = self.block_mid.get_anchor(type='t')
        self.bars_top.high_anchor = self.block_top.get_anchor(type='b')
        self.spring_top.Q.x = self.block_top.center.x
        self.spring_top.Q.y = self.bars_top.high_anchor.y
        self.spring_top.P.x = self.block_mid.center.x
        self.spring_top.P.y = self.block_mid.get_anchor(type='t').y
