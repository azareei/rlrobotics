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

        # Define sequence
        self.sequence = 'B'

        # Create first block
        _l = 4/100
        _d = 1/100
        _w = 6/100
        _h = 6/100
        _center = Coordinate(x=0, y=_d - (_h / 2))
        self.block_bot = Block(_w, _h, _center, _d, (0, 0, 0))

        # Create mid block
        _center = Coordinate(x=0, y=_l - _d + (_h / 2))
        self.block_mid = Block(_w, _h, _center, _d, (255, 0, 0))

        # Create top block
        _center = Coordinate(x=0, y=self.block_mid.get_anchor(type="t").y + _l - _d + (_h/2))
        self.block_top = Block(_w, _h, _center, _d, (0, 255, 0))

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

        self.prev_ui = None
        self.init_position()

    def compute_leg_height(self, A, B):
        legs_length = 5 / 100

        tmp = legs_length**2 - ((B.x - A.x) / 2)**2
        return Coordinate(x=(A.x + B.x)/2, y=np.sqrt(tmp))

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

        # Variables used to motion
        self.x_offset = self.block_top.center.x
        self.d_top = np.sin(self.theta_s_top) * self.bars_top.length * 2
        self.d_bot = np.sin(self.theta_s_bot) * self.bars_bot.length * 2

    def update_position(self, u_i, forward=True):
        """
        u_i is the delta between the previous position et the one now.
        """
        if self.sequence == 'A':
            self.update_seq_A(u_i, forward)
        elif self.sequence == 'B':
            self.update_seq_B(u_i, forward)

    def update_seq_A(self, u_i, forward):
        position = u_i + self.x_offset
        length = self.bars_bot.length
        if forward:
            if (u_i >= 0) and (u_i < self.d_bot):
                print("Case 1")
                dh = position

                _dh = dh - self.block_top.center.x
                new_anchor_x_pos = self.bars_bot.high_anchor.x + _dh
                dist = new_anchor_x_pos - self.bars_bot.low_anchor.x
                self.theta_i_bot = np.arcsin(dist/length)
                dv = np.cos(self.theta_i_bot) * length
                self.block_mid.set_position(
                    _x=self.block_mid.center.x + _dh,
                    _y=dv + (self.block_mid.height/2) - self.block_mid.anchor_d
                )
                self.bars_bot.low_anchor = self.block_bot.get_anchor(type='t')
                self.bars_bot.high_anchor = self.block_mid.get_anchor(type='b')
                self.spring_bot.Q.x = self.block_mid.center.x
                self.spring_bot.Q.y = self.bars_bot.high_anchor.y

                # Move top block 
                # Ensure theta_i is min
                self.theta_i_top = -self.theta_s_top
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
            if (u_i >= self.d_bot) and (u_i <= (self.d_top+self.d_bot)):
                print("Case 2")
                # Still need to ensure the bottom block is theta_s
                dh = position

                _dh = dh - self.block_top.center.x
                new_anchor_x_pos = self.bars_top.high_anchor.x + _dh
                dist = new_anchor_x_pos - self.bars_top.low_anchor.x
                self.theta_i_top = np.arcsin(dist / length)
                dv = np.cos(self.theta_i_top) * length
                self.block_top.set_position(
                    _x=dh,
                    _y=self.block_mid.get_anchor(type='t').y + (self.block_top.height/2) - self.block_top.anchor_d + dv
                )
                self.bars_top.low_anchor = self.block_mid.get_anchor(type='t')
                self.bars_top.high_anchor = self.block_top.get_anchor(type='b')
                self.spring_top.Q.x = self.block_top.center.x
                self.spring_top.Q.y = self.bars_top.high_anchor.y
                self.spring_top.P.x = self.block_mid.center.x
                self.spring_top.P.y = self.block_mid.get_anchor(type='t').y
        if forward is False:
            if (u_i <= (self.d_top + self.d_bot)) and (u_i > self.d_top):
                print("Case 3")
                dh = position

                _dh = dh - self.block_top.center.x
                new_anchor_x_pos = self.bars_bot.high_anchor.x + _dh
                dist = new_anchor_x_pos - self.bars_bot.low_anchor.x
                if dist >= length:
                    print("ERROR case 3")
                self.theta_i_bot = np.arcsin(dist/length)
                dv = np.cos(self.theta_i_bot) * length
                self.block_mid.set_position(
                    _x=self.block_mid.center.x + _dh,
                    _y=dv + (self.block_mid.height/2) - self.block_mid.anchor_d
                )
                self.bars_bot.low_anchor = self.block_bot.get_anchor(type='t')
                self.bars_bot.high_anchor = self.block_mid.get_anchor(type='b')
                self.spring_bot.Q.x = self.block_mid.center.x
                self.spring_bot.Q.y = self.bars_bot.high_anchor.y

                # Move top block 
                # Ensure theta_i is max
                self.theta_i_top = self.theta_s_top
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
            if (u_i >= 0) and (u_i <= self.d_top):
                print("Case 4")
                # Still need to ensure the bottom block is -theta_s
                dh = position

                _dh = dh - self.block_top.center.x
                new_anchor_x_pos = self.bars_top.high_anchor.x + _dh
                dist = new_anchor_x_pos - self.bars_top.low_anchor.x
                self.theta_i_top = np.arcsin(dist / length)
                dv = np.cos(self.theta_i_top) * length
                self.block_top.set_position(
                    _x=dh,
                    _y=self.block_mid.get_anchor(type='t').y + (self.block_top.height/2) - self.block_top.anchor_d + dv
                )
                self.bars_top.low_anchor = self.block_mid.get_anchor(type='t')
                self.bars_top.high_anchor = self.block_top.get_anchor(type='b')
                self.spring_top.Q.x = self.block_top.center.x
                self.spring_top.Q.y = self.bars_top.high_anchor.y
                self.spring_top.P.x = self.block_mid.center.x
                self.spring_top.P.y = self.block_mid.get_anchor(type='t').y

    def update_seq_B(self, u_i, forward):
        """
        Top block is moving before bottom block
        """
        position = u_i + self.x_offset
        length = self.bars_bot.length
        if forward:
            if (u_i >= 0) and (u_i < self.d_top):
                print("Case 1")
                dh = position

                _dh = dh - self.block_top.center.x
                new_anchor_x_pos = self.bars_top.high_anchor.x + _dh
                dist = new_anchor_x_pos - self.bars_top.low_anchor.x
                self.theta_i_top = np.arcsin(dist / length)
                dv = np.cos(self.theta_i_top) * length
                self.block_top.set_position(
                    _x=dh,
                    _y=self.block_mid.get_anchor(type='t').y + (self.block_top.height/2) - self.block_top.anchor_d + dv
                )
                self.bars_top.low_anchor = self.block_mid.get_anchor(type='t')
                self.bars_top.high_anchor = self.block_top.get_anchor(type='b')
                self.spring_top.Q.x = self.block_top.center.x
                self.spring_top.Q.y = self.bars_top.high_anchor.y
                self.spring_top.P.x = self.block_mid.center.x
                self.spring_top.P.y = self.block_mid.get_anchor(type='t').y
            if (u_i >= self.d_top) and (u_i <= (self.d_top+self.d_bot)):
                print("Case 2")
                dh = position

                _dh = dh - self.block_top.center.x
                new_anchor_x_pos = self.bars_bot.high_anchor.x + _dh
                dist = new_anchor_x_pos - self.bars_bot.low_anchor.x
                self.theta_i_bot = np.arcsin(dist/length)
                dv = np.cos(self.theta_i_bot) * length
                self.block_mid.set_position(
                    _x=self.block_mid.center.x + _dh,
                    _y=dv + (self.block_mid.height/2) - self.block_mid.anchor_d
                )
                self.bars_bot.low_anchor = self.block_bot.get_anchor(type='t')
                self.bars_bot.high_anchor = self.block_mid.get_anchor(type='b')
                self.spring_bot.Q.x = self.block_mid.center.x
                self.spring_bot.Q.y = self.bars_bot.high_anchor.y

                # Move top block 
                # Ensure theta_i is maxed
                self.theta_i_top = self.theta_s_top
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
        if forward is False:
            if (u_i <= (self.d_top + self.d_bot)) and (u_i > self.d_bot):
                print("Case 3")
                # Copy from above
                dh = position

                _dh = dh - self.block_top.center.x
                new_anchor_x_pos = self.bars_top.high_anchor.x + _dh
                dist = new_anchor_x_pos - self.bars_top.low_anchor.x
                self.theta_i_top = np.arcsin(dist / length)
                dv = np.cos(self.theta_i_top) * length
                self.block_top.set_position(
                    _x=dh,
                    _y=self.block_mid.get_anchor(type='t').y + (self.block_top.height/2) - self.block_top.anchor_d + dv
                )
                self.bars_top.low_anchor = self.block_mid.get_anchor(type='t')
                self.bars_top.high_anchor = self.block_top.get_anchor(type='b')
                self.spring_top.Q.x = self.block_top.center.x
                self.spring_top.Q.y = self.bars_top.high_anchor.y
                self.spring_top.P.x = self.block_mid.center.x
                self.spring_top.P.y = self.block_mid.get_anchor(type='t').y
            if (u_i <= self.d_bot) and (u_i >= 0):
                print("Case 4")
                # Ensure theta_s is min out
                self.theta_i_top = -self.theta_s_top
                dh = position

                _dh = dh - self.block_top.center.x
                new_anchor_x_pos = self.bars_bot.high_anchor.x + _dh
                dist = new_anchor_x_pos - self.bars_bot.low_anchor.x
                self.theta_i_bot = np.arcsin(dist/length)
                dv = np.cos(self.theta_i_bot) * length
                self.block_mid.set_position(
                    _x=self.block_mid.center.x + _dh,
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