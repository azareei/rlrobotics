from models.block import Block
from models.bar import Bar
from models.spring import Spring
from coordinates import Coordinate
import numpy as np
import cv2
from utils import Utils
from inspect import currentframe, getframeinfo


class Joint:
    """
    Represent a joint constituted by two blocs linked with two arms and a spring.
    """
    def __init__(self, _sequence, _structure_offset, _invert_y=False,
                 _invert_init_angle=False, _bot_color=(0, 0, 0),
                 _top_color=(255, 0, 0), _name='Joint',
                 _r1=3/100, _r2=3/100,
                 _theta1=0.785, _theta2=0.785,
                 _legs_length=5 / 100):
        # We define the following for now:
        #   anchor distance to side of the block is 1cm
        #   width is 6cm
        #   height is 6 cm

        # Define sequence
        self.sequence = _sequence
        self.structure_offset = _structure_offset
        self.invert_y = _invert_y
        self.bot_color = _bot_color
        self.top_color = _top_color
        self.invert_init_angle = _invert_init_angle
        self.name = _name

        # Create first block
        _d_bot = np.arccos(_theta1) * _r1 / 2
        _d_top = np.arccos(_theta2) * _r2 / 2
        _d_mid = 1 / 100
        _w = 5.5 / 100
        _h = 5.5 / 100

        _center = Coordinate(x=0, y=_d_bot - (_h / 2))
        self.block_bot = Block(
            _width=_w,
            _height=_h,
            _center=_center,
            _anchor_d=_d_bot,
            _color=self.bot_color,
            _type='bottom'
        )

        # Create mid block
        _center = Coordinate(x=0, y=_r1 - _d_mid + (_h / 2))
        self.block_mid = Block(
            _width=_w,
            _height=_h,
            _center=_center,
            _anchor_d=_d_mid,
            _color=Utils.black,
            _type='middle'
        )

        # Create top block
        _center = Coordinate(x=0, y=self.block_mid.get_anchor(type="t").y + _r2 - _d_top + (_h/2))
        self.block_top = Block(
            _width=_w,
            _height=_h,
            _center=_center,
            _anchor_d=_d_top,
            _color=self.top_color,
            _type='top'
        )

        # Create the bars_bot
        self.bars_bot = Bar(
            self.block_bot.get_anchor(type="t"),
            self.block_mid.get_anchor(type="b"),
            _r1,
            self.block_bot.get_anchor_distance()
        )

        # Create the bars_top
        self.bars_top = Bar(
            self.block_mid.get_anchor(type='t'),
            self.block_top.get_anchor(type='b'),
            _r2,
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

        self.legs_length = _legs_length

        self.A = []
        self.B = []
        self.C = []

        self.ground_distance = 0.0

        self.init_position()

    def compute_leg_height(self, _A, _B):
        tmp = self.legs_length**2 - ((_B.x - _A.x) / 2)**2
        return Coordinate(x=(_A.x + _B.x)/2, y=(_A.y + _B.y)/2, z=np.sqrt(tmp))

    def get_real_leg(self):
        """
            Return the real coordinate of C
        """
        if self.invert_y:
            inv = -1
        else:
            inv = 1

        c = self.C[-1]

        c = Coordinate(
            x=c.x + self.structure_offset.x,
            y=c.y * inv + self.structure_offset.y,
            z=c.z + self.structure_offset.z
        )
        return c

    def init_position(self):
        if self.invert_init_angle is False:
            self.theta_i_top = -self.theta_s_top
            self.theta_i_bot = -self.theta_s_bot
        else:
            self.theta_i_top = self.theta_s_top
            self.theta_i_bot = self.theta_s_bot

        self.move_mid_block(theta=self.theta_i_bot)
        self.move_top_block(theta=self.theta_i_top)

        # Variables used to motion
        self.x_offset = self.block_top.center.x
        self.d_top = np.sin(self.theta_s_top) * self.bars_top.length * 2
        self.d_bot = np.sin(self.theta_s_bot) * self.bars_bot.length * 2
        print(f'start on {self.block_top.center.x}')

    def update_position(self, u_i, forward):
        """
        u_i is the delta between the previous position et the one now.

        Return:
            Coordinate vector corresponding to the translation before and
            after the displacement of the legs.
        """
        # Call the right funcion for the joint sequence update_seq_G for example
        getattr(self, f'update_seq_{self.sequence}')(u_i, forward)
        return self.update_legs()

    def update_seq_A(self, u_i, forward):
        """
        Cyclic sequence where mid block always move first
        00 -> 10 -> 11 -> 01 -> 00
        """
        position = u_i + self.x_offset

        max_left = - (self.d_bot / 2) - (self.d_top / 2)
        max_right = (self.d_bot / 2) + (self.d_top / 2)

        if forward:
            if self.invert_init_angle:
                if (position >= (max_left + self.d_bot)) and (position <= max_right):
                    self.move_mid_block(position=position)
                    self.move_top_block(theta=self.theta_s_top)
                if (position >= max_left) and (position < (max_left + self.d_bot)):
                    self.move_mid_block(theta=-self.theta_s_top)
                    self.move_top_block(position=position)
            else:
                if (position >= max_left) and (position < (max_left + self.d_bot)):
                    self.move_mid_block(position=position)
                    self.move_top_block(theta=-self.theta_s_top)
                if (position >= (max_left + self.d_bot)) and (position <= max_right):
                    self.move_mid_block(theta=self.theta_s_top)
                    self.move_top_block(position=position)
        else:
            if self.invert_init_angle:
                if (position >= max_left) and (position <= (max_right - self.d_bot)):               
                    self.move_mid_block(position=position)
                    self.move_top_block(theta=-self.theta_s_top)
                if (position <= max_right) and (position > (max_right - self.d_bot)):
                    self.move_mid_block(theta=self.theta_s_top)
                    self.move_top_block(position=position)
            else:
                if (position <= max_right) and (position > (max_right - self.d_bot)):
                    self.move_mid_block(position=position)
                    self.move_top_block(theta=self.theta_s_top)
                if (position >= max_left) and (position <= (max_right - self.d_bot)):
                    self.move_mid_block(theta=-self.theta_s_top)
                    self.move_top_block(position=position)

    def update_seq_B(self, u_i, forward):
        """
        Cyclic sequence where top block always move first
        00 -> 01 -> 11 -> 10 -> 00
        """
        position = u_i + self.x_offset

        max_left = - (self.d_bot / 2) - (self.d_top / 2)
        max_right = (self.d_bot / 2) + (self.d_top / 2)

        if forward:
            if (position >= max_left) and (position < (max_left + self.d_top)):
                self.move_mid_block(theta=-self.theta_s_bot)
                self.move_top_block(position=position)
            if (position >= (max_left + self.d_top)) and (position <= max_right):
                self.move_mid_block(position=position)
                self.move_top_block(theta=self.theta_s_top)
        else:
            if (position <= max_right) and (position > (max_right - self.d_top)):
                self.move_mid_block(theta=self.theta_s_bot)
                self.move_top_block(position=position)
            if (position <= (max_right - self.d_bot)) and (position >= max_left):
                self.move_mid_block(position=position)
                self.move_top_block(theta=-self.theta_s_top)

    def update_seq_C(self, u_i, forward):
        """
        Cycle where first mid block move in forward pass, but top block move first
        in backward pass.
        00 -> 10 -> 11 -> 10 -> 00
        """
        position = u_i + self.x_offset

        max_left = - (self.d_bot / 2) - (self.d_top / 2)
        max_right = (self.d_bot / 2) + (self.d_top / 2)

        if forward:
            if (position >= max_left) and (position < (max_left + self.d_bot)):
                self.move_mid_block(position=position)
                self.move_top_block(theta=-self.theta_s_top)
            if (position >= (max_left + self.d_bot) and (position <= max_right)):
                self.move_mid_block(theta=self.theta_s_bot)
                self.move_top_block(position=position)
        else:
            if (position <= max_right) and (position > (max_right - self.d_top)):
                self.move_mid_block(theta=self.theta_s_bot)
                self.move_top_block(position=position)
            if (position <= (max_right - self.d_top)) and (position >= max_left):
                self.move_mid_block(position=position)
                self.move_top_block(theta=-self.theta_s_top)

    def update_seq_D(self, u_i, forward):
        """
        Cycle where first top block move in forward pass, but mid block move first
        in backward pass.
        00 -> 01 -> 11 -> 01 -> 00
        """
        position = u_i + self.x_offset

        max_left = - (self.d_bot / 2) - (self.d_top / 2)
        max_right = (self.d_bot / 2) + (self.d_top / 2)

        if forward:
            if self.invert_init_angle:
                if (position >= (max_left + self.d_top)) and (position <= max_right):
                    self.move_mid_block(theta=self.theta_s_bot)
                    self.move_top_block(position=position)
                if (position >= max_left) and (position < (max_left + self.d_top)):
                    self.move_mid_block(position=position)
                    self.move_top_block(theta=-self.theta_s_top)
            else:
                if (position >= max_left) and (position < (max_left + self.d_top)):
                    self.move_mid_block(theta=-self.theta_s_bot)
                    self.move_top_block(position=position)
                if (position >= (max_left + self.d_top)) and (position <= max_right):
                    self.move_mid_block(position=position)
                    self.move_top_block(theta=self.theta_s_top)
        else:
            if self.invert_init_angle:
                if (position <= (max_right - self.d_bot)) and (position >= max_left):
                    self.move_mid_block(position=position)
                    self.move_top_block(theta=-self.theta_s_top)
                if (position <= max_right) and (position > (max_right - self.d_bot)):
                    self.move_mid_block(theta=self.theta_s_bot)
                    self.move_top_block(position=position)
            else:
                if (position <= max_right) and (position > (max_right - self.d_bot)):
                    self.move_mid_block(position=position)
                    self.move_top_block(theta=self.theta_s_top)
                if (position <= (max_right - self.d_bot)) and (position >= max_left):
                    self.move_mid_block(theta=-self.theta_s_bot)
                    self.move_top_block(position=position)

    def update_seq_E(self, u_i, forward):
        """
        00 -> 10 -> 01 -> 11 -> 01 -> 00
        """
        position = u_i + self.x_offset

        max_left = - (self.d_bot / 2) - (self.d_top / 2)
        max_right = (self.d_bot / 2) + (self.d_top / 2)

        if forward:
            if (position >= max_left) and (position < (max_left + self.d_bot)):
                self.move_mid_block(position=position)
                self.move_top_block(theta=-self.theta_s_top)
            if (position >= (max_left + self.d_top)) and (position <= max_right):
                self.move_top_block(theta=self.theta_s_top)
                self.move_mid_block(position=position)
                self.move_top_block(theta=self.theta_s_top)
        else:
            if (position <= max_right) and (position > (max_right - self.d_bot)):
                self.move_mid_block(position=position)
                self.move_top_block(theta=self.theta_s_top)
            if (position >= max_left) and (position <= (max_right - self.d_bot)):
                self.move_mid_block(theta=-self.theta_s_top)
                self.move_top_block(position=position)

    def update_seq_F(self, u_i, forward):
        """
        00 -> 01 -> 10 -> 11 -> 10 -> 00
        """
        position = u_i + self.x_offset

        max_left = - (self.d_bot / 2) - (self.d_top / 2)
        max_right = (self.d_bot / 2) + (self.d_top / 2)

        if forward:
            if (position >= max_left) and (position < (max_left + self.d_top)):
                self.move_mid_block(theta=-self.theta_s_bot)
                self.move_top_block(position=position)
                # Would need a transition between this jump
            if (position >= (max_left + self.d_bot)) and (position <= max_right):
                self.move_mid_block(theta=self.theta_s_top)
                self.move_top_block(position=position)
        else:
            if (position <= max_right) and (position > (max_right - self.d_top)):
                self.move_mid_block(theta=self.theta_s_bot)
                self.move_top_block(position=position)

            if (position <= (max_right - self.d_bot)) and (position >= max_left):
                self.move_mid_block(position=position)
                self.move_top_block(theta=-self.theta_s_top)

    def update_seq_G(self, u_i, forward):
        """
        00 -> 10 -> 11 -> 01 -> 10 -> 00
        """
        position = u_i + self.x_offset

        max_left = - (self.d_bot / 2) - (self.d_top / 2)
        max_right = (self.d_bot / 2) + (self.d_top / 2)

        if forward:
            if (position >= max_left) and (position < (max_left + self.d_bot)):
                self.move_mid_block(position=position)
                self.move_top_block(theta=-self.theta_s_top)
            if (position >= (max_left + self.d_bot)) and (position <= max_right):
                self.move_mid_block(theta=self.theta_s_top)
                self.move_top_block(position=position)
        else:
            if (position <= max_right) and (position > (max_right - self.d_bot)):
                self.move_mid_block(position=position)
                self.move_top_block(theta=self.theta_s_top)
                # Would need a transition between this jump
            if (position <= (max_right - self.d_bot)) and (position >= max_left):
                self.move_top_block(theta=-self.theta_s_top)
                self.move_mid_block(position=position)
                self.move_top_block(theta=-self.theta_s_top)

    def update_seq_H(self, u_i, forward):
        """
        00 -> 01 -> 11 -> 10 -> 01 -> 00
        """
        position = u_i + self.x_offset

        max_left = - (self.d_bot / 2) - (self.d_top / 2)
        max_right = (self.d_bot / 2) + (self.d_top / 2)

        if forward:
            if (position >= max_left) and (position < (max_left + self.d_top)):
                self.move_mid_block(theta=-self.theta_s_bot)
                self.move_top_block(position=position)

            if (position >= (max_left + self.d_top)) and (position <= max_right):
                self.move_mid_block(position=position)
                self.move_top_block(theta=self.theta_s_top)
        else:
            if (position <= max_right) and (position > (max_right - self.d_top)):
                self.move_mid_block(theta=self.theta_s_bot)
                self.move_top_block(position=position)
                # Would need a transition between this jump
            if (position >= max_left) and (position <= (max_right - self.d_bot)):
                self.move_mid_block(theta=-self.theta_s_top)
                self.move_top_block(position=position)
        pass

    def update_seq_I(self, u_i, forward):
        """
        00 -> 10 -> 01 -> 11 -> 01 -> 10 -> 00
        """
        position = u_i + self.x_offset

        max_left = - (self.d_bot / 2) - (self.d_top / 2)
        max_right = (self.d_bot / 2) + (self.d_top / 2)

        if forward:
            if (position >= max_left) and (position < (max_left + self.d_bot)):
                self.move_mid_block(position=position)
                self.move_top_block(theta=-self.theta_s_top)
                # Would need a transition between this jump
            if (position >= (max_left + self.d_top)) and (position <= max_right):
                self.move_top_block(theta=self.theta_s_top)
                self.move_mid_block(position=position)
                self.move_top_block(theta=self.theta_s_top)
        else:
            if (position <= max_right) and (position > (max_right - self.d_bot)):
                self.move_mid_block(position=position)
                self.move_top_block(theta=self.theta_s_top)
                # Would need a transition between this jump
            if (position <= (max_right - self.d_bot)) and (position >= max_left):
                self.move_top_block(theta=-self.theta_s_top)
                self.move_mid_block(position=position)
                self.move_top_block(theta=-self.theta_s_top)

    def update_seq_J(self, u_i, forward):
        """
        00 -> 01 -> 10 -> 11 -> 10 -> 01 -> 00
        """
        position = u_i + self.x_offset

        max_left = - (self.d_bot / 2) - (self.d_top / 2)
        max_right = (self.d_bot / 2) + (self.d_top / 2)

        if forward:
            if (position >= max_left) and (position < (max_left + self.d_top)):
                self.move_mid_block(theta=-self.theta_s_bot)
                self.move_top_block(position=position)
                # Would need a transition between this jump
            if (position >= (max_left + self.d_bot)) and (position <= max_right):
                self.move_mid_block(theta=self.theta_s_top)
                self.move_top_block(position=position)
        else:
            if (position <= max_right) and (position > (max_right - self.d_bot)):
                self.move_mid_block(position=position)
                self.move_top_block(theta=self.theta_s_top)
                # Would need a transition between this jump
            if (position <= (max_right - self.d_bot)) and (position >= max_left):
                self.move_top_block(theta=-self.theta_s_top)
                self.move_mid_block(position=position)
                self.move_top_block(theta=-self.theta_s_top)

    def update_seq_K(self, u_i, forward):
        """
        Cycle where both blocks are moving at the same time
        00 -> 11 -> 00
        """
        position = u_i + self.x_offset
        delta_position = position - self.block_top.center.x
        half_position = position - (delta_position / 2)

        max_left = - (self.d_bot / 2) - (self.d_top / 2)
        max_right = (self.d_bot / 2) + (self.d_top / 2)

        if forward:
            if (position >= max_left) and (position <= max_right):
                self.move_mid_block(position=half_position)
                self.move_top_block(position=position)
        else:
            if (position <= max_right) and (position >= max_left):
                self.move_mid_block(position=half_position)
                self.move_top_block(position=position)

    def update_seq_L(self, u_i, forward):
        """
        Cycle where in forward motion both block move at the same time
        but in backward motion middle block is moving first
        00 -> 11 -> 10 -> 00
        """
        position = u_i + self.x_offset

        max_left = - (self.d_bot / 2) - (self.d_top / 2)
        max_right = (self.d_bot / 2) + (self.d_top / 2)

        if forward:
            delta_position = position - self.block_top.center.x
            half_position = position - (delta_position / 2)
            if (position >= max_left) and (position <= max_right):
                self.move_mid_block(position=half_position)
                self.move_top_block(position=position)
        else:
            if (position <= max_right) and (position > (max_right - self.d_bot)):
                self.move_mid_block(position=position)
                self.move_top_block(theta=self.theta_s_top)
            if (position >= max_left) and (position <= (max_right - self.d_bot)):
                self.move_mid_block(theta=-self.theta_s_top)
                self.move_top_block(position=position)

    def update_seq_M(self, u_i, forward):
        """
        Cycle where in forward motion both block move at the same time
        but in backward motion top block is moving first
        00 -> 11 -> 01 -> 00
        """
        position = u_i + self.x_offset

        max_left = - (self.d_bot / 2) - (self.d_top / 2)
        max_right = (self.d_bot / 2) + (self.d_top / 2)

        if forward:
            delta_position = position - self.block_top.center.x
            half_position = position - (delta_position / 2)
            if (position >= max_left) and (position <= max_right):
                self.move_mid_block(position=half_position)
                self.move_top_block(position=position)
        else:
            if (position <= max_right) and (position > (max_right - self.d_top)):
                self.move_mid_block(theta=self.theta_s_bot)
                self.move_top_block(position=position)

            if (position <= (max_right - self.d_bot)) and (position >= max_left):
                self.move_mid_block(position=position)
                self.move_top_block(theta=-self.theta_s_top)

    def update_seq_N(self, u_i, forward):
        """
        Cycle where in backward motion both block move at the same time
        but in forward motion middle block is moving first
        00 -> 10 -> 11 -> 00
        """
        position = u_i + self.x_offset

        max_left = - (self.d_bot / 2) - (self.d_top / 2)
        max_right = (self.d_bot / 2) + (self.d_top / 2)

        if forward:
            if (position >= max_left) and (position < (max_left + self.d_bot)):
                self.move_mid_block(position=position)
                self.move_top_block(theta=-self.theta_s_top)
            if (position >= (max_left + self.d_bot)) and (position <= max_right):
                self.move_mid_block(theta=self.theta_s_top)
                self.move_top_block(position=position)
        else:
            delta_position = position - self.block_top.center.x
            half_position = position - (delta_position / 2)
            if (position <= max_right) and (position >= max_left):
                self.move_mid_block(position=half_position)
                self.move_top_block(position=position)

    def update_seq_O(self, u_i, forward):
        """
        Cycle where first top block move in forward pass, but mid block move first
        in backward pass.
        00 -> 10 -> 11 -> 10 -> 00
        """
        position = u_i + self.x_offset

        max_left = - (self.d_bot / 2) - (self.d_top / 2)
        max_right = (self.d_bot / 2) + (self.d_top / 2)

        if forward:
            if (position >= max_left) and (position < (max_left + self.d_top)):
                self.move_mid_block(theta=-self.theta_s_bot)
                self.move_top_block(position=position)

            if (position >= (max_left + self.d_top)) and (position <= max_right):
                self.move_mid_block(position=position)
                self.move_top_block(theta=self.theta_s_top)
        else:
            delta_position = position - self.block_top.center.x
            half_position = position - (delta_position / 2)
            if (position <= max_right) and (position >= max_left):
                self.move_mid_block(position=half_position)
                self.move_top_block(position=position)

    def move_mid_block(self, position=None, theta=None):
        """
        Function to displace the middle block, two possibilities:
        1: by setting a position in the reference frame of the joint
        2: by setting an angle

        This will return the displacement that was done.
        """
        if position is not None:
            length = self.bars_bot.length
            dh = position

            _dh = dh - self.block_top.center.x
            new_anchor_x_pos = self.bars_bot.high_anchor.x + _dh
            dist = new_anchor_x_pos - self.bars_bot.low_anchor.x

            internal = dist / length
            if internal > 1.0:
                frameinfo = getframeinfo(currentframe())
                raise ValueError('FILE {0}, LINE {1} : internal = {2}'.format(
                    frameinfo.filename, frameinfo.lineno, internal))

            self.theta_i_bot = np.arcsin(internal)

            dv = np.cos(self.theta_i_bot) * length
            self.block_mid.set_position(
                _x=self.block_mid.center.x + _dh,
                _y=dv + (self.block_mid.height/2) - self.block_mid.anchor_d
            )

        if theta is not None:
            self.theta_i_bot = theta
            dh = np.sin(self.theta_i_bot) * self.bars_bot.length
            dv = np.cos(self.theta_i_bot) * self.bars_bot.length

            self.block_mid.set_position(
                _x=self.block_bot.center.x + dh,
                _y=self.block_bot.get_anchor(type='t').y + (self.block_mid.height/2) - self.block_mid.anchor_d + dv
            )

        self.bars_bot.low_anchor = self.block_bot.get_anchor(type='t')
        self.bars_bot.high_anchor = self.block_mid.get_anchor(type='b')
        self.spring_bot.Q.x = self.block_mid.center.x
        self.spring_bot.Q.y = self.bars_bot.high_anchor.y

    def move_top_block(self, position=None, theta=None):
        """
        Function to displace the top block, two possibilities:
        1: by setting a position in the reference frame of the joint
        2: by setting an angle

        This will return the displacement that was done.
        """
        displacement_done = 0.0

        if position is not None:
            length = self.bars_top.length

            _dh = position - self.block_top.center.x
            new_anchor_x_pos = self.bars_top.high_anchor.x + _dh
            dist = new_anchor_x_pos - self.bars_top.low_anchor.x

            internal = dist / length
            if internal > 1.0:
                frameinfo = getframeinfo(currentframe())
                raise ValueError('FILE {0}, LINE {1} : internal = {2}'.format(
                    frameinfo.filename, frameinfo.lineno, internal))

            self.theta_i_top = np.arcsin(internal)

            dv = np.cos(self.theta_i_top) * length
            self.block_top.set_position(
                _x=position,
                _y=self.block_mid.get_anchor(type='t').y + (self.block_top.height/2) - self.block_top.anchor_d + dv
            )
            displacement_done = _dh

        if theta is not None:
            prev_dh = np.sin(self.theta_i_top)
            self.theta_i_top = theta
            dh = np.sin(self.theta_i_top) * self.bars_top.length
            dv = np.cos(self.theta_i_top) * self.bars_top.length

            self.block_top.set_position(
                _x=self.block_mid.center.x + dh,
                _y=self.block_mid.get_anchor(type='t').y + (self.block_top.height/2) - self.block_top.anchor_d + dv
            )

            displacement_done = dh - prev_dh

        self.bars_top.low_anchor = self.block_mid.get_anchor(type='t')
        self.bars_top.high_anchor = self.block_top.get_anchor(type='b')
        self.spring_top.Q.x = self.block_top.center.x
        self.spring_top.Q.y = self.bars_top.high_anchor.y
        self.spring_top.P.x = self.block_mid.center.x
        self.spring_top.P.y = self.block_mid.get_anchor(type='t').y

        return displacement_done

    def update_legs(self):
        if len(self.C) != 0:
            old_C = self.C[-1]
        else:
            old_A = Coordinate(
                x=self.block_top.center.x - (Utils.LEG_OFFSET / 2),
                y=self.block_top.center.y,
                z=0
            )
            old_B = Coordinate(
                x=self.block_mid.center.x + (Utils.LEG_OFFSET / 2),
                y=self.block_mid.center.y,
                z=0
            )
            old_C = self.compute_leg_height(old_A, old_B)

        self.A.append(
            Coordinate(
                x=self.block_top.center.x - (Utils.LEG_OFFSET / 2),
                y=self.block_top.center.y,
                z=0
            )
        )

        self.B.append(
            Coordinate(
                x=self.block_mid.center.x + (Utils.LEG_OFFSET / 2),
                y=self.block_mid.center.y,
                z=0
            )
        )

        self.C.append(self.compute_leg_height(self.A[-1], self.B[-1]))
        movement = self.C[-1] - old_C
        if (self.invert_y):
            movement.y *= -1
        return movement

    def draw(self, frame):
        self.block_bot.draw(frame, self.structure_offset, self.invert_y)
        self.block_mid.draw(frame, self.structure_offset, self.invert_y)
        self.block_top.draw(frame, self.structure_offset, self.invert_y)

        # Draw bars
        self.bars_bot.draw(frame, self.structure_offset, self.invert_y)
        self.bars_top.draw(frame, self.structure_offset, self.invert_y)

        # Draw spring
        self.spring_bot.draw(frame, self.structure_offset, self.invert_y)
        self.spring_top.draw(frame, self.structure_offset, self.invert_y)

        self.draw_C(frame)

    def draw_C(self, frame):
        """
            Draw the C point in the 2D top view
        """
        c = self.get_real_leg()
        cv2.circle(
            frame,
            (
                Utils.ConvertX(c.x),
                Utils.ConvertY(c.y)
            ),
            8,
            color=Utils.blue,
            thickness=-1
        )

    def draw_legs(self, frame, location_x, location_y, touching):
        legs_thickness = 3

        frame = cv2.line(
            frame,
            (
                Utils.ConvertX_location(self.A[-1].x, location_x),
                Utils.ConvertY_location(self.A[-1].z, location_y)
            ),
            (
                Utils.ConvertX_location(self.C[-1].x, location_x),
                Utils.ConvertY_location(self.C[-1].z, location_y)
            ),
            self.top_color,
            thickness=legs_thickness
        )

        frame = cv2.line(
            frame,
            (
                Utils.ConvertX_location(self.B[-1].x, location_x),
                Utils.ConvertY_location(self.B[-1].z, location_y)
            ),
            (
                Utils.ConvertX_location(self.C[-1].x, location_x),
                Utils.ConvertY_location(self.C[-1].z, location_y)
            ),
            (0, 0, 0),
            thickness=legs_thickness
        )

        position_bot_left = (
            int(Utils.ConvertX_location(0, location_x)),
            int(Utils.ConvertY_location(-0.01, location_y))
        )
        frame = cv2.putText(
            frame,
            self.name,
            position_bot_left,
            Utils.font,
            Utils.fontScale,
            Utils.gray,
            Utils.text_thickness,
            cv2.LINE_AA
        )

        # Draw Ground
        if touching:
            ground_color = Utils.red
        else:
            ground_color = Utils.black
        frame = cv2.line(
            frame,
            (
                Utils.ConvertX_location(-0.1, location_x),
                Utils.ConvertY_location(self.ground_distance, location_y)
            ),
            (
                Utils.ConvertX_location(0.1, location_x),
                Utils.ConvertY_location(self.ground_distance, location_y)
            ),
            color=ground_color,
            thickness=legs_thickness
        )
        return frame
