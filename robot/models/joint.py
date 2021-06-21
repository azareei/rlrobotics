"""
Module Joint

This module represent a joint (a leg) for the robot. It is composed of 3 blocks
1 arm and 1 spring. 

Attributes
----------
sequence : str
    A character representing the sequence in which the joint is running (A-O)
structure_offset : Coordinates
    The offset to the robot reference frame to place the joint at the right place while drawing
invert_y : bool
    Is true when the joint is vertically reversed, but with x axis still in the right place, so
    the block will still start from left to right
invert_init_angle : bool
    Is true when we start from right to left instead of left to right
reverse_actuation : bool
    If true, we need to revert the attribute invert_init_angle. This is used to produce the symmetry
bot_color : tuple
    A tuple in BGR to draw the bottom block color
top_color : tuple
    A tuple in BGR to draw the top block color
name : str
    A name for the joint, generally J1, J2, J3 or J4
r1 : float
    Length of arms between bottom block and middle block (meter)
r2 : float
    Length of arms between top block and middle block (meter)
theta_1 : float
    Maximum angle to reach for middle block relative to bottom block
theta_2 : float
    Maximum angle to reach for top block relative to middle block
leg_legnth : float
    The length of the arms that link the middle and top block to the tip of the leg (meter)
block_bot : Block
    Represent bottom block
block_mid : Block
    Represent middle block
block_top : Block
    Represent top block
bars_bot : Arm
    Represent arms between bottom and middle blocks
bars_top : Arm
    Represent armas between middle and top blocks
spring_bot : Spring
    Represent spring between bottom and middle blocks
spring_top : Spring
    Represent spring between middle and top blocks
theta_s_bot : double
    Maximum angle the middle block can rotate from vertical axis relative to bottom block
theta_s_top : double
    Maximum angle the top block can rotate from vertical axis relative to middle block
theta_i_bot : double
    Current angle of the middle block relative to the top block
theta_i_top : double
    Current angle of the top block relative to the middle block
x_offset : double
    Initial offset of the x position of top block's center in Joint reference frame.
A : Coordinates
    Coordinates of top block's center in Joint reference frame
B : Coordinates
    Coordinates of the middle block's center in Joint reference frame
C : Coordinates
    Coordinates of the leg tip in Joint reference frame

Methods
-------
__init__(self, _sequence, _structure_offset,
        _invert_y, _invert_init_angle, _reverse_actuation,
        _bot_color, _top_color,
        _name,
        _r1, _r2,
        _theta1, _theta2,
        _leg_length)
    Initialize the Joint
compute_leg_height(self, _A, _B)
    Compute the point C for the Joint
init_position(self)
    Reset the position of the Joint to initial position (generally from left to right)
get_real_leg(self)
    Compute and returns the position of the leg in robot's reference frame
update_position(self, u_i, forward)
    Entry point to compute a step of motion. This function will dispatch to corresponding
    sequence motion update
update_seq_X(self, u_i, forward)
    Compute the displacement of the Joint with respect to the sequence
move_mid_block(self, position=None, theta=None)
    Specialized function to displace the middle block to a position or to an angle
move_top_block(self, position=None, theta=None)
    Specialized function to displace the top block to a position or to an angle
update_legs(self)
    Update the information of the leg endpoint to the array
draw(self, frame)
    Draw the Joint
draw_C(self, frame)
    Draw a circle to represent the position off the point C
draw_legs(self, frame, location_x, location_y, touching)
    draw a side view of the leg to see where the leg is relative to the ground
"""
from inspect import currentframe, getframeinfo

import cv2
import numpy as np
from coordinates import Coordinate
from utils import Utils

from models.arm import Arm
from models.block import Block
from models.spring import Spring


class Joint:
    def __init__(self, _sequence, _structure_offset,
                 _invert_y, _invert_init_angle, _reverse_actuation,
                 _bot_color, _top_color,
                 _name,
                 _r1, _r2,
                 _theta1, _theta2,
                 _leg_length):
        """
        Initialiation function to create the t=0 condition of the joint. Including the creation
        of the blocks but also their positions.

        Parameters
        ----------
        sequence : str
            A character representing the sequence in which the joint is running (A-O)
        structure_offset : Coordinates
            The offset to the robot reference frame to place the joint at the right place while drawing
        invert_y : bool
            Is true when the joint is vertically reversed, but with x axis still in the right place, so
            the block will still start from left to right
        invert_init_angle : bool
            Is true when we start from right to left instead of left to right
        reverse_actuation : bool
            If true, we need to revert the attribute invert_init_angle. This is used to produce the symmetry
        bot_color : tuple
            A tuple in BGR to draw the bottom block color
        top_color : tuple
            A tuple in BGR to draw the top block color
        name : str
            A name for the joint, generally J1, J2, J3 or J4
        r1 : float
            Length of arms between bottom block and middle block (meter)
        r2 : float
            Length of arms between top block and middle block (meter)
        theta_1 : float
            Maximum angle to reach for middle block relative to bottom block
        theta_2 : float
            Maximum angle to reach for top block relative to middle block
        leg_legnth : float
            The length of the arms that link the middle and top block to the tip of the leg (meter)
        """

        # Define sequence
        self.sequence = _sequence
        self.structure_offset = _structure_offset
        self.invert_y = _invert_y
        self.bot_color = _bot_color
        self.top_color = _top_color
        self.name = _name
        self.invert_init_angle = _invert_init_angle
        if _reverse_actuation:
            self.invert_init_angle = not self.invert_init_angle

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
        self.bars_bot = Arm(
            self.block_bot.get_anchor(type="t"),
            self.block_mid.get_anchor(type="b"),
            _r1,
            self.block_bot.get_anchor_distance()
        )

        # Create the bars_top
        self.bars_top = Arm(
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

        self.leg_length = _leg_length

        self.A = []
        self.B = []
        self.C = []

        self.ground_distance = 0.0

        self.init_position()

    def init_position(self):
        """
        Initialize the position of the blocks (both on the left or right)
        """
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

    def compute_leg_height(self, _A, _B):
        """
        Compute leg height (AKA point C) thanks to two points (A and B)

        Parameters
        ----------
        _A : Coordinates
            Point A coordinates
        _B : Coordinates
            Point B coordinates

        Returns
        -------
        Coordinates
            Point C coordinates
        """
        tmp = self.leg_length**2 - ((_B.x - _A.x) / 2)**2
        return Coordinate(x=(_A.x + _B.x)/2, y=(_A.y + _B.y)/2, z=np.sqrt(tmp))

    def get_real_leg(self):
        """
            Compute the coordinate of point C in the Robot reference frame

            Returns
            -------
            Coordinates
                Point C coordinates in robot reference frame
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

    def update_position(self, u_i, forward):
        """
        Update the position of both block given a position input

        Parameters
        ----------
        u_i : double
           The position of the actuator (meter)
        forward : bool
            Gives indication if we are doing a forward or backward motion

        Returns
        -------
            Coordinate
                A coordinate that is the position change vector of point C
                for the step
        """
        # Call the right funcion for the joint sequence update_seq_G for example
        getattr(self, f'update_seq_{self.sequence}')(u_i, forward)
        return self.update_legs()

    def update_seq_A(self, u_i, forward):
        """
        Cyclic sequence where mid block always move first
        00 -> 10 -> 11 -> 01 -> 00

        Update the position of both block given a position input w/ seq A

        Parameters
        ----------
        u_i : double
           The position of the actuator (meter)
        forward : bool
            Gives indication if we are doing a forward or backward motion
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

        Update the position of both block given a position input w/ seq B

        Parameters
        ----------
        u_i : double
           The position of the actuator (meter)
        forward : bool
            Gives indication if we are doing a forward or backward motion
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
                    self.move_mid_block(theta=-self.theta_s_bot)
                    self.move_top_block(position=position)
                if (position <= max_right) and (position > (max_right - self.d_top)):
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

        Update the position of both block given a position input w/ seq C

        Parameters
        ----------
        u_i : double
           The position of the actuator (meter)
        forward : bool
            Gives indication if we are doing a forward or backward motion
        """
        position = u_i + self.x_offset

        max_left = - (self.d_bot / 2) - (self.d_top / 2)
        max_right = (self.d_bot / 2) + (self.d_top / 2)

        if forward:
            if self.invert_init_angle:
                if (position >= (max_left + self.d_bot) and (position <= max_right)):
                    self.move_mid_block(position=position)
                    self.move_top_block(theta=self.theta_s_top)
                if (position >= max_left) and (position < (max_left + self.d_bot)):
                    self.move_mid_block(theta=-self.theta_s_bot)
                    self.move_top_block(position=position)
            else:
                if (position >= max_left) and (position < (max_left + self.d_bot)):
                    self.move_mid_block(position=position)
                    self.move_top_block(theta=-self.theta_s_top)
                if (position >= (max_left + self.d_bot) and (position <= max_right)):
                    self.move_mid_block(theta=self.theta_s_bot)
                    self.move_top_block(position=position)
        else:
            if self.invert_init_angle:
                if (position <= (max_right - self.d_top)) and (position >= max_left):
                    self.move_mid_block(theta=-self.theta_s_bot)
                    self.move_top_block(position=position)
                if (position <= max_right) and (position > (max_right - self.d_top)):
                    self.move_mid_block(position=position)
                    self.move_top_block(theta=self.theta_s_top)
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

        Update the position of both block given a position input w/ seq D

        Parameters
        ----------
        u_i : double
           The position of the actuator (meter)
        forward : bool
            Gives indication if we are doing a forward or backward motion
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

        Update the position of both block given a position input w/ seq E

        Parameters
        ----------
        u_i : double
           The position of the actuator (meter)
        forward : bool
            Gives indication if we are doing a forward or backward motion
        """
        position = u_i + self.x_offset

        max_left = - (self.d_bot / 2) - (self.d_top / 2)
        max_right = (self.d_bot / 2) + (self.d_top / 2)

        if forward:
            if self.invert_init_angle:
                if (position >= (max_left + self.d_top)) and (position <= max_right):
                    self.move_mid_block(position=position)
                    self.move_top_block(theta=self.theta_s_top)
                if (position >= max_left) and (position < (max_left + self.d_bot)):
                    self.move_top_block(theta=-self.theta_s_top)
                    self.move_mid_block(position=position)
                    self.move_top_block(theta=-self.theta_s_top)
            else:
                if (position >= max_left) and (position < (max_left + self.d_bot)):
                    self.move_mid_block(position=position)
                    self.move_top_block(theta=-self.theta_s_top)
                if (position >= (max_left + self.d_top)) and (position <= max_right):
                    self.move_top_block(theta=self.theta_s_top)
                    self.move_mid_block(position=position)
                    self.move_top_block(theta=self.theta_s_top)
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

    def update_seq_F(self, u_i, forward):
        """
        00 -> 01 -> 10 -> 11 -> 10 -> 00

        Update the position of both block given a position input w/ seq F

        Parameters
        ----------
        u_i : double
           The position of the actuator (meter)
        forward : bool
            Gives indication if we are doing a forward or backward motion
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
            if self.invert_init_angle:
                if (position <= (max_right - self.d_bot)) and (position >= max_left):
                    self.move_mid_block(theta=-self.theta_s_bot)
                    self.move_top_block(position=position)
                if (position <= max_right) and (position > (max_right - self.d_top)):
                    self.move_mid_block(position=position)
                    self.move_top_block(theta=self.theta_s_top)
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

        Update the position of both block given a position input w/ seq G

        Parameters
        ----------
        u_i : double
           The position of the actuator (meter)
        forward : bool
            Gives indication if we are doing a forward or backward motion
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
                if (position <= (max_right - self.d_bot)) and (position >= max_left):
                    self.move_mid_block(position=position)
                    self.move_top_block(theta=-self.theta_s_top)
                    # Would need a transition between this jump
                if (position <= max_right) and (position > (max_right - self.d_bot)):
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

    def update_seq_H(self, u_i, forward):
        """
        00 -> 01 -> 11 -> 10 -> 01 -> 00

        Update the position of both block given a position input w/ seq H

        Parameters
        ----------
        u_i : double
           The position of the actuator (meter)
        forward : bool
            Gives indication if we are doing a forward or backward motion
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
                if (position >= max_left) and (position <= (max_right - self.d_bot)):
                    self.move_mid_block(theta=-self.theta_s_bot)
                    self.move_top_block(position=position)
                    # Would need a transition between this jump
                if (position <= max_right) and (position > (max_right - self.d_top)):
                    self.move_mid_block(theta=self.theta_s_top)
                    self.move_top_block(position=position)
            else:
                if (position <= max_right) and (position > (max_right - self.d_top)):
                    self.move_mid_block(theta=self.theta_s_bot)
                    self.move_top_block(position=position)
                    # Would need a transition between this jump
                if (position >= max_left) and (position <= (max_right - self.d_bot)):
                    self.move_mid_block(theta=-self.theta_s_top)
                    self.move_top_block(position=position)

    def update_seq_I(self, u_i, forward):
        """
        00 -> 10 -> 01 -> 11 -> 01 -> 10 -> 00

        Update the position of both block given a position input w/ seq I

        Parameters
        ----------
        u_i : double
           The position of the actuator (meter)
        forward : bool
            Gives indication if we are doing a forward or backward motion
        """
        position = u_i + self.x_offset

        max_left = - (self.d_bot / 2) - (self.d_top / 2)
        max_right = (self.d_bot / 2) + (self.d_top / 2)

        if forward:
            if self.invert_init_angle:
                if (position >= (max_left + self.d_top)) and (position <= max_right):
                    self.move_mid_block(position=position)
                    self.move_top_block(theta=self.theta_s_top)
                    # Would need a transition between this jump
                if (position >= max_left) and (position < (max_left + self.d_bot)):
                    self.move_top_block(theta=-self.theta_s_top)
                    self.move_mid_block(position=position)
                    self.move_top_block(theta=-self.theta_s_top)
            else:
                if (position >= max_left) and (position < (max_left + self.d_bot)):
                    self.move_mid_block(position=position)
                    self.move_top_block(theta=-self.theta_s_top)
                    # Would need a transition between this jump
                if (position >= (max_left + self.d_top)) and (position <= max_right):
                    self.move_top_block(theta=self.theta_s_top)
                    self.move_mid_block(position=position)
                    self.move_top_block(theta=self.theta_s_top)
        else:
            if self.invert_init_angle:
                if (position <= (max_right - self.d_bot)) and (position >= max_left):
                    self.move_mid_block(position=position)
                    self.move_top_block(theta=-self.theta_s_top)
                    # Would need a transition between this jump
                if (position <= max_right) and (position > (max_right - self.d_bot)):
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

        Update the position of both block given a position input w/ seq J

        Parameters
        ----------
        u_i : double
           The position of the actuator (meter)
        forward : bool
            Gives indication if we are doing a forward or backward motion
        """
        position = u_i + self.x_offset

        max_left = - (self.d_bot / 2) - (self.d_top / 2)
        max_right = (self.d_bot / 2) + (self.d_top / 2)

        if forward:
            if self.invert_init_angle:
                if (position >= (max_left + self.d_bot)) and (position <= max_right):
                    self.move_mid_block(theta=self.theta_s_bot)
                    self.move_top_block(position=position)
                    # Would need a transition between this jump
                if (position >= max_left) and (position < (max_left + self.d_top)):
                    self.move_mid_block(theta=-self.theta_s_top)
                    self.move_top_block(position=position)
            else:
                if (position >= max_left) and (position < (max_left + self.d_top)):
                    self.move_mid_block(theta=-self.theta_s_bot)
                    self.move_top_block(position=position)
                    # Would need a transition between this jump
                if (position >= (max_left + self.d_bot)) and (position <= max_right):
                    self.move_mid_block(theta=self.theta_s_top)
                    self.move_top_block(position=position)
        else:
            if self.invert_init_angle:
                if (position <= (max_right - self.d_bot)) and (position >= max_left):
                    self.move_mid_block(theta=-self.theta_s_top)
                    self.move_top_block(position=position)
                    # Would need a transition between this jump
                if (position <= max_right) and (position > (max_right - self.d_bot)):
                    self.move_mid_block(theta=self.theta_s_top)
                    self.move_top_block(position=position)
                    self.move_mid_block(theta=self.theta_s_top)
            else:
                if (position <= max_right) and (position > (max_right - self.d_bot)):
                    self.move_mid_block(theta=self.theta_s_top)
                    self.move_top_block(position=position)
                    # Would need a transition between this jump
                if (position <= (max_right - self.d_bot)) and (position >= max_left):
                    self.move_mid_block(theta=-self.theta_s_top)
                    self.move_top_block(position=position)
                    self.move_mid_block(theta=-self.theta_s_top)

    def update_seq_K(self, u_i, forward):
        """
        [THEORETICAL ONLY]
        Cycle where both blocks are moving at the same time
        00 -> 11 -> 00

        Update the position of both block given a position input w/ seq K

        Parameters
        ----------
        u_i : double
           The position of the actuator (meter)
        forward : bool
            Gives indication if we are doing a forward or backward motion
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
        [THEORETICAL ONLY]
        Cycle where in forward motion both block move at the same time
        but in backward motion top block is moving first
        00 -> 11 -> 01 -> 00

        Update the position of both block given a position input w/ seq L

        Parameters
        ----------
        u_i : double
           The position of the actuator (meter)
        forward : bool
            Gives indication if we are doing a forward or backward motion
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
            if self.invert_init_angle:
                if (position <= (max_right - self.d_bot)) and (position >= max_left):
                    self.move_mid_block(theta=-self.theta_s_bot)
                    self.move_top_block(position=position)
                if (position <= max_right) and (position > (max_right - self.d_top)):
                    self.move_mid_block(position=position)
                    self.move_top_block(theta=self.theta_s_top)
            else:
                if (position <= max_right) and (position > (max_right - self.d_top)):
                    self.move_mid_block(theta=self.theta_s_bot)
                    self.move_top_block(position=position)
                if (position <= (max_right - self.d_bot)) and (position >= max_left):
                    self.move_mid_block(position=position)
                    self.move_top_block(theta=-self.theta_s_top)

    def update_seq_M(self, u_i, forward):
        """
        [THEORETICAL ONLY]
        Cycle where in forward motion both block move at the same time
        but in backward motion middle block is moving first
        00 -> 11 -> 10 -> 00

        Update the position of both block given a position input w/ seq M

        Parameters
        ----------
        u_i : double
           The position of the actuator (meter)
        forward : bool
            Gives indication if we are doing a forward or backward motion
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

    def update_seq_N(self, u_i, forward):
        """
        [THEORETICAL ONLY]
        Cycle where in backward motion both block move at the same time
        but in forward motion middle block is moving first
        00 -> 10 -> 11 -> 00

        Update the position of both block given a position input w/ seq N

        Parameters
        ----------
        u_i : double
           The position of the actuator (meter)
        forward : bool
            Gives indication if we are doing a forward or backward motion
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
            delta_position = position - self.block_top.center.x
            half_position = position - (delta_position / 2)
            if (position <= max_right) and (position >= max_left):
                self.move_mid_block(position=half_position)
                self.move_top_block(position=position)

    def update_seq_O(self, u_i, forward):
        """
        [THEORETICAL ONLY]
        Cycle where first top block move in forward pass, but mid block move first
        in backward pass.
        00 -> 10 -> 11 -> 10 -> 00

        Update the position of both block given a position input w/ seq O

        Parameters
        ----------
        u_i : double
           The position of the actuator (meter)
        forward : bool
            Gives indication if we are doing a forward or backward motion
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

        Parameters
        ----------
        position : double, optional
            the position of the top block after the position
        theta : double, optional
            the angle that the middle block need to be

        Returns
        -------
            Coordinates
                This will return the displacement that was done as a vector
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

        Parameters
        ----------
        position : double, optional
            the position of the top block after the position
        theta : double, optional
            the angle that the top block need to be

        Returns
        -------
            Coordinates
                This will return the displacement that was done as a vector
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
        """
        This function is storing the position of the legs after each steps

        Returns
        -------
        Coordinates
            Last displacement in a coordinate vector
        """
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
        """
        Classic function to draw the complete joint

        Parameters
        ----------
        frame : numpy Array
            Image of the current frame to draw on.

        Returns
        -------
        frame : numpy Array
            Updated image with the joint on it.
        """
        self.block_bot.draw(frame, self.structure_offset, self.invert_y)
        self.block_mid.draw(frame, self.structure_offset, self.invert_y)
        self.block_top.draw(frame, self.structure_offset, self.invert_y)

        # Draw bars
        self.bars_bot.draw(frame, self.structure_offset, self.invert_y)
        self.bars_top.draw(frame, self.structure_offset, self.invert_y)

        # Draw spring
        self.spring_bot.draw(frame, self.structure_offset, self.invert_y)
        self.spring_top.draw(frame, self.structure_offset, self.invert_y)

        # Draw point C
        self.draw_C(frame)

    def draw_C(self, frame):
        """
        Method responsible to draw the point C on top view.

        Parameters
        ----------
        frame : numpy Array
            Image of the current frame to draw on.

        Returns
        -------
        frame : numpy Array
            Updated image with the C point on it.
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
        """
        Method responsible to draw the legs side view.

        Parameters
        ----------
        frame : numpy Array
            Image of the current frame to draw on.
        location_x : str
            can be left, middle or right to draw on left middle or right of the frame
        location_y : str
            Can be top, middle or bottom to draw on different y positionf of the frame
        touching : bool
            If true, means that this joint's leg is touching the ground

        Returns
        -------
        frame : numpy Array
            Updated image with the arms on it.
        """
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
