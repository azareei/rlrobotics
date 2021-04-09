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
                 _theta1=0.785, _theta2=0.785):
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
        _d = 1/100
        _w = 5.5/100
        _h = 5.5/100
        _center = Coordinate(x=0, y=_d - (_h / 2))
        self.block_bot = Block(_w, _h, _center, _d, self.bot_color)

        # Create mid block
        _center = Coordinate(x=0, y=_r1 - _d + (_h / 2))
        self.block_mid = Block(_w, _h, _center, _d, (0, 0, 0))

        # Create top block
        _center = Coordinate(x=0, y=self.block_mid.get_anchor(type="t").y + _r2 - _d + (_h/2))
        self.block_top = Block(_w, _h, _center, _d, self.top_color)

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
        # print("Theta S : {}".format(self.theta_s_top))

        self.theta_i_bot = 0
        self.theta_i_top = 0

        self.prev_ui = None

        self.A = Coordinate(x=0, y=0, z=0)
        self.B = Coordinate(x=0, y=0, z=0)
        self.C = Coordinate(x=0, y=0, z=0)

        self.init_position()

    def compute_leg_height(self, A, B):
        legs_length = 5 / 100

        tmp = legs_length**2 - ((B.x - A.x) / 2)**2
        return Coordinate(x=(A.x + B.x)/2, y=(A.y + B.y)/2, z=np.sqrt(tmp))

    def init_position(self):
        if self.invert_init_angle is False:
            self.theta_i_top = -self.theta_s_top
            self.theta_i_bot = -self.theta_s_bot
        else:
            self.theta_i_top = self.theta_s_top
            self.theta_i_bot = self.theta_s_bot

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

    def update_position(self, u_i, forward):
        """
        u_i is the delta between the previous position et the one now.

        Return:
            Coordinate vector corresponding to the translation before and
            after the displacement of the legs.
        """
        if self.sequence == 'A':
            self.update_seq_A(u_i, forward)
        elif self.sequence == 'B':
            self.update_seq_B(u_i, forward)
        return self.update_legs()

    def update_seq_A(self, u_i, forward):
        position = u_i + self.x_offset
        length = self.bars_bot.length

        max_left = - (self.d_bot / 2) - (self.d_top / 2)
        max_right = (self.d_bot / 2) + (self.d_top / 2)

        if forward:
            if (position >= max_left) and \
                    (position < (max_left + self.d_bot)):
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

            if (position >= (max_left + self.d_bot)) and (position <= max_right):
                # Still need to ensure the bottom block is theta_s
                dh = position

                _dh = dh - self.block_top.center.x
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
                    _x=dh,
                    _y=self.block_mid.get_anchor(type='t').y + (self.block_top.height/2) - self.block_top.anchor_d + dv
                )
                self.bars_top.low_anchor = self.block_mid.get_anchor(type='t')
                self.bars_top.high_anchor = self.block_top.get_anchor(type='b')
                self.spring_top.Q.x = self.block_top.center.x
                self.spring_top.Q.y = self.bars_top.high_anchor.y
                self.spring_top.P.x = self.block_mid.center.x
                self.spring_top.P.y = self.block_mid.get_anchor(type='t').y
        else:
            if (position <= max_right) and (position > (max_right - self.d_bot)):
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
            if (position >= max_left) and (position <= (max_right - self.d_bot)):
                # Still need to ensure the bottom block is -theta_s
                dh = position

                _dh = dh - self.block_top.center.x
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

        max_left = - (self.d_bot / 2) - (self.d_top / 2)
        max_right = (self.d_bot / 2) + (self.d_top / 2)

        if forward:
            if (position >= max_left) and (position < (max_left + self.d_top)):
                self.move_top_block(position)

            if (position >= (max_left + self.d_top)) and (position <= max_right):
                # delta = 0
                # # Need to ensure theta_top_i was maxed, otherwise move top block and compute delta
                # if self.theta_i_top < self.theta_s_top:
                #     gt = np.sin(self.theta_s_top) * length
                #     rel = np.sin(self.theta_i_top) * length
                #     delta = gt-rel
                #     self.move_top_block(self.block_top.center.x + delta)

                # # Move bottom block
                # self.move_bot_block(position)

                # # Ensure theta_i is maxed for top block

                # Ensure theta_s is min out
                self.theta_i_top = self.theta_s_top
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

        else:
            if (position <= max_right) and (position > (max_right - self.d_top)):
                self.move_top_block(position)

            if (position <= (max_right - self.d_bot)) and (position >= max_left):
                # Ensure theta_s is min out
                self.theta_i_top = -self.theta_s_top
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

    def move_bot_block(self, dh=None, theta=None):
        length = self.bars_bot.length

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
        self.bars_bot.low_anchor = self.block_bot.get_anchor(type='t')
        self.bars_bot.high_anchor = self.block_mid.get_anchor(type='b')
        self.spring_bot.Q.x = self.block_mid.center.x
        self.spring_bot.Q.y = self.bars_bot.high_anchor.y

    def move_top_block(self, dh):
        length = self.bars_top.length

        _dh = dh - self.block_top.center.x
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
            _x=dh,
            _y=self.block_mid.get_anchor(type='t').y + (self.block_top.height/2) - self.block_top.anchor_d + dv
        )
        self.bars_top.low_anchor = self.block_mid.get_anchor(type='t')
        self.bars_top.high_anchor = self.block_top.get_anchor(type='b')
        self.spring_top.Q.x = self.block_top.center.x
        self.spring_top.Q.y = self.bars_top.high_anchor.y
        self.spring_top.P.x = self.block_mid.center.x
        self.spring_top.P.y = self.block_mid.get_anchor(type='t').y

    def update_legs(self):
        offset = 4 / 100
        old_C = self.C
        self.A = Coordinate(x=self.block_top.center.x - offset, y=self.block_top.center.y, z=0)
        self.B = Coordinate(x=self.block_mid.center.x, y=self.block_mid.center.y, z=0)
        self.C = self.compute_leg_height(self.A, self.B)
        movement = self.C - old_C
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

    def draw_legs(self, frame, location_x, location_y, touching, ground):
        legs_thickness = 3

        frame = cv2.line(
            frame,
            (
                Utils.ConvertX_location(self.A.x, location_x),
                Utils.ConvertY_location(self.A.z, location_y)
            ),
            (
                Utils.ConvertX_location(self.C.x, location_x),
                Utils.ConvertY_location(self.C.z, location_y)
            ),
            self.top_color,
            thickness=legs_thickness
        )

        frame = cv2.line(
            frame,
            (
                Utils.ConvertX_location(self.B.x, location_x),
                Utils.ConvertY_location(self.B.z, location_y)
            ),
            (
                Utils.ConvertX_location(self.C.x, location_x),
                Utils.ConvertY_location(self.C.z, location_y)
            ),
            (0, 0, 0),
            thickness=legs_thickness
        )

        # Show T if touching
        if touching:
            position_bot_left = (
                int(Utils.ConvertX_location(0, location_x)),
                int(Utils.ConvertY_location(-0.01, location_y))
            )
            frame = cv2.putText(
                frame,
                'T',
                position_bot_left,
                Utils.font,
                Utils.fontScale,
                Utils.red,
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
                Utils.ConvertY_location(ground, location_y)
            ),
            (
                Utils.ConvertX_location(0.1, location_x),
                Utils.ConvertY_location(ground, location_y)
            ),
            color=ground_color,
            thickness=legs_thickness
        )
        return frame
