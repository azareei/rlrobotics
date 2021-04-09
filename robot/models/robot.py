from models.joint import Joint
from coordinates import Coordinate
import cv2
from utils import Utils
import numpy as np
import numpy.ma as ma
from inspect import currentframe, getframeinfo


class Robot:
    def __init__(self):
        # Actuation 1
        self.J1 = Joint(
            'A',
            _structure_offset=Coordinate(x=20/100, y=4/100),
            _invert_y=False,
            _invert_init_angle=False,
            _bot_color=Utils.yellow,
            _top_color=Utils.magenta,
            _name='J1'
        )
        self.J4 = Joint(
            'A',
            _structure_offset=Coordinate(x=-20/100, y=-4/100),
            _invert_y=True,
            _invert_init_angle=False,
            _bot_color=Utils.yellow,
            _top_color=Utils.magenta,
            _name='J4'
        )

        # Actuation 2
        self.J2 = Joint(
            'B',
            _structure_offset=Coordinate(x=-20/100, y=4/100),
            _invert_y=False,
            _invert_init_angle=True,
            _bot_color=Utils.yellow,
            _top_color=Utils.green,
            _name='J2'
        )
        self.J3 = Joint(
            'B',
            _structure_offset=Coordinate(x=20/100, y=-4/100),
            _invert_y=True,
            _invert_init_angle=True,
            _bot_color=Utils.yellow,
            _top_color=Utils.green,
            _name='J3'
        )

        self.position = Coordinate(x=0, y=0, z=0)

    def update_position(self, actuation_1, actuation_2, actuation_1_dir, actuation_2_dir):
        mov1 = self.J1.update_position(actuation_1, actuation_1_dir)
        mov2 = self.J2.update_position(actuation_2, actuation_2_dir)

        mov3 = self.J3.update_position(actuation_2, actuation_2_dir)
        mov4 = self.J4.update_position(actuation_1, actuation_1_dir)

        # Fiter out to keep only x axis.
        mov_array_x = np.array([mov1.x, mov2.x, mov3.x, mov4.x])
        mov_array_y = np.array([mov1.y, mov2.y, mov3.y, mov4.y])
        self.update_attitude(mov_array_x, mov_array_y)

    def update_attitude(self, mov_array_x, mov_array_y):
        """
        Compute the ground height relative to the robot and compute the displacement of the robot with the legs
        that is touching the floor
        """
        h = np.array([self.J1.C.z, self.J2.C.z, self.J3.C.z, self.J4.C.z])
        self.ground = max(h)
        _touching_legs = np.where(h == self.ground)
        # Create self.touching = [True, False, True, False] or similar
        self.touching_legs = np.isin(np.arange(4), _touching_legs)
        nb_touching_legs = np.sum(self.touching_legs)

        delta_x = 0
        delta_y = 0  # for now we consider that it is symetrical

        # X Case
        mov_mx_x = ma.masked_array(mov_array_x, mask=np.invert(self.touching_legs))

        if np.abs(np.sum(np.sign(mov_mx_x))) == nb_touching_legs:
            # Means that all legs touching the floor moved in same x direction
            # First move in X by the smallest common movement
            min_mov = mov_mx_x[np.argmin(np.abs(mov_mx_x))]
            delta_x += min_mov
            mov_mx_x -= min_mov

            # Need to do somth with the rest of the movement
            if np.sum(mov_mx_x) != 0:
                frameinfo = getframeinfo(currentframe())
                print('[ATT] FILE {0}, LINE {1} : Some movement not added'.format(
                        frameinfo.filename, frameinfo.lineno))
        else:
            # Means that all legs touching the floor didn't moved in same x direction
            min_mov = mov_mx_x[np.argmin(np.abs(mov_mx_x))]
            mov_mx_x -= min_mov
            delta_x += np.sum(mov_mx_x)

        self.position.x -= delta_x
        self.position.y -= delta_y

    def draw(self, frame):
        self.draw_joints(frame)
        self.draw_main_block(frame)
        self.draw_legs(frame)
        self.draw_ground(frame)

    def draw_joints(self, frame):
        self.J1.draw(frame)
        self.J2.draw(frame)
        self.J3.draw(frame)
        self.J4.draw(frame)

    def draw_main_block(self, frame):
        if self.J2.invert_y is True:
            inv = -1
        else:
            inv = 1

        end = (
            int(Utils.ConvertX(self.J2.block_bot.center.x + self.J2.structure_offset.x)),
            int(Utils.ConvertY(inv * self.J2.block_bot.center.y + self.J2.structure_offset.y))
        )

        if self.J3.invert_y is True:
            inv = -1
        else:
            inv = 1

        start = (
            int(Utils.ConvertX(self.J3.block_bot.center.x + self.J3.structure_offset.x)),
            int(Utils.ConvertY(inv * self.J3.block_bot.center.y + self.J3.structure_offset.y))
        )

        return cv2.rectangle(
            frame,
            start,
            end,
            Utils.yellow,
            thickness=10
        )

    def draw_legs(self, frame):
        self.J1.draw_legs(
            frame,
            location_x='right',
            location_y='bottom',
            touching=self.touching_legs[0],
            ground=self.ground
        )
        self.J2.draw_legs(
            frame,
            location_x='left',
            location_y='bottom',
            touching=self.touching_legs[1],
            ground=self.ground
        )
        self.J3.draw_legs(
            frame,
            location_x='right',
            location_y='top',
            touching=self.touching_legs[2],
            ground=self.ground
        )
        self.J4.draw_legs(
            frame,
            location_x='left',
            location_y='top',
            touching=self.touching_legs[3],
            ground=self.ground
        )

    def draw_ground(self, frame):
        center = (
            Utils.ConvertX(-self.position.x),
            Utils.ConvertY(-self.position.y)
        )
        cv2.circle(
            frame,
            center,    
            15,
            Utils.red,
            thickness=-1
        )
