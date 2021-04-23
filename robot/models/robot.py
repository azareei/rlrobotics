from models.joint import Joint
from coordinates import Coordinate
import cv2
from utils import Utils
import numpy as np
import numpy.ma as ma
from inspect import currentframe, getframeinfo


class Robot:
    def __init__(self,
                 _seq1, _invert_y1, _invert_init_angle1,
                 _seq2, _invert_y2, _invert_init_angle2,
                 _seq3, _invert_y3, _invert_init_angle3,
                 _seq4, _invert_y4, _invert_init_angle4):
        # Actuation 1
        self.J1 = Joint(
            _seq1,
            _structure_offset=Coordinate(x=20/100, y=4/100),
            _invert_y=_invert_y1,
            _invert_init_angle=_invert_init_angle1,
            _bot_color=Utils.yellow,
            _top_color=Utils.magenta,
            _name='J1'
        )
        self.J4 = Joint(
            _seq4,
            _structure_offset=Coordinate(x=-20/100, y=-4/100),
            _invert_y=_invert_y4,
            _invert_init_angle=_invert_init_angle4,
            _bot_color=Utils.yellow,
            _top_color=Utils.magenta,
            _name='J4'
        )

        # Actuation 2
        self.J2 = Joint(
            _seq2,
            _structure_offset=Coordinate(x=-20/100, y=4/100),
            _invert_y=_invert_y2,
            _invert_init_angle=_invert_init_angle2,
            _bot_color=Utils.yellow,
            _top_color=Utils.green,
            _name='J2'
        )
        self.J3 = Joint(
            _seq3,
            _structure_offset=Coordinate(x=20/100, y=-4/100),
            _invert_y=_invert_y3,
            _invert_init_angle=_invert_init_angle3,
            _bot_color=Utils.yellow,
            _top_color=Utils.green,
            _name='J3'
        )

        self.position = []
        self.angle = []
        self.ground = 0  # Represente the high on the Centre of Gravity of the robot

    def update_position(self, actuation_1, actuation_2, actuation_1_dir, actuation_2_dir):
        mov1 = self.J1.update_position(actuation_1, actuation_1_dir)
        mov2 = self.J2.update_position(actuation_2, actuation_2_dir)

        mov3 = self.J3.update_position(actuation_2, actuation_2_dir)
        mov4 = self.J4.update_position(actuation_1, actuation_1_dir)

        mov_array_x = np.array([mov1.x, mov2.x, mov3.x, mov4.x])
        mov_array_y = np.array([mov1.y, mov2.y, mov3.y, mov4.y])
        self.update_attitude(mov_array_x, mov_array_y)

    def update_attitude(self, mov_array_x, mov_array_y):
        """
        Compute the ground height relative to the robot and compute the displacement of the robot with the legs
        that is touching the floor
        """
        self.update_orientation()

        delta_x = 0
        delta_y = 0  # for now we consider that it is symetrical

        # X Case
        mov_mx_x = ma.masked_array(mov_array_x, mask=np.invert(self.touching_legs))

        if np.abs(np.sum(np.sign(mov_mx_x))) == self.nb_touching_legs:
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

        delta_x, delta_y, delta_z = 0, 0, 0
        delta = Coordinate(x=delta_x, y=delta_y, z=delta_z)
        if len(self.position) == 0:
            self.position.append(-delta)
        else:
            self.position.append(self.position[-1] - delta)

    def update_orientation(self):
        legs_c = np.array([
            self.J1.C[-1].to_list('xyz'),
            self.J2.C[-1].to_list('xyz'),
            self.J3.C[-1].to_list('xyz'),
            self.J4.C[-1].to_list('xyz')
        ])

        # get max distance to frame
        legs_z = legs_c[:, 2]
        h = max(legs_z)

        # First pass to understand touching legs
        touching_legs = np.where(legs_z == h)
        # Create self.touching = [True, False, True, False] or similar
        touching_legs = np.isin(np.arange(4), touching_legs)
        nb_touching_legs = np.sum(touching_legs)

        if nb_touching_legs == 4:
            print('[FIRST PASS 4 legs]')
            # Means the robot is flat and no update for the legs
            self.angle.append(Coordinate(x=0, y=0, z=0))
            self.touching_legs = touching_legs
            self.nb_touching_legs = nb_touching_legs
            self.ground = h
            return
        elif nb_touching_legs == 3:  # TODO 3 legs update
            print('[FIRST PASS 3 legs]')
        elif nb_touching_legs == 2:  # TODO 2 legs update
            if (touching_legs[0] == touching_legs[3]) or (touching_legs[1] == touching_legs[2]):  # TODO 2 legs diag update
                print('[FIRST PASS 2 legs diag]')
                self.angle.append(Coordinate(x=0, y=0, z=0))
                self.touching_legs = touching_legs
                self.nb_touching_legs = nb_touching_legs
                self.ground = h
            else:  # TODO 2 legs no dial update
                print('[FIRST PASS 2 legs no diag]')
        elif nb_touching_legs == 1:  # TODO one legs update
            print('[FIRST PASS 1 legs]')

    def draw(self, frame):
        self.draw_joints(frame)
        self.draw_main_block(frame)
        self.draw_legs(frame)
        return frame

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

        cv2.rectangle(
            frame,
            start,
            end,
            Utils.yellow,
            thickness=10
        )

    def draw_legs(self, frame):  # TODO Need to  show ground with angle of the robot.
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

    def max_actuation(self):
        a1 = min(
            self.J1.d_top + self.J1.d_bot,
            self.J4.d_top + self.J4.d_bot
        )
        a2 = min(
            self.J2.d_top + self.J2.d_bot,
            self.J3.d_top + self.J3.d_bot
        )

        return a1, a2
