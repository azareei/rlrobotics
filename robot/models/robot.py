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
            _structure_offset=Coordinate(x=20/100, y=4/100, z=0),
            _invert_y=_invert_y1,
            _invert_init_angle=_invert_init_angle1,
            _bot_color=Utils.yellow,
            _top_color=Utils.magenta,
            _name='J1'
        )
        self.J4 = Joint(
            _seq4,
            _structure_offset=Coordinate(x=-20/100, y=-4/100, z=0),
            _invert_y=_invert_y4,
            _invert_init_angle=_invert_init_angle4,
            _bot_color=Utils.yellow,
            _top_color=Utils.magenta,
            _name='J4'
        )

        # Actuation 2
        self.J2 = Joint(
            _seq2,
            _structure_offset=Coordinate(x=-20/100, y=4/100, z=0),
            _invert_y=_invert_y2,
            _invert_init_angle=_invert_init_angle2,
            _bot_color=Utils.yellow,
            _top_color=Utils.green,
            _name='J2'
        )
        self.J3 = Joint(
            _seq3,
            _structure_offset=Coordinate(x=20/100, y=-4/100, z=0),
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

    def update_attitude(self, mov_x, mov_y):
        """
        Compute the ground height relative to the robot and compute the displacement of the robot with the legs
        that is touching the floor
        """
        pitch, roll = self.update_orientation()

        dx, dy, dz = 0, 0, 0
        yaw = 0.0

        # First compute displalcement for X and Y
        # Only compute with touching legs
        mx = ma.masked_array(mov_x, mask=np.invert(self.touching_legs))
        my = ma.masked_array(mov_y, mask=np.invert(self.touching_legs))
        dx = np.sum(mx)
        dy = np.sum(my)

        delta = Coordinate(x=dx, y=dy, z=dz)
        if len(self.position) == 0:
            self.position.append(-delta)
        else:
            self.position.append(self.position[-1] - delta)
        self.angle.append(np.array([pitch, roll, yaw]))

    def update_orientation(self):
        """
        This function will compute the orientation of the robot relative to the ground.

        It will also compute in <=3 passes what legs are touching the floor and from which
        height.
        """
        ground1, ground2, ground3 = 0, 0, 0
        touching_legs_index = np.array([])
        touching_legs_index_P1 = np.array([])
        touching_legs_index_P2 = np.array([])
        touching_legs_index_P3 = np.array([])
        touching_legs_P1 = np.array([])
        touching_legs_P2 = np.array([])
        touching_legs_P3 = np.array([])

        legs_c = np.array([
            self.J1.C[-1].to_list('xyz'),
            self.J2.C[-1].to_list('xyz'),
            self.J3.C[-1].to_list('xyz'),
            self.J4.C[-1].to_list('xyz')
        ])

        # get max distance to frame
        legs_z = legs_c[:, 2]
        ground1 = max(legs_z)

        # First pass to understand touching legs
        touching_legs_index = touching_legs_index_P1 = np.where(legs_z == ground1)
        # Create self.touching = [True, False, True, False] or similar
        touching_legs = touching_legs_P1 = np.isin(np.arange(4), touching_legs_index)
        nb_touching_legs = np.sum(touching_legs)

        a_pitch = 0.0
        a_roll = 0.0

        if nb_touching_legs == 1:  # TODO need to handle the case where 3 others legs are same height.
            print('[FIRST PASS 1 legs]')
            # We need to find the next touching legs
            m = np.ones(legs_z.size, dtype=bool)
            m[touching_legs_index] = False
            sub_legs = legs_z[m]

            # now find second highest point
            ground2 = np.max(sub_legs)
            # First pass to understand touching legs
            touching_legs_index_P2 = np.where(legs_z == ground2)
            # Create self.touching = [True, False, True, False] or similar
            touching_legs_P2 = np.isin(np.arange(4), touching_legs_index_P2)
            nb_touching_legs += np.sum(touching_legs_P2)
            touching_legs_index = np.unique(np.concatenate((touching_legs_index_P1[0], touching_legs_index_P2[0])))
            touching_legs = np.isin(np.arange(4), touching_legs_index)

        if nb_touching_legs == 2:
            if (touching_legs[0] == touching_legs[3]) \
                    or (touching_legs[1] == touching_legs[2]):
                print('[FIRST PASS 2 legs diag]')  # TODO 2 legs are not always at the same height in diagonal
            else:
                print('[FIRST PASS 2 legs no diag]')
                # We need to find the next touching legs
                m = np.ones(legs_z.size, dtype=bool)
                m[touching_legs_index] = False
                sub_legs = legs_z[m]

                # now find second highest point
                ground3 = np.max(sub_legs)
                # First pass to understand touching legs
                touching_legs_index_P3 = np.where(legs_z == ground3)
                # Create self.touching = [True, False, True, False] or similar
                touching_legs_P3 = np.isin(np.arange(4), touching_legs_index_P3)
                nb_touching_legs += np.sum(touching_legs_P3)
                if len(touching_legs_index_P2) == 0:
                    touching_legs_index = np.unique(np.concatenate(
                        (
                            touching_legs_index_P1,
                            touching_legs_index_P3
                        )
                    ))
                else:
                    touching_legs_index = np.unique(np.concatenate(
                        (
                            touching_legs_index_P1,
                            touching_legs_index_P2,
                            touching_legs_index_P3
                        )
                    ))
                touching_legs = np.isin(np.arange(4), touching_legs_index)

        if nb_touching_legs == 3:
            print('[FIRST PASS 3 legs]')
            # Compute plane vector
            sub_legs = legs_c[touching_legs_index]
            offsets = np.array([
                self.J1.structure_offset.to_list('xyz'),
                self.J2.structure_offset.to_list('xyz'),
                self.J3.structure_offset.to_list('xyz'),
                self.J4.structure_offset.to_list('xyz')
            ])
            sub_offset = offsets[touching_legs_index]

            # Compute cross vector to get plane vector
            v1 = np.subtract(
                np.add(sub_legs[1, :], sub_offset[1, :]),
                np.add(sub_legs[0, :], sub_offset[0, :])
            )
            v2 = np.subtract(
                np.add(sub_legs[2, :], sub_offset[2, :]),
                np.add(sub_legs[0, :], sub_offset[0, :])
            )

            plane = np.cross(v1, v2)

            a_pitch, a_roll = Utils.angle2ground(plane)

        if nb_touching_legs == 4:
            print('[FIRST PASS 4 legs]')
            # Means the robot is flat and no update for the legs
            pass
        self.touching_legs = touching_legs
        self.touching_legs_P1 = touching_legs_P1
        self.touching_legs_P2 = touching_legs_P2
        self.touching_legs_P3 = touching_legs_P3
        self.ground = np.array([ground1, ground2, ground3])
        return a_pitch, a_roll

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
            ground=self.ground[0]
        )
        self.J2.draw_legs(
            frame,
            location_x='left',
            location_y='bottom',
            touching=self.touching_legs[1],
            ground=self.ground[0]
        )
        self.J3.draw_legs(
            frame,
            location_x='right',
            location_y='top',
            touching=self.touching_legs[2],
            ground=self.ground[0]
        )
        self.J4.draw_legs(
            frame,
            location_x='left',
            location_y='top',
            touching=self.touching_legs[3],
            ground=self.ground[0]
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
