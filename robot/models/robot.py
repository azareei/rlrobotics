from models.joint import Joint
from coordinates import Coordinate
import cv2
from utils import Utils
import numpy as np
import numpy.ma as ma


class Robot:
    def __init__(self, _J1, _J2, _J3, _J4, phase, reverse_actuation):
        # Actuation 1
        self.J1 = Joint(
            _J1['sequence'],
            _structure_offset=Coordinate(
                x=_J1['coordinates']['x'],
                y=_J1['coordinates']['y'],
                z=_J1['coordinates']['z']
            ),
            _invert_y=False,
            _invert_init_angle=False,
            _reverse_actuation=reverse_actuation,
            _bot_color=Utils.yellow,
            _top_color=Utils.magenta,
            _name='J1',
            _r1=_J1['r1'], _r2=_J1['r2'],
            _theta1=_J1['theta1'], _theta2=_J1['theta2']
        )
        self.J4 = Joint(
            _J4['sequence'],
            _structure_offset=Coordinate(
                x=_J4['coordinates']['x'],
                y=_J4['coordinates']['y'],
                z=_J4['coordinates']['z']
            ),
            _invert_y=True,
            _invert_init_angle=False,
            _reverse_actuation=reverse_actuation,
            _bot_color=Utils.yellow,
            _top_color=Utils.magenta,
            _name='J4',
            _r1=_J4['r1'], _r2=_J4['r2'],
            _theta1=_J4['theta1'], _theta2=_J4['theta2']
        )

        # Actuation 2
        self.J2 = Joint(
            _J2['sequence'],
            _structure_offset=Coordinate(
                x=_J2['coordinates']['x'],
                y=_J2['coordinates']['y'],
                z=_J2['coordinates']['z']
            ),
            _invert_y=False,
            _invert_init_angle=True if phase == 0 else False,
            _reverse_actuation=reverse_actuation,
            _bot_color=Utils.yellow,
            _top_color=Utils.green,
            _name='J2',
            _r1=_J2['r1'], _r2=_J2['r2'],
            _theta1=_J2['theta1'], _theta2=_J2['theta2']
        )
        self.J3 = Joint(
            _J3['sequence'],
            _structure_offset=Coordinate(
                x=_J3['coordinates']['x'],
                y=_J3['coordinates']['y'],
                z=_J3['coordinates']['z']
            ),
            _invert_y=True,
            _invert_init_angle=True if phase == 0 else False,
            _reverse_actuation=reverse_actuation,
            _bot_color=Utils.yellow,
            _top_color=Utils.green,
            _name='J3',
            _r1=_J3['r1'], _r2=_J3['r2'],
            _theta1=_J3['theta1'], _theta2=_J3['theta2']
        )

        self.position = []
        self.angle = []
        self.ground = 0.0  # Represent the high on the Centre of Gravity of the robot

    def update_position(self, actuation_1, actuation_2, actuation_1_dir, actuation_2_dir):
        mov1 = self.J1.update_position(actuation_1, actuation_1_dir)
        mov4 = self.J4.update_position(actuation_1, actuation_1_dir)

        mov2 = self.J2.update_position(actuation_2, actuation_2_dir)
        mov3 = self.J3.update_position(actuation_2, actuation_2_dir)

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
        self.angle.append([pitch, roll, yaw])

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
        touching_legs_P1 = np.array([False, False, False, False])
        touching_legs_P2 = np.copy(touching_legs_P1)
        touching_legs_P3 = np.copy(touching_legs_P1)

        legs_c = np.array([
            self.J1.C[-1].to_list('xyz'),
            self.J2.C[-1].to_list('xyz'),
            self.J3.C[-1].to_list('xyz'),
            self.J4.C[-1].to_list('xyz')
        ])

        a_pitch = 0.0
        a_roll = 0.0

        # get max distance to frame
        legs_z = legs_c[:, 2]
        ground1 = max(legs_z)

        # First pass to understand touching legs
        touching_legs_index = touching_legs_index_P1 = np.where(legs_z == ground1)[0]
        # Create self.touching = [True, False, True, False] or similar
        touching_legs = touching_legs_P1 = np.isin(np.arange(4), touching_legs_index)
        nb_touching_legs = np.sum(touching_legs)

        if nb_touching_legs == 1:  # TODO need to handle the case where 3 others legs are same height.
            # print('[FIRST PASS 1 legs]')
            # We need to find the next touching legs
            m = np.ones(legs_z.size, dtype=bool)
            m[touching_legs_index] = False
            sub_legs = legs_z[m]

            # now find second highest point
            ground2 = np.max(sub_legs)
            # First pass to understand touching legs
            touching_legs_index_P2 = np.where(legs_z == ground2)[0]

            # We check if next leg contains diagonal leg, if so keep only this one
            if len(touching_legs_index_P2) == 3:
                if np.isin(0, touching_legs_index) and np.isin(3, touching_legs_index_P2):
                    touching_legs_index_P2 = np.array([3])
                if np.isin(3, touching_legs_index) and np.isin(0, touching_legs_index_P2):
                    touching_legs_index_P2 = np.array([0])
                if np.isin(1, touching_legs_index) and np.isin(2, touching_legs_index_P2):
                    touching_legs_index_P2 = np.array([2])
                if np.isin(2, touching_legs_index) and np.isin(1, touching_legs_index_P2):
                    touching_legs_index_P2 = np.array([1])

            # Create self.touching = [True, False, True, False] or similar
            touching_legs_P2 = np.isin(np.arange(4), touching_legs_index_P2)
            nb_touching_legs += np.sum(touching_legs_P2)
            touching_legs_index = np.unique(np.concatenate((touching_legs_index_P1, touching_legs_index_P2)))
            touching_legs = np.isin(np.arange(4), touching_legs_index)

        if nb_touching_legs == 2:
            if (touching_legs[0] == touching_legs[3]) \
                    or (touching_legs[1] == touching_legs[2]):
                if touching_legs[0] or touching_legs[3]:
                    # print('[FIRST PASS 2 legs diag 1-4]')
                    v = np.subtract(
                        np.add(legs_c[0, :], self.J1.structure_offset.to_list('xyz')),
                        np.add(legs_c[3, :], self.J4.structure_offset.to_list('xyz'))
                    )

                else:
                    # print('[FIRST PASS 2 legs diag 2-3]')
                    v = np.subtract(
                        np.add(legs_c[1, :], self.J2.structure_offset.to_list('xyz')),
                        np.add(legs_c[2, :], self.J3.structure_offset.to_list('xyz'))
                    )
                v_pitch = np.array([v[0], v[2]])
                v_roll = np.array([v[1], v[2]])
                w = np.array([1, 0])
                v_pnorm = np.linalg.norm(v_pitch)
                v_rnorm = np.linalg.norm(v_roll)

                a_pitch = np.arccos(v_pitch.dot(w) / v_pnorm) * np.sign(np.cross(v_pitch, w))
                a_roll = np.arccos(v_roll.dot(w) / v_rnorm) * np.sign(np.cross(v_roll, w))

            else:
                # print('[FIRST PASS 2 legs no diag]')
                # We need to find the next touching legs
                m = np.ones(legs_z.size, dtype=bool)
                m[touching_legs_index] = False
                sub_legs = legs_z[m]

                # now find second highest point
                ground3 = np.max(sub_legs)
                # First pass to understand touching legs
                touching_legs_index_P3 = np.where(legs_z == ground3)[0]
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
            # print('[FIRST PASS 3 legs]')
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
            # print('[FIRST PASS 4 legs]')
            # Compute plane vector
            offset = np.array([
                self.J1.structure_offset.to_list('xyz'),
                self.J2.structure_offset.to_list('xyz'),
                self.J3.structure_offset.to_list('xyz'),
                self.J4.structure_offset.to_list('xyz')
            ])

            # Compute cross vector to get plane vector
            v1 = np.subtract(
                np.add(legs_c[1, :], offset[1, :]),
                np.add(legs_c[0, :], offset[0, :])
            )
            v2 = np.subtract(
                np.add(legs_c[2, :], offset[2, :]),
                np.add(legs_c[0, :], offset[0, :])
            )

            plane = np.cross(v1, v2)

            a_pitch, a_roll = Utils.angle2ground(plane)

        self.touching_legs = touching_legs
        self.touching_legs_P1 = touching_legs_P1
        self.touching_legs_P2 = touching_legs_P2
        self.touching_legs_P3 = touching_legs_P3
        self.update_ground(a_pitch, a_roll)

        # Need to place the angles inside -pi/2 -> pi/2
        a_pitch = Utils.angle_correction(a_pitch)
        a_roll = Utils.angle_correction(a_roll)

        # print(f"{a_pitch} {a_roll}")
        return a_pitch, a_roll

    def update_ground(self, pitch, roll):
        # First compute COG ground high
        concat = [self.J1, self.J2, self.J3, self.J4]
        self.ground = 0.0  # seems to always be 0.00
        medium = max(self.J1.C[-1].z, self.J2.C[-1].z, self.J3.C[-1].z, self.J4.C[-1].z)
        for leg, index in zip(self.touching_legs, range(4)):
            if leg:
                concat[index].ground_distance = concat[index].C[-1].z
            else:
                concat[index].ground_distance = medium  # TODO this need to be compupted

    def draw(self, frame):
        self.draw_joints(frame)
        self.draw_main_block(frame)
        self.draw_legs(frame)
        self.draw_angle(frame)
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
        )
        self.J2.draw_legs(
            frame,
            location_x='left',
            location_y='bottom',
            touching=self.touching_legs[1],
        )
        self.J3.draw_legs(
            frame,
            location_x='right',
            location_y='top',
            touching=self.touching_legs[2],
        )
        self.J4.draw_legs(
            frame,
            location_x='left',
            location_y='top',
            touching=self.touching_legs[3],
        )

    def draw_angle(self, frame):
        DISTANCE = 5 / 100
        angle = self.angle[-1]
        pitch = np.rad2deg(angle[0])
        pitch = min(abs(pitch), 10) * np.sign(pitch)
        roll = np.rad2deg(angle[1])
        roll = min(abs(roll), 10) * np.sign(roll)

        # Draw region
        # Start with a cross
        cv2.line(
            frame,
            (
                Utils.ConvertX_location(0, 'middle'),
                Utils.ConvertY_location(-DISTANCE, 'top')
            ),
            (
                Utils.ConvertX_location(0, 'middle'),
                Utils.ConvertY_location(DISTANCE, 'top')
            ),
            color=Utils.gray,
            thickness=2
        )
        p = (
            int(Utils.ConvertX_location(DISTANCE, 'middle')),
            int(Utils.ConvertY_location(-1 / 100, 'top'))
        )
        cv2.putText(
            frame,
            f'Pitch {pitch:.1f}',
            p,
            Utils.font,
            Utils.fontScale,
            Utils.gray,
            Utils.text_thickness,
            cv2.LINE_AA
        )
        cv2.line(
            frame,
            (
                Utils.ConvertX_location(-DISTANCE, 'middle'),
                Utils.ConvertY_location(0, 'top')
            ),
            (
                Utils.ConvertX_location(DISTANCE, 'middle'),
                Utils.ConvertY_location(0, 'top')
            ),
            color=Utils.gray,
            thickness=2
        )
        r = (
            int(Utils.ConvertX_location(1 / 100, 'middle')),
            int(Utils.ConvertY_location(-DISTANCE, 'top'))
        )
        cv2.putText(
            frame,
            f'Roll {roll:.1f}',
            r,
            Utils.font,
            Utils.fontScale,
            Utils.gray,
            Utils.text_thickness,
            cv2.LINE_AA
        )

        # Scale position from 5° to +5° from -5 to +5 cm
        cv2.circle(
            frame,
            (
                Utils.ConvertX_location(pitch / 100, 'middle'),
                Utils.ConvertY_location(roll / 100, 'top')
            ),
            5,
            color=Utils.green,
            thickness=-1
        )
        cv2.circle(
            frame,
            (
                Utils.ConvertX_location(0, 'middle'),
                Utils.ConvertY_location(0, 'top')
            ),
            Utils.ConvertCM2PX(2.5 / 100),
            color=Utils.gray,
            thickness=1
        )

        # SECOND REPRESENTATION PITCH
        start = (
            Utils.ConvertX_location(2*DISTANCE * np.cos(angle[0]), 'middle'),
            Utils.ConvertY_location(2*DISTANCE * np.sin(angle[0]), 'bottom')
        )

        end = (
            Utils.ConvertX_location(-2*DISTANCE * np.cos(angle[0]), 'middle'),
            Utils.ConvertY_location(-2*DISTANCE * np.sin(angle[0]), 'bottom')
        )

        cv2.line(
            frame,
            start,
            end,
            color=Utils.gray,
            thickness=4
        )

        # SECOND REPRESENTATION ROLL
        start = (
            Utils.ConvertX_location(2*DISTANCE * np.sin(angle[1]), 'right'),
            Utils.ConvertY_location(2*DISTANCE * np.cos(angle[1]), 'middle')
        )

        end = (
            Utils.ConvertX_location(-2*DISTANCE * np.sin(angle[1]), 'right'),
            Utils.ConvertY_location(-2*DISTANCE * np.cos(angle[1]), 'middle')
        )
        cv2.line(
            frame,
            start,
            end,
            color=Utils.gray,
            thickness=4
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
