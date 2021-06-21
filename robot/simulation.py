"""
Module simulation

This module handles a complete simulation with different parameters. Then it will store the results
of the simulation in different files.
Typically Generate those files:
A.png               -> Leg J1 pattern
AAAA-0.png          -> 4 legs pattern
AAAA-0.mp4          -> video of the simulation
AAAA-0_motion.png   -> robot's displacement plots
AAAA.csv            -> Data output in CSV format
AAAA.pkl            -> Data output in pickle format (useful for pandas and python)

Attributes
----------
camera_in_robot_ref : bool
    If true, the camera will stick to the reference frame of the robot and we will see the ground moving.
    If false, the camera will be fixed and the robot will move out of the frame.
actuation_steps : int
    Number of steps we want to compute for half a cycle.
nb_cycles : int
    Number of repetition of one cycle to compute. It can be usefull to do it longer to see a rotation better
draw : bool
    If true, drawing video will be done, otherwise drawing process will be skipped (faster execution)
phase_diff : int
    Can be 0 or 180. If 0, actuators will be in phase (extending at the same time), 180 will give actuators
    running in opposite phase.
reverse_actuation : bool
    Used to generate symmetry in results. It will reverse all the actuations.
mapping : bool
    If true, it will not produce any output. Used to run batch of simulations. (for example mapping.py)
grid_size : float
    Specify the grid size of the background (in meter)
robot : Robot
    Robot
actuation1_direction : numpy Array
    Array containing the direction for all the steps. 1 corresponding to forward motion, 0 for backward
actuation2_direction : numpy Array
    Array containing the direction for all the steps. 1 corresponding to forward motion, 0 for backward
actuation1 : numpy Array
    Array containing the position of the actuation in 2*steps
actuation2 : numpy Array
    Array containing the position of the actuation in 2*steps
blank_frame : numpy Array
    blank image that is copied to generate a new frame instead of creating a new one.
frame : numpy Array
    current frame to be draw.

Methods
-------
__init__(self, *params)
    Initialize the simulatio environment including the robot and the actuators
simulate(self)
    Run the simulation given the parameters
draw_blocks(self)
    draw the robot, the different views and the legs
init_video(self, name)
    initialize a new video file
new_frame(self displacement, yaw=0.0)
    generate a new frame to draw with the correct background orientation and displacement
save_video(self, video)
    save the video file
create_blank_frame(self)
    initialize the first frame
generate_actuation(self, phase, reverse=False)
    Create the arrays of actuations that will be used during the simulation.
get_joints_data(self, actuation, actuation_direction, joint)
    Gather the different information we need on the displacement and position to save them
save_data(self)
    Save all the datapoint generated during the simulation
plot_legs_motion(self)
    Export the sequence pattern for all the legs
plot_robot_motion(self)
    Export the total displacement in X and Y and the heading for the simulation.
"""
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from coordinates import Coordinate
from cv2 import VideoWriter, VideoWriter_fourcc
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from models.robot import Robot
from utils import Utils


class Simulation:
    def __init__(self, *params):
        """
        Initialize the simulation from the config files.

        Parameters
        ----------
        *params : dictionary
            Contain basically all the config file that were merged previously
        """
        s = params[0]['simulation']
        r = params[0]['robot']

        self.camera_in_robot_ref = s['camera_robot_ref']
        self.actuation_steps = s['actuation']['steps']
        self.nb_cycles = s['actuation']['cycles']
        self.draw = s['draw']
        self.phase_diff = s['actuation']['phase']
        self.reverse_actuation = s['actuation']['reverse']
        self.mapping = False
        self.camera_rotation = s['camera_rotation']
        self.grid_size = s['grid_size']

        self.robot = Robot(
            _J1=r['J1'], _J2=r['J2'],
            _J3=r['J3'], _J4=r['J4'],
            phase=self.phase_diff,
            reverse_actuation=self.reverse_actuation
        )

        if self.draw:
            # Initialize the videos
            self.blocks_video = self.init_video('{0}/results/{1}{2}{3}{4}-{5}.mp4'.format(
                Path(__file__).resolve().parent,
                self.robot.J1.sequence,
                self.robot.J2.sequence,
                self.robot.J3.sequence,
                self.robot.J4.sequence,
                self.phase_diff
            ))

        self.generate_actuation(self.phase_diff, self.reverse_actuation)

    def simulate(self):
        """
        Run the simulation

        Returns
        -------
            double
                Displacement X
            double
                Displacement Y
            double
                Heading (yaw)
        """
        start_time = time.time()
        for a_1, a_2, d_1, d_2, s in zip(self.actuation1,
                                         self.actuation2,
                                         self.actuation1_direction,
                                         self.actuation2_direction,
                                         range(len(self.actuation1))):

            if (s % 20 == 0) and (not self.mapping):
                print(f'step : {s}')
            self.robot.update_position(a_1, a_2, d_1, d_2)
            if self.draw:
                self.draw_blocks()

        end_time = time.time()

        seq = f'{self.robot.J1.sequence}{self.robot.J2.sequence}{self.robot.J3.sequence}{self.robot.J4.sequence}'
        print(f'Simulation time [{seq}] : {(end_time - start_time):.2f}s')

        if self.draw:
            self.save_video(self.blocks_video)

        if not self.mapping:
            self.save_data()
            self.plot_legs_motion()
            self.plot_robot_motion()

        return self.robot.position[-1].x, self.robot.position[-1].y, self.robot.angle[-1][2]

    def draw_blocks(self):
        """
        Draw the robot, the legs and the different views of the robot in a frame
        """
        # Draw blocks
        if self.camera_in_robot_ref:
            self.new_frame(self.robot.position[-1], self.robot.angle[-1][2])
            self.robot.draw(self.frame)
            self.blocks_video.write(self.frame)
        else:
            # Work in progress
            self.new_frame(Coordinate(x=0, y=0, z=0))
            Utils.draw_offset_x = self.robot.position[-1].x
            Utils.draw_offset_y = self.robot.position[-1].y
            self.robot.draw(self.frame)
            self.blocks_video.write(self.frame)

    def init_video(self, name):
        """
        Create a new video in MP4 format

        Parameters
        ----------
        name : str
            File's name
        
        Returns
        -------
        VideoWriter
        """
        fourcc = VideoWriter_fourcc('m', 'p', '4', 'v')
        self.create_blank_frame()
        return VideoWriter(name, fourcc, float(Utils.FPS), (Utils.WIDTH, Utils.HEIGHT))

    def new_frame(self, displacement, yaw=0.0):
        """
        Create a new frame and draw the grid inside. The grid will move given the displacement
        and the heading of the robot

        Parameters
        ----------
        displacement : Coordinates
            Coordinates of the robot
        yaw : float, optional
            The heading of the robot (zero by default)
        """
        frame = self.blank_frame.copy()
        if not self.camera_in_robot_ref:
            self.frame = frame
            return

        # Add grid
        max_coordinates = Utils.Pixel2Coordinate(Utils.WIDTH, Utils.HEIGHT)
        max_x = max_coordinates.x
        max_y = max_coordinates.y
        mid = Utils.Pixel2Coordinate(Utils.HALF_WIDTH, Utils.HALF_HEIGHT)

        c_x = 0
        while c_x < (max_x + abs(displacement.x)):
            if self.camera_rotation:
                px1, py1 = Utils.rotate_point(mid.x, mid.y, c_x - displacement.x, -max_y, yaw)
                px2, py2 = Utils.rotate_point(mid.x, mid.y, c_x - displacement.x, max_y, yaw)
            else:
                px1, py1 = c_x - displacement.x, -max_y
                px2, py2 = c_x - displacement.x, max_y

            cv2.line(
                frame,
                (
                    Utils.ConvertX(px1),
                    Utils.ConvertY(py1)
                ),
                (
                    Utils.ConvertX(px2),
                    Utils.ConvertY(py2)
                ),
                color=Utils.red if c_x == 0 else Utils.light_gray,
                thickness=1
            )

            if self.camera_rotation:
                px1, py1 = Utils.rotate_point(mid.x, mid.y, -c_x - displacement.x, -max_y, yaw)
                px2, py2 = Utils.rotate_point(mid.x, mid.y, -c_x - displacement.x, max_y, yaw)
            else:
                px1, py1 = -c_x - displacement.x, -max_y
                px2, py2 = -c_x - displacement.x, max_y

            cv2.line(
                frame,
                (
                    Utils.ConvertX(px1),
                    Utils.ConvertY(py1)
                ),
                (
                    Utils.ConvertX(px2),
                    Utils.ConvertY(py2)
                ),
                color=Utils.red if c_x == 0 else Utils.light_gray,
                thickness=1
            )
            c_x += self.grid_size

        c_y = 0
        while c_y < (max_y + abs(displacement.y)):
            if self.camera_rotation:
                px1, py1 = Utils.rotate_point(mid.x, mid.y, -max_x, c_y - displacement.y, yaw)
                px2, py2 = Utils.rotate_point(mid.x, mid.y, max_x, c_y - displacement.y, yaw)
            else:
                px1, py1 = -max_x, c_y - displacement.y
                px2, py2 = max_x, c_y - displacement.y

            cv2.line(
                frame,
                (
                    Utils.ConvertX(px1),
                    Utils.ConvertY(py1)
                ),
                (
                    Utils.ConvertX(px2),
                    Utils.ConvertY(py2)
                ),
                color=Utils.red if c_y == 0 else Utils.light_gray,
                thickness=1
            )

            if self.camera_rotation:
                px1, py1 = Utils.rotate_point(mid.x, mid.y, -max_x, -c_y - displacement.y, yaw)
                px2, py2 = Utils.rotate_point(mid.x, mid.y, max_x, -c_y - displacement.y, yaw)
            else:
                px1, py1 = -max_x, -c_y - displacement.y
                px2, py2 = max_x, -c_y - displacement.y

            cv2.line(
                frame,
                (
                    Utils.ConvertX(px1),
                    Utils.ConvertY(py1)
                ),
                (
                    Utils.ConvertX(px2),
                    Utils.ConvertY(py2)
                ),
                color=Utils.red if c_y == 0 else Utils.light_gray,
                thickness=1
            )
            c_y += self.grid_size

        self.frame = frame

    def save_video(self, video):
        video.release()

    def create_blank_frame(self):
        self.blank_frame = np.ones((Utils.HEIGHT, Utils.WIDTH, 3), dtype=np.uint8) * 255

    def generate_actuation(self, phase, reverse=False):
        """
        Create the different array of displacement for the artuators

        Parameters
        ----------
        phase : int
            Phase difference of the actuators, can be 0 or 180 (zero is both actuators extends at the same time)
        reverse : bool, optional
            Optional parameter to reverse the actuation. Used only to generate symmetric results.
        """
        # Get maximum actuation movement
        steps = self.actuation_steps
        max_1, max_2 = self.robot.max_actuation()
        if phase == 0:
            self.actuation1_direction = np.concatenate(
                (np.zeros(steps), np.ones(steps)), axis=0
            ) < 1

            self.actuation2_direction = np.concatenate(
                (np.zeros(steps), np.ones(steps)), axis=0
            ) < 1

            self.actuation1 = np.concatenate(
                (
                    np.linspace(0, max_1, num=steps),
                    np.linspace(max_1, 0, num=steps)
                ),
                axis=0
            )

            self.actuation2 = np.concatenate(
                (
                    np.linspace(0, -max_2, num=steps),
                    np.linspace(-max_2, 0, num=steps)
                ),
                axis=0
            )
        elif phase == 180:
            self.actuation1_direction = np.concatenate(
                (np.zeros(steps), np.ones(steps)), axis=0
            ) < 1

            self.actuation2_direction = np.concatenate(
                (np.zeros(steps), np.ones(steps)), axis=0
            ) < 1

            self.actuation1 = np.concatenate(
                (
                    np.linspace(0, max_1, num=steps),
                    np.linspace(max_1, 0, num=steps)
                ),
                axis=0
            )

            self.actuation2 = np.concatenate(
                (
                    np.linspace(0, max_2, num=steps),
                    np.linspace(max_2, 0, num=steps)
                ),
                axis=0
            )

        self.actuation1 = np.tile(self.actuation1, self.nb_cycles)
        self.actuation2 = np.tile(self.actuation2, self.nb_cycles)
        self.actuation1_direction = np.tile(self.actuation1_direction, self.nb_cycles)
        self.actuation2_direction = np.tile(self.actuation2_direction, self.nb_cycles)

        if reverse:
            t = self.actuation1
            self.actuation1 = self.actuation2
            self.actuation2 = t

    def get_joints_data(self, actuation, actuation_direction, joint):
        """
        Gather the data of a specific joint

        Parameters
        ----------
        actuation : numpy Array
            The associated actuation array for the joint
        actuation_direction : numpy Array
            The associated actuation_direction array to the joint
        joint : Joint
            The joint we want to gather data

        Results
        -------
        DataFrame
            A pandas dataframe that contains all the relevant data for the joint during the simulation
        """
        a_x, a_y, a_z = Utils.list_coord2list(joint.A)
        b_x, b_y, b_z = Utils.list_coord2list(joint.B)
        c_x, c_y, c_z = Utils.list_coord2list(joint.C)

        data = [
            actuation,
            actuation_direction,
            a_x, a_y, a_z,
            b_x, b_y, b_z,
            c_x, c_y, c_z
        ]

        df = pd.DataFrame(
            np.array(data).T,
            columns=[
                'u',
                'u_dir',
                'a_x', 'a_y', 'a_z',
                'b_x', 'b_y', 'b_z',
                'c_x', 'c_y', 'c_z'
            ]
        )

        return df

    def save_data(self):
        """
        Gather all the data from the 4 joints to save them in a pandas dataframe and save them after the simulation.
        Save in CSV and PKL format.
        """
        J1 = self.get_joints_data(self.actuation1, self.actuation1_direction, self.robot.J1)
        J2 = self.get_joints_data(self.actuation2, self.actuation1_direction, self.robot.J2)
        J3 = self.get_joints_data(self.actuation2, self.actuation1_direction, self.robot.J3)
        J4 = self.get_joints_data(self.actuation1, self.actuation1_direction, self.robot.J4)

        x, y, z = Utils.list_coord2list(self.robot.position)

        robot = pd.DataFrame(
            np.array([
                self.actuation1,
                self.actuation1_direction,
                self.actuation2,
                self.actuation2_direction,
                x, y, z,
                np.array(self.robot.angle)[:, 0],
                np.array(self.robot.angle)[:, 1],
                np.array(self.robot.angle)[:, 2]
            ]).T,
            columns=[
                'u1', 'u1_dir',
                'u2', 'u2_dir',
                'x', 'y', 'z',
                'pitch', 'roll', 'yaw'
            ]
        )

        self.data = {}
        self.data['J1'] = J1
        self.data['J2'] = J2
        self.data['J3'] = J3
        self.data['J4'] = J4
        self.data['robot'] = robot
        self.data = pd.concat(self.data, axis=1)

        self.data.to_csv('{0}/results/{1}{2}{3}{4}.csv'.format(
            Path(__file__).resolve().parent,
            self.robot.J1.sequence,
            self.robot.J2.sequence,
            self.robot.J3.sequence,
            self.robot.J4.sequence
        ))
        self.data.to_pickle('{0}/results/{1}{2}{3}{4}.pkl'.format(
            Path(__file__).resolve().parent,
            self.robot.J1.sequence,
            self.robot.J2.sequence,
            self.robot.J3.sequence,
            self.robot.J4.sequence
        ))

    def plot_legs_motion(self):
        """
        Generate the plot for the patterns of the legs during one cycle of the simulation.
        """
        fig, axs = plt.subplots(2, 2, figsize=(13, 10))
        cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())

        # J1
        u = abs(self.data['J1']['u'])
        x = self.data['J1']['c_x']
        z = self.data['J1']['c_z']

        j1_plot = axs[1, 1].scatter(x-x[0], z-z[0], c=u, cmap=cmap)
        axs[1, 1].set_xlabel('X [m]')
        axs[1, 1].set_ylabel('Z [m]')
        dx, dz = x[int(len(x) / (30 * self.nb_cycles))] - x[0], z[int(len(z) / (30 * self.nb_cycles))] - z[0]
        axs[1, 1].arrow(0, 0, dx, dz, width=1e-4, head_width=1e-3, color=(0, 0, 0, 0.4))
        axs[1, 1].title.set_text('J1')

        # J2
        u = abs(self.data['J2']['u'])
        x = self.data['J2']['c_x']
        z = self.data['J2']['c_z']

        j2_plot = axs[1, 0].scatter(x-x[0], z-z[0], c=u, cmap=cmap)
        axs[1, 0].set_xlabel('X [m]')
        axs[1, 0].set_ylabel('Z [m]')
        dx, dz = x[int(len(x) / (30 * self.nb_cycles))] - x[0], z[int(len(z) / (30 * self.nb_cycles))] - z[0]
        axs[1, 0].arrow(0, 0, dx, dz, width=1e-4, head_width=1e-3, color=(0, 0, 0, 0.4))
        axs[1, 0].title.set_text('J2')

        # J3
        u = abs(self.data['J3']['u'])
        x = self.data['J3']['c_x']
        z = self.data['J3']['c_z']

        j3_plot = axs[0, 1].scatter(x-x[0], z-z[0], c=u, cmap=cmap)
        axs[0, 1].set_xlabel('X [m]')
        axs[0, 1].set_ylabel('Z [m]')
        dx, dz = x[int(len(x) / (30 * self.nb_cycles))] - x[0], z[int(len(z) / (30 * self.nb_cycles))] - z[0]
        axs[0, 1].arrow(0, 0, dx, dz, width=1e-4, head_width=1e-3, color=(0, 0, 0, 0.4))
        axs[0, 1].title.set_text('J3')

        # J4
        u = abs(self.data['J4']['u'])
        x = self.data['J4']['c_x']
        z = self.data['J4']['c_z']

        j4_plot = axs[0, 0].scatter(x-x[0], z-z[0], c=u, cmap=cmap)
        axs[0, 0].set_xlabel('X [m]')
        axs[0, 0].set_ylabel('Z [m]')
        dx, dz = x[int(len(x) / (30 * self.nb_cycles))] - x[0], z[int(len(z) / (30 * self.nb_cycles))] - z[0]
        axs[0, 0].arrow(0, 0, dx, dz, width=1e-4, head_width=1e-3, color=(0, 0, 0, 0.4))
        axs[0, 0].title.set_text('J4')

        plt.colorbar(j1_plot, label='u [m]', ax=axs[0, 0])
        plt.colorbar(j2_plot, label='u [m]', ax=axs[0, 1])
        plt.colorbar(j3_plot, label='u [m]', ax=axs[1, 0])
        plt.colorbar(j4_plot, label='u [m]', ax=axs[1, 1])

        fig.suptitle('Robot legs pattern | sequence: {}{}{}{}-{}'.format(
            self.robot.J1.sequence,
            self.robot.J2.sequence,
            self.robot.J3.sequence,
            self.robot.J4.sequence,
            self.phase_diff
        ), fontsize=20)

        plt.savefig('{0}/results/{1}{2}{3}{4}-{5}.png'.format(
            Path(__file__).resolve().parent,
            self.robot.J1.sequence,
            self.robot.J2.sequence,
            self.robot.J3.sequence,
            self.robot.J4.sequence,
            self.phase_diff
        ))

        # J1
        u = abs(self.data['J1']['u'])
        x = self.data['J1']['c_x']
        z = self.data['J1']['c_z']

        # Uncomment to plot only leg 1
        import matplotlib.patches as mpatches
        _, axs = plt.subplots(1, 1, figsize=(8, 4))
        p = plt.scatter(x-x[0], z-z[0], c=u, cmap=cmap)
        plt.xlabel('X [m]')
        plt.ylabel('Z [m]')
        dx, dz = x[int(len(x) / (30 * self.nb_cycles))] - x[0], z[int(len(z) / (30 * self.nb_cycles))] - z[0]
        arrow = mpatches.FancyArrowPatch(
            (0, 0),
            (dx, dz),
            arrowstyle='simple',
            mutation_scale=20,
            fc=(0, 0, 0, 0.4),
            ec=(0, 0, 0, 0.4)
        )
        axs.add_patch(arrow)
        axs.set_xlim(left=-1e-3, right=None)
        axs.set_ylim(bottom=-1e-4, top=0.012)
        plt.title('Pattern sequence {}'.format(self.robot.J1.sequence))
        plt.colorbar(p, label='u [m]', ax=axs)
        plt.savefig('{0}/results/{1}.png'.format(
            Path(__file__).resolve().parent,
            self.robot.J1.sequence,
        ))

    def plot_robot_motion(self):
        """
        Generate the plot for the robot displacement during the simulation + the orientation of the robot
        """
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        # X Axis is shared among subplots
        t = np.arange(len(self.actuation1)) / (self.actuation_steps * 2)

        # X Position
        x = self.data['robot']['x']
        x = x - x[0]
        y = self.data['robot']['y']
        y = y - y[0]
        yaw = np.degrees(self.data['robot']['yaw'])
        yaw = yaw - yaw[0]
        max_text = 'max x : {:.1f}cm; max y : {:.1f}cm; max heading : {:.1f}°'.format(
            x[len(x) - 1] * 100,
            y[len(y) - 1] * 100,
            yaw[len(y) - 1]
        )

        axs[0].plot(t, x, 'g-')
        axs[0].set_ylabel('x [m]', color='g')

        # Limit y axis
        ymin, ymax = np.min(x), np.max(x)
        if abs(ymax) + abs(ymin) < 1e-2:
            mid = (ymin + ymax) / 2
            ymin = mid - 0.1
            ymax = mid + 0.1
        axs[0].set_ylim(ymin, ymax)

        ax2 = axs[0].twinx()

        ax2.plot(t, y, 'b-')
        ax2.set_ylabel('y [m]', color='b')

        # Limit y axis
        ymin, ymax = np.min(y), np.max(y)
        if abs(ymax) + abs(ymin) < 1e-2:
            mid = (ymin + ymax) / 2
            ymin = mid - 0.1
            ymax = mid + 0.1
        ax2.set_ylim(ymin, ymax)

        axs[1].plot(t, yaw, 'r-')
        axs[1].set_ylabel('heading [deg]', color='r')
        axs[1].grid()

        # Limit y axis
        ymin, ymax = np.min(yaw), np.max(yaw)
        if abs(ymax) + abs(ymin) < 5:
            mid = (ymin + ymax) / 2
            ymin = mid - 5
            ymax = mid + 5
        axs[1].set_ylim(ymin, ymax)

        if self.nb_cycles > 1:
            axs[1].set_xlabel('Cycles')
        else:
            axs[1].set_xlabel('Cycle')

        plt.title('Robot attitude - Seq: [{}{}{}{}] - Phase: {}°\n{}'.format(
            self.robot.J1.sequence,
            self.robot.J2.sequence,
            self.robot.J3.sequence,
            self.robot.J4.sequence,
            self.phase_diff,
            max_text
        ))
        plt.setp(axs[0].get_xticklabels(), visible=False)
        plt.setp(axs[1].get_xticklabels(), visible=True)

        plt.savefig('{0}/results/{1}{2}{3}{4}-{5}_motion.png'.format(
            Path(__file__).resolve().parent,
            self.robot.J1.sequence,
            self.robot.J2.sequence,
            self.robot.J3.sequence,
            self.robot.J4.sequence,
            self.phase_diff
        ))
