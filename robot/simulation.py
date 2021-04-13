from models.robot import Robot
import numpy as np
from pathlib import Path
from coordinates import Coordinate
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
from utils import Utils
import time
import pandas as pd


class Simulation:
    def __init__(self):
        self.robot = Robot()
        self.camera_in_robot_ref = True

        # Initialize the videos
        self.blocks_video = self.init_video('{0}/blocks/out.mp4'.format(Path(__file__).resolve().parent))

        self.generate_actuation()
        self.generate_data_list()

    def simulate(self):
        start_time = time.time()
        for a_1, a_2, d_1, d_2, s in zip(self.actuation1,
                                         self.actuation2,
                                         self.actuation1_direction,
                                         self.actuation2_direction,
                                         range(len(self.actuation1))):

            if s % 20 == 0:
                print('step : {}'.format(s))
            self.robot.update_position(a_1, a_2, d_1, d_2)
            self.save_positions()
            self.draw_blocks()

        end_time = time.time()

        print("Simulation time : {0:.2f}s".format(end_time - start_time))

        self.save_video(self.blocks_video)

    def draw_blocks(self):
        # Draw blocks
        if self.camera_in_robot_ref:
            self.new_frame(self.robot.position)
            self.robot.draw(self.frame)
            self.blocks_video.write(self.frame)
        else:
            # Work in progress
            self.new_frame(Coordinate(x=0, y=0, z=0))
            Utils.draw_offset_x = self.robot.position.x
            Utils.draw_offset_y = self.robot.position.y
            self.robot.draw(self.frame)
            self.blocks_video.write(self.frame)

    def init_video(self, name):
        fourcc = VideoWriter_fourcc('m', 'p', '4', 'v')
        self.create_main_frame()
        return VideoWriter(name, fourcc, float(Utils.FPS), (Utils.WIDTH, Utils.HEIGHT))

    def new_frame(self, displacement):
        frame = self.main_frame.copy()
        if not self.camera_in_robot_ref:
            self.frame = frame
            return

        # Add grid
        max_coordinates = Utils.Pixel2Coordinate(Utils.WIDTH, Utils.HEIGHT)
        max_x = max_coordinates.x
        max_y = max_coordinates.y

        # Draw middle cross lines
        cv2.line(
            frame,
            (
                Utils.ConvertX(-max_x),
                Utils.ConvertY(0 - displacement.y)
            ),
            (
                Utils.ConvertX(max_x),
                Utils.ConvertY(0 - displacement.y)
            ),
            color=Utils.red,
            thickness=1
        )

        c_x = 0
        while c_x < (max_x + abs(displacement.x)):
            cv2.line(
                frame,
                (
                    Utils.ConvertX(c_x - displacement.x),
                    Utils.ConvertY(-max_y)
                ),
                (
                    Utils.ConvertX(c_x - displacement.x),
                    Utils.ConvertY(max_y)
                ),
                color=Utils.red if c_x == 0 else Utils.light_gray,
                thickness=1
            )

            cv2.line(
                frame,
                (
                    Utils.ConvertX(-c_x - displacement.x),
                    Utils.ConvertY(-max_y)
                ),
                (
                    Utils.ConvertX(-c_x - displacement.x),
                    Utils.ConvertY(max_y)
                ),
                color=Utils.red if c_x == 0 else Utils.light_gray,
                thickness=1
            )
            c_x += 5 / 100
        self.frame = frame

    def save_video(self, video):
        video.release()

    def create_main_frame(self):
        self.main_frame = np.ones((Utils.HEIGHT, Utils.WIDTH, 3), dtype=np.uint8) * 255

    def generate_actuation(self):
        # Get maximum actuation movement
        steps = 50
        max_1, max_2 = self.robot.max_actuation()
        self.actuation1_direction = np.concatenate(
            (np.zeros(steps), np.ones(steps)), axis=0
        ) < 1

        self.actuation2_direction = np.concatenate(
            (np.zeros(steps), np.ones(steps)), axis=0
        ) > 0

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

        self.actuation1 = np.tile(self.actuation1, 2)
        self.actuation2 = np.tile(self.actuation2, 2)
        self.actuation1_direction = np.tile(self.actuation1_direction, 2)
        self.actuation2_direction = np.tile(self.actuation2_direction, 2)
        self.steps = steps

    def create_dataframe(self):
        data = {}
        _cols = ['u', 'A', 'B', 'C', 'pos_x', 'pos_y']
        data['J1'] = pd.DataFrame(columns=_cols)
        data['J2'] = pd.DataFrame(columns=_cols)
        data['J3'] = pd.DataFrame(columns=_cols)
        data['J4'] = pd.DataFrame(columns=_cols)

        self.position_data = pd.concat(data, axis=1)