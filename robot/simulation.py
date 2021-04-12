from models.robot import Robot
import numpy as np
from pathlib import Path
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
from utils import Utils
import time


class Simulation:
    def __init__(self):
        self.robot = Robot()

        steps = 50
        # Initialize the videos
        self.blocks_video = self.init_video('{0}/blocks/out.mp4'.format(Path(__file__).resolve().parent))

        self.actuation1_direction = np.concatenate(
            (np.zeros(steps), np.ones(steps)),
            axis=0
        ) < 1

        self.actuation2_direction = np.concatenate(
            (np.zeros(steps), np.ones(steps)),
            axis=0
        ) > 0

        self.actuation1 = np.concatenate(
            (
                np.linspace(0, 0.044721 * 2, num=steps),
                np.linspace(0.044721 * 2, 0, num=steps)
            ),
            axis=0
        )

        self.actuation2 = np.concatenate(
            (
                np.linspace(0, -0.044721 * 2, num=steps),
                np.linspace(-0.044721 * 2, 0, num=steps)
            ),
            axis=0
        )

        self.actuation1 = np.tile(self.actuation1, 2)
        self.actuation2 = np.tile(self.actuation2, 2)
        self.actuation1_direction = np.tile(self.actuation1_direction, 2)
        self.actuation2_direction = np.tile(self.actuation2_direction, 2)
        self.steps = steps

    def simulate(self):
        start_time = time.time()
        for a_1, a_2, d_1, d_2, s in zip(self.actuation1,
                                         self.actuation2,
                                         self.actuation1_direction,
                                         self.actuation2_direction,
                                         range(len(self.actuation1))):

            print('step : {}'.format(s))
            self.robot.update_position(a_1, a_2, d_1, d_2)
            self.draw_blocks()

        end_time = time.time()

        print("Simulation time : {0}s".format(end_time - start_time))

        self.save_video(self.blocks_video)

    def draw_blocks(self):
        # Draw blocks
        self.new_frame(self.robot.position)
        self.robot.draw(self.frame)
        self.blocks_video.write(self.frame)

    def init_video(self, name):
        fourcc = VideoWriter_fourcc('m', 'p', '4', 'v')
        self.create_main_frame()
        return VideoWriter(name, fourcc, float(Utils.FPS), (Utils.WIDTH, Utils.HEIGHT))

    def new_frame(self, displacement):
        frame = self.main_frame.copy()
        
        # Add grid
        max_coordinates = Utils.Pixel2Coordinate(Utils.WIDTH, Utils.HEIGHT)
        max_x = max_coordinates.x
        max_y = max_coordinates.y

        # Draw middle cross lines
        cv2.line(
            frame,
            (
                Utils.ConvertX(-max_x),
                Utils.ConvertY(displacement.y)
            ),
            (
                Utils.ConvertX(max_x),
                Utils.ConvertY(displacement.y)
            ),
            color=Utils.light_gray,
            thickness=1
        )

        c_x = 0
        while c_x < max_x:
            cv2.line(
                frame,
                (
                    Utils.ConvertX(c_x + displacement.x),
                    Utils.ConvertY(-max_y)
                ),
                (
                    Utils.ConvertX(c_x + displacement.x),
                    Utils.ConvertY(max_y)
                ),
                color=Utils.light_gray,
                thickness=1
            )

            cv2.line(
                frame,
                (
                    Utils.ConvertX(-c_x + displacement.x),
                    Utils.ConvertY(-max_y)
                ),
                (
                    Utils.ConvertX(-c_x + displacement.x),
                    Utils.ConvertY(max_y)
                ),
                color=Utils.light_gray,
                thickness=1
            )
            c_x += 5 / 100
        self.frame = frame

    def save_video(self, video):
        video.release()

    def create_main_frame(self):
        self.main_frame = np.ones((Utils.HEIGHT, Utils.WIDTH, 3), dtype=np.uint8) * 255

