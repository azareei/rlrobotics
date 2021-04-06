from models.joint import Joint
from models.robot import Robot
import numpy as np
from coordinates import Coordinate
from pathlib import Path
from cv2 import VideoWriter, VideoWriter_fourcc
from utils import Utils


class Simulation:
    def __init__(self):
        self.robot = Robot()

    def simulate(self):
        steps = 100
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

        for a_1, a_2, d_1, d_2, s in \
                zip(self.actuation1, self.actuation2, self.actuation1_direction, self.actuation2_direction, range(2*steps)):
            print('step : {}'.format(s))

            self.robot.update_position(a_1, a_2, d_1, d_2)
            self.draw_blocks()

        self.save_video(self.blocks_video)

    def draw_blocks(self):
        # Draw blocks
        self.new_frame()
        self.robot.draw_blocks(self.frame)
        self.blocks_video.write(self.frame)

    def init_video(self, name):
        fourcc = VideoWriter_fourcc('m', 'p', '4', 'v')
        self.main_frame = np.ones((Utils.HEIGHT, Utils.WIDTH, 3), dtype=np.uint8) * 255
        return VideoWriter(name, fourcc, float(Utils.FPS), (Utils.WIDTH, Utils.HEIGHT))

    def new_frame(self):
        self.frame = self.main_frame.copy()

    def save_video(self, video):
        video.release()
