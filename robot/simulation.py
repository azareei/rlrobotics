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
        self.legs_video = self.init_video('{0}/legs/leg.mp4'.format(Path(__file__).resolve().parent))

        self.x = np.linspace(0, 0.044721 * 2, num=steps)

        # Forward pass
        for x_i in self.x:
            self.robot.update_position(x_i, forward=True)
            self.draw_blocks()
            self.draw_legs()

        # Backward pass
        self.x = np.linspace(0.044721 * 2, 0, num=steps)
        for x_i in self.x:
            self.robot.update_position(x_i, forward=False)
            self.draw_blocks()
            self.draw_legs()

        self.save_video(self.blocks_video)
        self.save_video(self.legs_video)

    def draw_blocks(self):
        # Draw blocks
        self.new_frame()
        self.robot.draw_blocks(self.frame)
        self.blocks_video.write(self.frame)

    def draw_legs(self):
        self.new_frame()
        self.robot.draw_legs(self.frame)
        self.legs_video.write(self.frame)

    def init_video(self, name):
        fourcc = VideoWriter_fourcc('m', 'p', '4', 'v')
        self.main_frame = np.ones((Utils.HEIGHT, Utils.WIDTH, 3), dtype=np.uint8) * 255
        return VideoWriter(name, fourcc, float(Utils.FPS), (Utils.WIDTH, Utils.HEIGHT))

    def new_frame(self):
        self.frame = self.main_frame.copy()

    def save_video(self, video):
        video.release()
