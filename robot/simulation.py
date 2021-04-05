from models.joint import Joint
import numpy as np
from coordinates import Coordinate
from pathlib import Path
from cv2 import VideoWriter, VideoWriter_fourcc
import cv2
from utils import Utils


class Simulation:
    def __init__(self):
        self.joint = Joint('B', _structure_offset=Coordinate(x=10/100, y=10/100), _invert_y=True)
        # input movement

    def simulate(self):
        steps = 100
        # Initialize the videos
        self.blocks_video = self.init_video('{0}/blocks/out.mp4'.format(Path(__file__).resolve().parent))
        self.legs_video = self.init_video('{0}/legs/leg.mp4'.format(Path(__file__).resolve().parent))

        self.x = np.linspace(0, 0.044721 * 2, num=steps)

        # Forward pass
        for x_i in self.x:
            self.joint.update_position(x_i, True)
            self.draw_blocks()
            self.draw_legs()

        # Backward pass
        self.x = np.linspace(0.044721 * 2, 0, num=steps)
        for x_i in self.x:
            self.joint.update_position(x_i, False)
            self.draw_blocks()
            self.draw_legs()

        self.save_video(self.blocks_video)
        self.save_video(self.legs_video)

    def draw_blocks(self):
        # Draw blocks
        self.new_frame()

        self.joint.draw(self.frame)

        self.blocks_video.write(self.frame)

    def draw_legs(self):
        offset = 4 / 100
        A = Coordinate(x=self.joint.block_top.center.x - offset, y=0)
        B = Coordinate(x=self.joint.block_mid.center.x, y=0)
        C = self.joint.compute_leg_height(A, B)
        self.new_frame()
        self.frame = cv2.line(
            self.frame,
            (
                int(Utils.ConvertX(A.x)),
                int(Utils.ConvertY(A.y))
            ),
            (
                int(Utils.ConvertX(C.x)),
                int(Utils.ConvertY(C.y))
            ),
            (0, 255, 0),
            thickness=3
        )
        self.frame = cv2.line(
            self.frame,
            (
                int(Utils.ConvertX(B.x)),
                int(Utils.ConvertY(B.y))
            ),
            (
                int(Utils.ConvertX(C.x)),
                int(Utils.ConvertY(C.y))
            ),
            (255, 0, 0),
            thickness=3
        )
        self.legs_video.write(self.frame)

    def init_video(self, name):
        fourcc = VideoWriter_fourcc('m', 'p', '4', 'v')
        self.main_frame = np.ones((Utils.HEIGHT, Utils.WIDTH, 3), dtype=np.uint8) * 255
        return VideoWriter(name, fourcc, float(Utils.FPS), (Utils.WIDTH, Utils.HEIGHT))

    def new_frame(self):
        self.frame = self.main_frame.copy()

    def save_video(self, video):
        video.release()
