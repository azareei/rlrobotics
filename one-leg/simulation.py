from models.joint import Joint
import numpy as np
from coordinates import Coordinate
from pathlib import Path
from cv2 import VideoWriter, VideoWriter_fourcc
import cv2
from utils import Utils


class Simulation:
    fig = None
    A = []
    B = []
    C = []

    def __init__(self):
        self.joint = Joint()
        # input movement

    def simulate(self):
        steps = 100
        self.init_video('{0}/blocks/out.mp4'.format(Path(__file__).resolve().parent))
        self.x = np.linspace(0, 3.46 / 100 * 4, num=steps)
        offset = 4 / 100
        for x_i, s in zip(self.x, range(steps)):
            print(s)
            self.joint.update_position(x_i, False)
            _A = Coordinate(x=self.joint.block_top.center.x - offset, y=0)
            _B = Coordinate(x=self.joint.block_mid.center.x, y=0)
            self.A.append(_A)
            self.B.append(_B)
            self.C.append(self.joint.compute_leg_height(_A, _B))
            self.draw()

        self.x = np.linspace(3.46 / 100 * 4, 0, num=steps)
        for x_i, s in zip(self.x, range(steps)):
            print(s)
            self.joint.update_position(x_i, True)
            _A = Coordinate(x=self.joint.block_top.center.x - offset, y=0)
            _B = Coordinate(x=self.joint.block_mid.center.x, y=0)
            self.A.append(_A)
            self.B.append(_B)
            self.C.append(self.joint.compute_leg_height(_A, _B))
            self.draw()

        self.save_video()
        self.gen_leg_animation()

    def draw(self):
        # Draw blocks
        self.new_frame()
        self.joint.block_bot.draw(self.frame)
        self.joint.block_mid.draw(self.frame)
        self.joint.block_top.draw(self.frame)

        # Draw bars
        self.joint.bars_bot.draw(self.frame)
        self.joint.bars_top.draw(self.frame)

        # Draw spring
        self.joint.spring_bot.draw(self.frame)
        self.joint.spring_top.draw(self.frame)
        self.video.write(self.frame)

    def gen_leg_animation(self):
        self.init_video('{0}/legs/leg.mp4'.format(Path(__file__).resolve().parent))
        for A, B, C in zip(self.A, self.B, self.C):
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
            self.video.write(self.frame)
        self.save_video()

    def init_video(self, name):
        fourcc = VideoWriter_fourcc('m','p','4','v')
        self.main_frame = np.ones((Utils.HEIGHT, Utils.WIDTH, 3), dtype=np.uint8) * 255
        self.video = VideoWriter(name, fourcc, float(Utils.FPS), (Utils.WIDTH, Utils.HEIGHT))

    def new_frame(self):
        self.frame = self.main_frame.copy()

    def save_video(self):
        self.video.release()
