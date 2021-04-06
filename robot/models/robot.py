from models.joint import Joint
from coordinates import Coordinate
import cv2
from utils import Utils


class Robot:
    def __init__(self):
        self.J2 = Joint(
            'B',
            _structure_offset=Coordinate(x=-20/100, y=5/100),
            _invert_y=False,
            bot_color=Utils.yellow,
            top_color=Utils.green
        )
        self.J3 = Joint(
            'B',
            _structure_offset=Coordinate(x=20/100, y=-5/100),
            _invert_y=True,
            bot_color=Utils.yellow,
            top_color=Utils.green
        )

        self.J1 = Joint(
            'B',
            _structure_offset=Coordinate(x=20/100, y=5/100),
            _invert_y=False,
            bot_color=Utils.yellow,
            top_color=Utils.magenta
        )
        self.J4 = Joint(
            'B',
            _structure_offset=Coordinate(x=-20/100, y=-5/100),
            _invert_y=True,
            bot_color=Utils.yellow,
            top_color=Utils.magenta
        )

    def update_position(self, x_i, forward):
        self.J1.update_position(x_i, forward)
        self.J2.update_position(x_i, forward)
        self.J3.update_position(x_i, forward)
        self.J4.update_position(x_i, forward)

    def draw_blocks(self, frame):
        self.J1.draw(frame)
        self.J2.draw(frame)
        self.J3.draw(frame)
        self.J4.draw(frame)
        self.draw_main_block(frame)

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
        return cv2.rectangle(
            frame,
            start,
            end,
            Utils.yellow,
            thickness=10
        )

    def draw_legs(self, frame):
        self.J1.draw_legs(frame)
        self.J2.draw_legs(frame)
        self.J3.draw_legs(frame)
        self.J4.draw_legs(frame)
