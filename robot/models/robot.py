from models.joint import Joint
from coordinates import Coordinate


class Robot:
    def __init__(self):
        self.J1 = Joint('B', _structure_offset=Coordinate(x=20/100, y=20/100), _invert_y=True)
        self.J2 = Joint('A', _structure_offset=Coordinate(x=-20/100, y=20/100), _invert_y=True)
        self.J3 = Joint('B', _structure_offset=Coordinate(x=-20/100, y=-20/100))
        self.J4 = Joint('A', _structure_offset=Coordinate(x=20/100, y=-20/100))

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

    def draw_legs(self, frame):
        self.J1.draw_legs(frame)
        self.J2.draw_legs(frame)
        self.J3.draw_legs(frame)
        self.J4.draw_legs(frame)
