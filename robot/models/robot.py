from models.joint import Joint
from coordinates import Coordinate


class Robot:
    def __init__(self):
        self.J1 = Joint('B', _structure_offset=Coordinate(x=20/100, y=20/100))
        self.J2 = Joint('B', _structure_offset=Coordinate(x=-20/100, y=20/100)))
        self.J3 = Joint('B', _structure_offset=Coordinate(x=20/100, y=-20/100)))
        self.J4 = Joint('B', _structure_offset=Coordinate(x=-20/100, y=-20/100)))
