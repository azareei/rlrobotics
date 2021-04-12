from coordinates import Coordinate
import cv2
from utils import Utils


class Block:
    def __init__(self, _width, _height, _center, _anchor_d, _color, _type):
        """
        Create a new block given width and height

        Parameters:
        -----------

        width : double
            Represent the width of the block in meters
        height : double
            Represent the height of the block in meters
        _center : Coordinate
            Represent the center of mass of the block
        _d : double
            Represent the distance between the anchor point 
            and the border of the block
        """
        self.width = _width
        self.height = _height
        self.center = _center
        self.anchor_d = _anchor_d
        self.color = _color
        self.type = _type

    def get_anchor(self, type):
        """
        Returns the coordinate of the bottom left anchor location from the block
        type is a string, can be
            t for top anchors
            b for bottom anchors
        """
        if type == 'b':
            _x = self.center.x - (self.width / 2) + self.anchor_d
            _y = self.center.y - (self.height / 2) + self.anchor_d
        elif type == 't':
            _x = self.center.x - (self.width / 2) + self.anchor_d
            _y = self.center.y + (self.height / 2) - self.anchor_d
        return Coordinate(x=_x, y=_y)

    def get_anchor_distance(self):
        """
        Returns the distance offset for the two parallel bar of the block
        """
        return self.width - (2 * self.anchor_d)

    def set_position(self, _x, _y):
        self.center.x = _x
        self.center.y = _y

    def draw(self, frame, offset, invert_y):
        if invert_y:
            inv = -1
        else:
            inv = 1

        if self.type == 'top':
            start_x = Utils.ConvertX(self.center.x - (self.width / 2) + offset.x - Utils.LEG_OFFSET)
            end_x = Utils.ConvertX(self.center.x + (self.width / 2) + offset.x)
        else:
            start_x = Utils.ConvertX(self.center.x - (self.width / 2) + offset.x)
            end_x = Utils.ConvertX(self.center.x + (self.width / 2) + offset.x)

        start = (
            start_x,
            Utils.ConvertY(inv * (self.center.y - (self.height / 2)) + offset.y)
        )
        end = (
            end_x,
            Utils.ConvertY(inv * (self.center.y + (self.height / 2)) + offset.y)
        )
        return cv2.rectangle(frame, start, end, self.color, thickness=-1)
