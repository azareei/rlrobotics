"""
Module Block

This module represent a block inside a Joint.

Attributes
----------
width : float
    Width of the block in meters
height : float
    Height of the block in meters
center : Coordinates
    Coordinates of the center of the block in the reference system of
    the Joint
anchor_d : float
    Distance to the side of the block to place the anchor
color : tuple
    Color of the block in BGR color (0-255, 0-255, 0-255)
type : str
    Type of block, can be 'middle' 'top' or 'bottom'

Methods
-------
__init__(self, _width, _height, _center, _anchor_d, _color, _type)
    Initialize a block with specific parameters
get_anchor(self, type)
    Compute and returns the coordinates of a specific anchor on the block
get_anchor_distance(self)
    Compute the offset distance between two parallel arms
set_position(self, _x, _y)
    Update the position of the block in Joint reference frame
draw(self, frame, offset, invert_y)
    Draw the block
"""
import cv2
from coordinates import Coordinate
from utils import Utils


class Block:
    def __init__(self, _width, _height, _center, _anchor_d, _color, _type):
        """
        Create a new block given width and height

        Parameters:
        -----------
        _width : float
            Width of the block in meters
        _height : float
            Height of the block in meters
        _center : Coordinates
            Coordinates of the center of the block in the reference system of
            the Joint
        _anchor_d : float
            Distance to the side of the block to place the anchor
        _color : tuple
            Color of the block in BGR color (0-255, 0-255, 0-255)
        _type : str
            Type of block, can be 'middle' 'top' or 'bottom'
        """
        self.width = _width
        self.height = _height
        self.center = _center
        self.anchor_d = _anchor_d
        self.color = _color
        self.type = _type

    def get_anchor(self, type):
        """
        Compute the coordinate of the bottom left or top left anchor location
        in Joint reference frame

        Parameters
        ----------
        type : str
            Can be 'b' for bottom or 't' for top. Will determine which anchor to
            compute.

        Returns
        -------
        Coordinates
            The computed coordinates of the left anchor position in Joint reference
            frame.
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
        Compute the distance between two parallels arms.

        Returns
        -------
        float
        """
        return self.width - (2 * self.anchor_d)

    def set_position(self, _x, _y):
        """
        Update the position of the center

        Parameters
        ----------
        _x : float
        _y : float
        """
        self.center.x = _x
        self.center.y = _y

    def draw(self, frame, offset, invert_y):
        """
        Method responsible to draw the arms.

        Parameters
        ----------
        frame : numpy Array
            Image of the current frame to draw on.
        offset : Coordinate
            Offset of the arms to the robot coordinates
        invert_y : bool
            If true, means that we the structure is reverted in y axis.

        Returns
        -------
        frame : numpy Array
            Updated image with the block on it.
        """
        inv = -1 if invert_y else 1
        block_thickness = -1

        if self.type == 'top':
            start_x = Utils.ConvertX(self.center.x - (self.width / 2) + offset.x)  # - (Utils.LEG_OFFSET / 2))
            end_x = Utils.ConvertX(self.center.x + (self.width / 2) + offset.x)
        elif self.type == 'middle':
            start_x = Utils.ConvertX(self.center.x - (self.width / 2) + offset.x)
            end_x = Utils.ConvertX(self.center.x + (self.width / 2) + offset.x)  # + (Utils.LEG_OFFSET / 2))
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

        return cv2.rectangle(
            frame,
            start,
            end,
            self.color,
            thickness=block_thickness
        )
