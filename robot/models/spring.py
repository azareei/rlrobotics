"""
Module spring

This module represent a spring inside a Joint between two blocks

Attributes
----------
P : Coordinates
    Coordinates of the anchor on the bottom block for the  spring in Joint reference frame
Q : Coordinates
    Coordinates of the anchor on the top block for the  spring in Joint reference frame
k : float
    Stiffness of the spring
l_0 : float
    Initial length of the spring.

Methods
-------
__init__(self, _P, _Q)
    Initialize the spring
draw(self, frame, offset, invert_y)
    Draw the spring
"""
import cv2
from utils import Utils


class Spring:
    def __init__(self, _P, _Q):
        """
        Define a spring between two blocks.

        Parameters
        ----------
        _P : Coordinates
            Coordinates of the anchor on the bottom block for the  spring in Joint reference frame
        _Q : Coordinates
            Coordinates of the anchor on the top block for the  spring in Joint reference frame
        """
        self.P = _P
        self.Q = _Q
        self.k = 20  # N/m
        self.l_0 = 1/100

    def draw(self, frame, offset, invert_y):
        """
        Method responsible to draw the spring.

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
            Updated image with the spring on it.
        """
        inv = -1 if invert_y else 1
        spring_thickness = 3

        return cv2.line(
            frame,
            (
                Utils.ConvertX(self.P.x + offset.x),
                Utils.ConvertY(inv * (self.P.y) + offset.y)
            ),
            (
                Utils.ConvertX(self.Q.x + offset.x),
                Utils.ConvertY(inv * (self.Q.y) + offset.y)
            ),
            (0, 100, 100),
            thickness=spring_thickness
        )
