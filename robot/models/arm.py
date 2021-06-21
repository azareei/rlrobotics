"""
Module Arm

This module represent the two arms that links each blocks togethers.

Attributes
----------
low_anchor : Coordinates
    Correspond to the coordinates on the lower left of the block. The coordinates of the lower
    right are deducted during computation and drawing.
high_anchor : Coordinates
    Correspond to the coordinates on the top left of the block. The coordinates of the top
    right are deducted during computation and drawing.
length : float
    Store the length of the arm.
offset : Correspond to the offset to draw and compute the position of the right anchors.

Methods
-------
__init__(self, _low_anchor, _high_anchor, _length, _offset)
    Initialize an Arm
draw(self, frame, offset, invert_y)
    Draw the Arm
"""
import cv2
from utils import Utils


class Arm:
    def __init__(self, _low_anchor, _high_anchor, _length, _offset):
        """
        Initialize an arm with basics attributes.

        Parameters
        ----------
        _low_anchor : Coordinates
            Correspond to the coordinates on the lower left of the block. The coordinates of the lower
            right are deducted during computation and drawing.
        _high_anchor : Coordinates
            Correspond to the coordinates on the top left of the block. The coordinates of the top
            right are deducted during computation and drawing.
        _length : float
            Store the length of the arm.
        _offset : Correspond to the offset to draw and compute the position of the right anchors.
        """
        self.low_anchor = _low_anchor
        self.high_anchor = _high_anchor
        self.length = _length
        self.offset = _offset

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
            Updated image with the arms on it.
        """
        inv = -1 if invert_y else 1
        arm_thickness = 5

        frame = cv2.line(
            frame,
            (
                Utils.ConvertX(self.low_anchor.x + offset.x),
                Utils.ConvertY(inv * (self.low_anchor.y) + offset.y)
            ),
            (
                Utils.ConvertX(self.high_anchor.x + offset.x),
                Utils.ConvertY(inv * (self.high_anchor.y) + offset.y)
            ),
            Utils.red,
            thickness=arm_thickness
        )

        return cv2.line(
            frame,
            (
                Utils.ConvertX(self.low_anchor.x + self.offset + offset.x),
                Utils.ConvertY(inv * (self.low_anchor.y) + offset.y)
            ),
            (
                Utils.ConvertX(self.high_anchor.x + self.offset + offset.x),
                Utils.ConvertY(inv * (self.high_anchor.y) + offset.y)
            ),
            Utils.red,
            thickness=arm_thickness
        )
