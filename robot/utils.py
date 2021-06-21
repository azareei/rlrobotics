"""
Module Utils

Contains different variables and methods to facilitate the works in different parts of the project.
Attributes are documented through the code directly
"""
import collections

import cv2
import numpy as np
from coordinates import Coordinate
from numpy.linalg import norm


class Utils:
    ZOOM = 1800  # Zoom level for the video. the smaller it is, the smaller the robot looks
    WIDTH = 1920  # Video frame's width (pxl)
    HEIGHT = 1280  # Video frame's height (pxl)
    FPS = 30  # Number of frame per second in the video file

    HALF_HEIGHT = int(HEIGHT / 2)
    HALF_WIDTH = int(WIDTH / 2)

    PI = np.pi
    HALF_PI = np.pi / 2

    # Anchor offset on the top block of a joint to attach the leg
    # Otherwise cannot produce change in height
    LEG_OFFSET = 4 / 100

    # Colors (Blue Green Red)
    black = (0, 0, 0)
    magenta = (255, 0, 255)
    green = (0, 255, 0)
    yellow = (5, 226, 252)
    red = (0, 0, 255)
    light_gray = (200, 200, 200)
    gray = (100, 100, 100)
    blue = (255, 5, 5)

    # Text settings
    font = cv2.FONT_HERSHEY_PLAIN
    fontScale = 1
    text_thickness = 2

    # Drawing functions (used to draw the robot when camera not in robot's reference frame, half-deprecated)
    draw_offset_x = 0
    draw_offset_y = 0

    # Method to convert a variable in centimeter to a number of pixels
    def ConvertCM2PX(d):
        return int(d * Utils.ZOOM)

    # Method to convert a pixel value to a coordinate
    def Pixel2Coordinate(_x, _y):
        return Coordinate(
            x=(_x - Utils.HALF_WIDTH) / Utils.ZOOM,
            y=(_y - Utils.HALF_HEIGHT) / Utils.ZOOM
        )

    # Method to convert a position x in meter to a position in the frame
    def ConvertX(p):
        return int((p + Utils.draw_offset_x) * Utils.ZOOM + Utils.HALF_WIDTH)

    # Method to convert a y position in meter to a position in the frame
    def ConvertY(p):
        return int(((p + Utils.draw_offset_y) * Utils.ZOOM) + Utils.HALF_HEIGHT)

    # Method to convert a x position in meter to a position in the frame with a specification of the location
    def ConvertX_location(p, location):
        if location == 'right':
            return int(Utils.ConvertX(p) + (Utils.WIDTH / 3))
        elif location == 'left':
            return int(Utils.ConvertX(p) - (Utils.WIDTH / 3))
        elif location == 'middle':
            return Utils.ConvertX(p)

    # Method to convert a y position in meter to a position in the frame with a specification of the location
    def ConvertY_location(p, location):
        if location == 'bottom':
            return int(Utils.ConvertY(p) + (Utils.WIDTH / 4))
        elif location == 'top':
            return int(Utils.ConvertY(p) - (Utils.WIDTH / 4))
        elif location == 'middle':
            return Utils.ConvertY(p)

    # Method to convert a list of coordinates to a double axis list
    def list_coord2list(list_coordinates):
        """
        Convert a list(Coordinates) to 3 list of axis coordinate
        x[], y[] and z[]
        """
        x = [c.x for c in list_coordinates]
        y = [c.y for c in list_coordinates]
        z = [c.z for c in list_coordinates]

        return x, y, z

    # Method to compute the pich and roll angle to a ground place
    def angle2ground(v):
        w_roll = np.array([0, 1])
        w_pitch = np.array([0, 1])

        v_roll = np.array([v[1], v[2]])
        v_pitch = np.array([v[0], v[2]])

        roll = np.arccos(v_roll.dot(w_roll) / (norm(v_roll) * norm(w_roll)))
        pitch = np.arccos(v_pitch.dot(w_pitch) / (norm(v_pitch) * norm(w_pitch)))
        if np.isnan(roll):
            roll = 0.0
            print("roll nan")
        if np.isnan(pitch):
            pitch = 0.0
            print("pitch nan")
        return pitch * np.sign(np.cross(v_pitch, w_pitch)), roll * np.sign(np.cross(v_roll, w_roll))

    # Method to correct an angle between -pi/2 and pi/2
    def angle_correction(angle):
        if abs(angle) > Utils.HALF_PI:
            return np.sign(angle) * ((abs(angle) % Utils.PI) - Utils.PI)
        else:
            return angle

    def dict_merge(dct, merge_dct):
        """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
        updating only top-level keys, dict_merge recurses down into dicts nested
        to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
        ``dct``. There is no overrite from default file. It preserve existing keys
        :param dct: dict onto which the merge is executed
        :param merge_dct: dct merged into dct
        :return: None
        """
        for k, v in merge_dct.items():
            if (k in dct and isinstance(dct[k], dict)
                    and isinstance(merge_dct[k], collections.Mapping)):
                Utils.dict_merge(dct[k], merge_dct[k])
            else:
                if k not in dct.keys():
                    dct[k] = merge_dct[k]

    # Method to rotate any point around any other point in 2D with a specific angle
    def rotate_point(origin_x, origin_y, p_x, p_y, angle):
        s = np.sin(angle)
        c = np.cos(angle)

        # Translate point to origin
        p_x -= origin_x
        p_y -= origin_y

        # Rotate point
        xnew = p_x * c - p_y * s
        ynew = p_x * s + p_y * c

        # Translate point back
        return xnew + origin_x, ynew + origin_y
