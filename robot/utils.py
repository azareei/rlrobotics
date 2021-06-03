import cv2
from coordinates import Coordinate
import numpy as np
from numpy.linalg import norm
import collections


class Utils:
    ZOOM = 1800
    WIDTH = 1920
    HEIGHT = 1280
    FPS = 30
    HALF_HEIGHT = int(HEIGHT / 2)
    HALF_WIDTH = int(WIDTH / 2)
    PI = np.pi
    HALF_PI = np.pi / 2

    # LEGS Settings
    LEG_OFFSET = 4 / 100

    # Colors
    black = (0, 0, 0)
    magenta = (255, 0, 255)
    green = (0, 255, 0)
    yellow = (5, 226, 252)
    red = (0, 0, 255)
    light_gray = (200, 200, 200)
    gray = (100, 100, 100)

    # Text settings
    font = cv2.FONT_HERSHEY_PLAIN
    fontScale = 1
    text_thickness = 2

    # Drawing functions

    draw_offset_x = 0
    draw_offset_y = 0

    def ConvertCM2PX(d):
        return int(d * Utils.ZOOM)

    def ConvertX(p):
        return int((p + Utils.draw_offset_x) * Utils.ZOOM + Utils.HALF_WIDTH)

    def ConvertY(p):
        return int(((p + Utils.draw_offset_y) * Utils.ZOOM) + Utils.HALF_HEIGHT)

    def ConvertX_location(p, location):
        if location == 'right':
            return int(Utils.ConvertX(p) + (Utils.WIDTH / 3))
        elif location == 'left':
            return int(Utils.ConvertX(p) - (Utils.WIDTH / 3))
        elif location == 'middle':
            return Utils.ConvertX(p)

    def ConvertY_location(p, location):
        if location == 'bottom':
            return int(Utils.ConvertY(p) + (Utils.WIDTH / 4))
        elif location == 'top':
            return int(Utils.ConvertY(p) - (Utils.WIDTH / 4))
        elif location == 'middle':
            return Utils.ConvertY(p)

    def Pixel2Coordinate(_x, _y):
        return Coordinate(
            x=(_x - Utils.HALF_WIDTH) / Utils.ZOOM,
            y=(_y - Utils.HALF_HEIGHT) / Utils.ZOOM
        )

    # General utilization
    def list_coord2list(list_coordinates):
        """
        Convert a list(Coordinates) to 3 list of axis coordinate
        x[], y[] and z[]
        """
        x = [c.x for c in list_coordinates]
        y = [c.y for c in list_coordinates]
        z = [c.z for c in list_coordinates]

        return x, y, z

    def angle2ground(v):
        """
        Compute pitch and roll angle to a ground plane
        """

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

    def angle_correction(angle):
        """
        Correct a input angle to an angle between
        -pi/2 and pi/2
        """
        if abs(angle) > Utils.HALF_PI:
            return np.sign(angle) * ((abs(angle) % Utils.PI) - Utils.PI)
        else:
            return angle

    def dict_merge(dct, merge_dct):
        """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
        updating only top-level keys, dict_merge recurses down into dicts nested
        to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
        ``dct``.
        :param dct: dict onto which the merge is executed
        :param merge_dct: dct merged into dct
        :return: None
        """
        for k, v in merge_dct.items():
            if (k in dct and isinstance(dct[k], dict)
                    and isinstance(merge_dct[k], collections.Mapping)):
                Utils.dict_merge(dct[k], merge_dct[k])
            else:
                dct[k] = merge_dct[k]
