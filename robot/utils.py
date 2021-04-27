import cv2
from coordinates import Coordinate
import numpy as np
from numpy.linalg import norm


class Utils:
    ZOOM = 1800
    WIDTH = 1920
    HEIGHT = 1280
    FPS = 30
    HALF_HEIGHT = int(HEIGHT / 2)
    HALF_WIDTH = int(WIDTH / 2)

    # LEGS Settings
    LEG_OFFSET = 4 / 100

    # Colors
    black = (0, 0, 0)
    magenta = (255, 0, 255)
    green = (0, 255, 0)
    yellow = (5, 226, 252)
    red = (0, 0, 255)
    light_gray = (200, 200, 200)

    # Text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    text_thickness = 2

    # Drawing functions

    draw_offset_x = 0
    draw_offset_y = 0

    def ConvertX(p):
        return int((p + Utils.draw_offset_x) * Utils.ZOOM + Utils.HALF_WIDTH)

    def ConvertY(p):
        return int(((p + Utils.draw_offset_y) * Utils.ZOOM) + Utils.HALF_HEIGHT)

    def ConvertX_location(p, location):
        if location == 'right':
            return int(Utils.ConvertX(p) + (Utils.WIDTH / 3))
        elif location == 'left':
            return int(Utils.ConvertX(p) - (Utils.WIDTH / 3))

    def ConvertY_location(p, location):
        if location == 'bottom':
            return int(Utils.ConvertY(p) + (Utils.WIDTH / 4))
        elif location == 'top':
            return int(Utils.ConvertY(p) - (Utils.WIDTH / 4))

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
        return pitch * np.sign(np.cross(v_pitch, w_pitch)), roll * np.sign(np.cross(v_roll, w_roll))
