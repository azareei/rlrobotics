import cv2
from coordinates import Coordinate


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
