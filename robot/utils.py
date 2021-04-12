import cv2
from coordinates import Coordinate


class Utils:
    ZOOM = 2000
    WIDTH = 1920
    HEIGHT = 1280
    FPS = 30
    HALF_HEIGHT = HEIGHT / 2
    HALF_WIDTH = WIDTH / 2

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

    def ConvertY(p):
        return int((p * Utils.ZOOM) + Utils.HALF_HEIGHT)

    def ConvertX(p):
        return int(p * Utils.ZOOM + Utils.HALF_WIDTH)

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
