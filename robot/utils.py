import cv2


class Utils:
    ZOOM = 2000
    WIDTH = 1920
    HEIGHT = 1280
    FPS = 60

    black = (0, 0, 0)
    magenta = (255, 0, 255)
    green = (0, 255, 0)
    yellow = (5, 226, 252)
    red = (0, 0, 255)

    # Text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    text_thickness = 2

    def ConvertY(p):
        return p * Utils.ZOOM + int(Utils.HEIGHT / 2)

    def ConvertX(p):
        return p * Utils.ZOOM + int(Utils.WIDTH / 2)

    def ConvertX_location(p, location):
        if location == 'right':
            return Utils.ConvertX(p) + int(Utils.WIDTH / 3)
        elif location == 'left':
            return Utils.ConvertX(p) - int(Utils.WIDTH / 3)

    def ConvertY_location(p, location):
        if location == 'bottom':
            return Utils.ConvertY(p) + int(Utils.WIDTH / 4)
        elif location == 'top':
            return Utils.ConvertY(p) - int(Utils.WIDTH / 4)
