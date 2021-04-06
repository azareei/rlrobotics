class Utils:
    ZOOM = 2000
    WIDTH = 1920
    HEIGHT = 1280
    FPS = 60

    magenta = (255, 0, 255)
    green = (0, 255, 0)
    yellow = (5, 226, 252)

    def ConvertY(p):
        return p * Utils.ZOOM + (Utils.HEIGHT / 2)

    def ConvertX(p):
        return p * Utils.ZOOM + (Utils.WIDTH / 2)

    def ConvertX_location(p, location):
        if location == 'right':
            return Utils.ConvertX(p) + (Utils.WIDTH / 4)
        elif location == 'left':
            return Utils.ConvertX(p) - (Utils.WIDTH / 4)

    def ConvertY_location(p, location):
        if location == 'bottom':
            return Utils.ConvertY(p) + (Utils.WIDTH / 4)
        elif location == 'top':
            return Utils.ConvertY(p) - (Utils.WIDTH / 4)
