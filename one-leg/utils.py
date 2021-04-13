class Utils:
    ZOOM = 3000
    WIDTH = 1920
    HEIGHT = 1280
    FPS = 60

    def ConvertY(p):
        return p * Utils.ZOOM + (Utils.HEIGHT / 2)

    def ConvertX(p):
        return p * Utils.ZOOM + (Utils.WIDTH / 2)
