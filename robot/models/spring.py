import cv2
from utils import Utils


class Spring:
    def __init__(self, _P, _Q):
        """
        Define a spring between two blocks.
        """
        self.P = _P
        self.Q = _Q
        self.k = 20  # N/m
        self.l_0 = 1/100

    def draw(self, frame, offset):
        return cv2.line(
            frame,
            (
                int(Utils.ConvertX(self.P.x + offset.x)),
                int(Utils.ConvertY(self.P.y + offset.y))
            ),
            (
                int(Utils.ConvertX(self.Q.x + offset.x)),
                int(Utils.ConvertY(self.Q.y + offset.y))
            ),
            (0, 100, 255),
            thickness=3
        )
