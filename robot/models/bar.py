import cv2
from utils import Utils

class Bar:
    def __init__(self, _low_anchor, _high_anchor, _length, _offset):
        """
        Create a bar that will link two block,
        by default it will also create a second
        bar parallel to this one.
        """
        self.low_anchor = _low_anchor
        self.high_anchor = _high_anchor
        self.length = _length
        self.offset = _offset

    def draw(self, frame, offset, invert_y):
        if invert_y:
            inv = -1
        else:
            inv = 1

        frame = cv2.line(
            frame,
            (
                int(Utils.ConvertX(self.low_anchor.x + offset.x)),
                int(Utils.ConvertY(inv * (self.low_anchor.y) + offset.y))
            ),
            (
                int(Utils.ConvertX(self.high_anchor.x + offset.x)),
                int(Utils.ConvertY(inv * (self.high_anchor.y) + offset.y))
            ),
            (100, 100, 100),
            thickness=3
        )

        return cv2.line(
            frame,
            (
                int(Utils.ConvertX(self.low_anchor.x + self.offset + offset.x)),
                int(Utils.ConvertY(inv * (self.low_anchor.y) + offset.y))
            ),
            (
                int(Utils.ConvertX(self.high_anchor.x + self.offset + offset.x)),
                int(Utils.ConvertY(inv * (self.high_anchor.y) + offset.y))
            ),
            (100, 100, 100),
            thickness=3
        )
