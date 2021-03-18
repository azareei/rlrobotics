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

    def draw(self, ax):
        _x = (self.low_anchor.x, self.high_anchor.x)
        _y = (self.low_anchor.y, self.high_anchor.y)
        ax.plot(_x, _y, 'coral')
        _x = (self.low_anchor.x + self.offset, self.high_anchor.x + self.offset)
        _y = (self.low_anchor.y, self.high_anchor.y)
        ax.plot(_x, _y, 'coral')
        
