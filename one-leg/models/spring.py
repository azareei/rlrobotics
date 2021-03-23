class Spring:
    def __init__(self, _P, _Q):
        """
        Define a spring between two blocks.
        """
        self.P = _P
        self.Q = _Q
        self.k = 20  # N/m
        self.l_0 = 1/100

    def draw(self, ax):
        _x = (self.P.x, self.Q.x)
        _y = (self.P.y, self.Q.y)
        ax.plot(_x, _y, 'fuchsia', linewidth=3)
