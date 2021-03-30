from models.joint import Joint


class Robot:
    def __init__(self):
        self.L1 = Joint()
        self.L2 = Joint()
        self.L3 = Joint()
        self.L4 = Joint()