from models.joint import Joint
import numpy as np


class Simulation:
    def __init__(self):
        self.joint = Joint()

    def simulate(self):
        raise NotImplementedError
