from models.block import Block
from coordinates import Coordinate

class Joint:
    """
    Represent a joint constituted by two blocs linked with two arms and a spring.
    """
    def __init__(self):
        self.block_top = Block(10/100, 5/100)
        self.block_bot = Block(10/100, 5/100)
        self.r = 3/100 # need to check that r is bigger the 
        self.P = Coordinate(x=0, y=0)
        self.k = 1 / 0.05
