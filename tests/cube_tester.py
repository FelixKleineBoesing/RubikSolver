import unittest
from rubiksolver.cube import Cube, Side, Direction
import random
sides = [Side.back, Side.bottom, Side.front, Side.left, Side.right, Side.top]


class CubeTester(unittest.TestCase):

    def test_naive(self):
        cube = Cube()
        sides = [random.choice(side) for side in sides]




