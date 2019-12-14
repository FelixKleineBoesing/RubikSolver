import unittest
from rubiksolver.cube import Cube, Side, Direction
import random
import copy


all_sides = [Side.back, Side.bottom, Side.front, Side.left, Side.right, Side.top]
all_directions = [Direction.clockwise, Direction.counter_clockwise]


class CubeTester(unittest.TestCase):

    def test_naive(self):
        cube = Cube()
        initial_cube = copy.deepcopy(cube)

        reverse_directions = {Direction.counter_clockwise: Direction.clockwise,
                            Direction.clockwise: Direction.counter_clockwise}
        sides = [random.choice(all_sides) for _ in range(20)]
        reversed_sides = list(reversed(sides))
        directions = [random.choice(all_directions) for _ in range(20)]
        reversed_directions = [reverse_directions[dir_] for dir_ in directions]

        for direc, side in zip(directions, sides):
            cube.rotate(side, direc)

        for direc, side in zip(reversed_directions, reversed_sides):
            cube.rotate(side, direc)

        self.assertEqual(cube, initial_cube)