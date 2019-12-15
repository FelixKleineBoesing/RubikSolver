import itertools
import unittest
from rubiksolver.cube import Cube, Side, Direction
import random
import copy
import numpy as np

all_sides = [Side.back, Side.bottom, Side.front, Side.left, Side.right, Side.top]
all_directions = [Direction.clockwise, Direction.counter_clockwise]

# Check bugs: Front seems to work
# There seems to be an error if different sides are involved


class CubeTester(unittest.TestCase):

    def test_naive(self):
        cube = Cube()
        initial_cube = copy.deepcopy(cube)
        number_rotations = 2000

        reverse_directions = {Direction.counter_clockwise: Direction.clockwise,
                              Direction.clockwise: Direction.counter_clockwise}
        sides = [random.choice(all_sides) for _ in range(number_rotations)]
        reversed_sides = list(reversed(sides))
        directions = [random.choice(all_directions) for _ in range(number_rotations)]
        reversed_directions = [reverse_directions[dir_] for dir_ in reversed(directions)]
        print("test")
        for direc, side in zip(directions, sides):
            cube.rotate(side, direc)

        for direc, side in zip(reversed_directions, reversed_sides):
            cube.rotate(side, direc)

        self.assertTrue(np.array_equal(cube.cube, initial_cube.cube))

    def test_each_side_alone(self):
        self._test_sides(Side.front)
        self._test_sides(Side.back)
        self._test_sides(Side.left)
        self._test_sides(Side.right)
        self._test_sides(Side.top)
        self._test_sides(Side.bottom)

    def test_two_sides(self):
        combinations = list(itertools.combinations_with_replacement(all_sides, 2))
        for index, combination in enumerate(combinations):
            try:
                self._test_sides(*combination)
            except AssertionError as e:
                print("AssertionError in combination: {} at run {} of {}".format(combination, index,
                                                                                 len(combinations)))
                raise AssertionError(e)

    def _test_sides(self, side: Side, *args):
        if len(args) == 0:
            sides = [side, side]
        else:
            sides = [side] + list(args)
        cube = Cube()
        initial_cube = copy.deepcopy(cube)

        reverse_directions = {Direction.counter_clockwise: Direction.clockwise,
                              Direction.clockwise: Direction.counter_clockwise}
        reversed_sides = list(reversed(sides))
        directions = [Direction.clockwise, Direction.clockwise]
        reversed_directions = [reverse_directions[dir_] for dir_ in reversed(directions)]

        first_iteration_cube = None
        for direc, side in zip(directions, sides):
            cube.rotate(side, direc)
            if not first_iteration_cube:
                first_iteration_cube = copy.deepcopy(cube)

        for direc, side in zip(reversed_directions, reversed_sides):
            cube.rotate(side, direc)

        self.assertTrue(np.array_equal(cube.cube, initial_cube.cube))


if __name__ == "__main__":
    CubeTester().test_naive()