from aenum import Enum
import numpy as np
import random

COLOR_DICT = {"white": 1, "yellow": 2, "red": 3, "blue": 4, "green": 5, "orange": 6}
COLOR_DICT_REV = {value: key for key, value in COLOR_DICT.items()}
# This dict is used to mark which sides are aside the key side, they are stored in clockwise direction inside the list
NEIGHBOR_DICT = {0: [2, 4, 3, 5], 1: [2, 4, 3, 5], 2: [0, 4, 1, 5], 3: [0, 4, 1, 5], 4: [0, 3, 1, 2], 5: [0, 3, 1, 2]}
# this dict is for specifying which index should be taken
INDEX_DICT = {0: 0, 1: 2, 2: 0, 3: 2, 4: 0, 5: 2}
# this dict marks which side is on which axis, 0 is x, 1 is y, 2 is z
AXIS_DICT = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
# this dict marks which side should turn which axis at what axis
TRANSITION_AXIS_DICT = {0: {1: 1, 2: 2}, 1: {1: 1, 2: 2}, 2: {2: 1, 0: 2},
                        3: {2: 1, 0: 2}, 4: {0: 1, 1: 2}, 5: {0: 1, 1: 2}}


class Side(Enum):
    top = 0
    bottom = 1
    left = 2
    right = 3
    front = 4
    back = 5


class Direction(Enum):
    clockwise = 0
    counter_clockwise = 1


all_sides = [Side.back, Side.bottom, Side.front, Side.left, Side.right, Side.top]
all_directions = [Direction.clockwise, Direction.counter_clockwise]


class Cube:

    def __init__(self):
        self.cube = None
        self.init_cube()

    def init_cube(self):
        """
        initialises a solved cube

        :return:
        """
        self.cube = np.array([[[i for _ in range(3)] for _ in range(3)] for i in range(1, 7)])

    def init_random_cube(self, number_rotations: int = 50):
        """
        initialises a solved cube

        :return:
        """
        self.cube = np.array([[[i for _ in range(3)] for _ in range(3)] for i in range(1, 7)])
        sides = [random.choice(all_sides) for _ in range(number_rotations)]
        directions = [random.choice(all_directions) for _ in range(number_rotations)]
        for side, direc in zip(sides, directions):
            self.rotate(side, direc)

    def rotate(self, side: Side, direction: Direction):
        """
        rotates the specified side of this cube by the specified direction

        :param side:
        :param direction:
        :return:
        """
        cube = self.cube.copy()
        k = 1 if direction is direction.clockwise else 3
        cube[side.value, :, :] = np.rot90(cube[side.value, :, :], k=k)
        neighbors = NEIGHBOR_DICT[side.value]
        if direction is Direction.counter_clockwise:
            neighbors = list(reversed(neighbors))
        for i, n in enumerate(neighbors):
            side_before = get_neighbor_before(i, neighbors)
            axis_after = AXIS_DICT[n]
            index = INDEX_DICT[side.value]
            axis_to_copy_to = TRANSITION_AXIS_DICT[side.value][axis_after]
            axis_before = AXIS_DICT[side_before]
            axix_to_copy_from = TRANSITION_AXIS_DICT[side.value][axis_before]
            LI = [index, index, index]
            LI[0] = n
            LI[axis_to_copy_to] = [0, 1, 2]
            RI = [index, index, index]
            RI[0] = side_before
            RI[axix_to_copy_from] = [0, 1, 2]

            cube[tuple(LI)] = self.cube[tuple(RI)]
        self.cube = cube

    def solved(self):
        """
        indicates whether the cube is solved or not

        :return:
        """
        solved = True
        for i in range(self.cube.shape[0]):
            solved = solved & (np.sum(self.cube[i, :, :] == i + 1) == 9)
            if not solved:
                break
        return solved

    def __str__(self):
        for i in self.cube.shape[0]:
            print(self.cube[i, :, :])


def get_neighbor_before(index: int, neighbors: list):
    """
    return the neighbor that is before the current side.
    ATTENTION! List neighbords must be ordered in the right order
    Example:
        # >>> neighbors = [1, 2, 3, 4]
        # >>> index = 2
        # >>> print(get_neighbor_before(index, neighbors))

    :param index: the index of the side in neighbords list
    :param neighbors: list of the neighbors. the neighbor that is specified by index must be a part of neighbors
    :return:
    """
    assert isinstance(neighbors, list)
    assert isinstance(index, int)
    assert index >= 0
    assert index < len(neighbors), "index must be smaller than the length of the list neighbors"
    if index == 0:
        return neighbors[len(neighbors) - 1]
    else:
        return neighbors[index - 1]


if __name__ == "__main__":
    cube = Cube()
    cube.init_cube()
    cube.rotate(Side.front, Direction.clockwise)
    cube.rotate(Side.top, Direction.clockwise)
    print(cube.cube)