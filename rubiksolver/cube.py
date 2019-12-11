from aenum import Enum
import numpy as np


COLOR_DICT = {"white": 1, "yellow": 2, "red": 3, "blue": 4, "green": 5, "orange": 6}
COLOR_DICT_REV = {value: key for key, value in COLOR_DICT.items()}
# This dict is used to mark which sides are aside the key side, they are stored in clockwise direction inside the list
NEIGHBOR_DICT = {0: [2, 4, 3, 5], 1: [2, 4, 3, 5], 2: [0, 4, 1, 5], 3: [0, 4, 1, 5], 4: [0, 3, 1, 2], 5: [0, 3, 1, 2]}
# this dict marks which side is on which axis, 0 is x, 1 is y, 2 is z
AXIS_DICT = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
# this dict marks which side should turn which axis at what axis
TRANSITION_AXIS_DICT = {0: {1: 1, 2: 2}, 1: {1: {1: 1, 2: 2}, 2: {0: 1, 1: 2}},
                        3: {0: 1, 1: 2}, 4: {2: 1, 0: 2}, 5: {2: 1, 0: 2}}


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


class Cube:

    def __init__(self):
        self.cube = None
        self.init_cube()

    def init_cube(self):
        self.cube = np.array([[[i for _ in range(3)] for _ in range(3)] for i in range(1, 7)])

    def rotate(self, side: Side, direction: Direction):
        cube = self.cube.copy()
        k = 1 if direction.clockwise else 3
        cube[side.value, :, :] = np.rot90(cube[side.value, :, :], k=k)









if __name__ == "__main__":
    cube = Cube()
    cube.init_cube()
    cube.rotate(Side.front, Direction.clockwise)