import tensorflow
from rubiksolver.cube import Cube
from rubiksolver.cube import Direction, Side
import numpy as np


class DQNAgent:

    def __init__(self, gamma: float, epsilon: float):
        self.gamma = gamma
        self.epsilon = epsilon

    def _train_network(self):
        pass

    def get_action(self, state: np.ndarray):
        pass


class CubeWrapper:

    def __init__(self, reward_function: dict = None):
        if reward_function is None:
            self.rewards = {"solved": {True: 1000, False: 0}}
        self.rewards = reward_function
        self.cube = Cube()
        self.cube.init_random_cube(50)

    def solve(self, agent):
        pass


    def take_action(self, action: int = 0):
        assert 15 >= action >= 0
        assert isinstance(action, int)
        state = self.cube.cube.copy()
        direction = int(action / 6)
        side = action % 6
        assert 6 >= side >= 0
        if direction:
            direction = Direction.clockwise
        else:
            direction = Direction.counter_clockwise
        side = Side(side)
        self.cube.rotate(direction, side)

        solved = self.cube.solved()
        reward = self.rewards["solved"][solved]
        return state, action, reward, self.cube.cube.copy(), solved
