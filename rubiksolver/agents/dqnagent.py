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
