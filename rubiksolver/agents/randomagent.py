import numpy as np

from rubiksolver.agents.agent import Agent
from rubiksolver.cube import Direction, Side


class RandomAgent(Agent):

    def __init__(self):
        super().__init__()

    def _get_action(self, state: np.ndarray):
        return np.random.choice(Direction), np.random.choice(Side)

    def get_feedback(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, next_action: float,
                     finished: bool):
        pass
