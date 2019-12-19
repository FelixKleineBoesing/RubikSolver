import numpy as np
import abc


class Agent(abc.ABC):

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_action(self, state: np.ndarray):
        pass

    @abc.abstractmethod
    def get_feedback(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, next_action: float,
                     finished: bool):
        pass
