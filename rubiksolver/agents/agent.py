import numpy as np
import abc

from rubiksolver.cube import ActionSerializer
from rubiksolver.cube import Direction, Side


class Agent(abc.ABC):

    def __init__(self):
        self.reward_history = []
        self.average_reward_history = []
        self.number_turns = 0

    def get_action(self, state: np.ndarray) -> (Side, Direction):
        action = self._get_action(state=state)
        return ActionSerializer.deserialize(action=action)

    @abc.abstractmethod
    def _get_action(self, state: np.ndarray) -> int:
        """
        this function should return the action number (0 to 11)

        :param state:
        :return:
        """
        pass

    @abc.abstractmethod
    def get_feedback(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, next_action: float,
                     finished: bool) -> None:
        pass
