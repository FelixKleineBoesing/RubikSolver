import numpy as np
import abc

from rubiksolver.cube import ActionSerializer


class Agent(abc.ABC):

    @abc.abstractmethod
    def __init__(self):
        pass

    def get_action(self, state: np.ndarray) -> int:
        direction, side = self._get_action(state=state)
        return ActionSerializer.serialize(direction=direction, side=side)

    @abc.abstractmethod
    def _get_action(self, state: np.ndarray):
        pass

    @abc.abstractmethod
    def get_feedback(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, next_action: float,
                     finished: bool) -> None:
        pass
