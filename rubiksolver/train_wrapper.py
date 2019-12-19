from types import FunctionType

from rubiksolver.cube import Cube, Direction, Side, ActionSerializer

class TrainWrapper:
    """
    A Wrapper that lets an agent solve a agent and resets the cube to a random state if solved
    The Wrapper will also supply the agent feedback in terms of state, action, reward, next state, next action
    and finished
    """
    def __init__(self, reward_function: dict = None, number_shuffles: int = 50,
                 callback_number_shuffles: FunctionType = None, actions_reset_threshold: int = 1000):
        """


        :param reward_function: a dictionary that supplies the rewards for specific actions. At the moment there is
            only a reward for solving
        :param number_shuffles: the number of turns to be executed to init the cube
        :param callback_number_shuffles: a callback function that may change the number of shuffles during training 
            process. the arguments are the current number of shuffles and the number of iteration.
        :param actions_reset_threshold: the number of actions after which the cube will be randomly initialised again
        """
        if reward_function is None:
            self.rewards = {"solved": {True: 1000, False: 0}}
        self.rewards = reward_function
        self.cube = Cube()
        self.number_shuffles = number_shuffles
        self.callback_shuffles = callback_number_shuffles
        self.cube.init_random_cube(number_shuffles)

    def solve(self, agent):
        pass

    def take_action(self, action: int = 0):
        """
        Actions are decoded as the following

        :param action:
        :return:
        """
        side, direction = ActionSerializer.deserialize(action)
        state = self.cube.cube.copy()
        self.cube.rotate(direction, side)

        solved = self.cube.solved()
        reward = self.rewards["solved"][solved]
        return state, action, reward, self.cube.cube.copy(), solved


if __name__ == "__main__":
    wrapper = TrainWrapper()
