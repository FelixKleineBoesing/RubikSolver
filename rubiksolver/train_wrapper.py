from types import FunctionType

from rubiksolver.cube import Cube, Direction, Side, ActionSerializer
from rubiksolver.agents.agent import Agent

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
            process. the arguments are the current number of shuffles and the number of iterations.
        :param actions_reset_threshold: the number of actions after which the cube will be randomly initialised again
        """
        if reward_function is None:
            self.rewards = {"solved": {True: 1000, False: 0}}
        if callback_number_shuffles is None:
            def callback_number_shuffles(number_shuffles, number_inner_iterations, all_iterations):
                return int(number_shuffles + number_shuffles * number_inner_iterations)

        self.rewards = reward_function
        self.cube = Cube()
        self.number_shuffles = number_shuffles
        self.callback_shuffles = callback_number_shuffles
        self.cube.init_random_cube(number_shuffles)
        self.actions_reset_threshold = actions_reset_threshold

    def run_training(self, agent: Agent, number_runs: int = 2e6):
        i = 0
        k = 0
        while i < number_runs:
            j = 0
            while not self.cube.solved and j < self.actions_reset_threshold:
                action = agent.get_action(self.cube.cube)
                state, action, reward, next_state, solved = self.take_action(action)
                if j > 0:
                    agent.get_feedback(state=last_state, action=last_action, reward=last_reward, next_state=state,
                                       next_action=action, finished=last_solved)

                last_state = state
                last_action = action
                last_reward = reward
                last_solved = solved
            i += j
            k += 1
            self.number_shuffles = self.callback_shuffles(self.number_shuffles, k, i)
            self.cube.init_random_cube(number_rotations=self.number_shuffles)



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
