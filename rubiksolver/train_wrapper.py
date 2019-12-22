import logging
from types import FunctionType

from rubiksolver.cube import Cube, Direction, Side, ActionSerializer
from rubiksolver.agents.agent import Agent
from rubiksolver.agents.randomagent import RandomAgent


class TrainWrapper:
    """
    A Wrapper that lets an agent solve a agent and resets the cube to a random state if solved
    The Wrapper will also supply the agent feedback in terms of state, action, reward, next state, next action
    and finished
    """
    def __init__(self, reward_function: dict = None, number_shuffles: int = 50, actions_reset_threshold: int = 1000,
                 agent_callback: FunctionType = None, wrapper_callback: FunctionType = None):
        """


        :param reward_function: a dictionary that supplies the rewards for specific actions. At the moment there is
            only a reward for solving
        :param number_shuffles: the number of turns to be executed to init the cube
        :param wrapper_callback: a callback that gets the wrapper
        :param agent_callback: a callback that gets the agent
        :param actions_reset_threshold: the number of actions after which the cube will be randomly initialised again
        """
        if reward_function is None:
            reward_function = {"solved": {True: 1, False: 0}}

        self.rewards = reward_function
        self.cube = Cube()
        self.number_shuffles = number_shuffles
        self.wrapper_callback = wrapper_callback
        self.agent_callback = agent_callback
        self.cube.init_random_cube(number_shuffles)
        self.actions_reset_threshold = actions_reset_threshold
        self.all_iterations = None
        self.inner_iterations = None
        self.k = None
        self.number_runs = None
        self.init_stats()

    def init_stats(self):
        self.all_iterations = 0
        self.inner_iterations = 0
        self.k = 0
        self.number_runs = None

    def run_training(self, agent: Agent, number_runs: int = 2e6, logging_frequency: int = 10):
        """
        runs training for the number of runs. The supplied agent decides which action to take and receives feedback
        for his actions.

        :param agent: the agent that get feedback and decideds the
        :param number_runs: number of rotations that should be done. If a cube is solved it will be randomly
            initialised again
        :param logging_frequency: the logging frequency related to the number_runs. If number_runs is 1000 and
            logging_frequency is 10, every 100 runs the important stats will be logged
        :return:
        """
        self.init_stats()
        i = 0
        k = 0
        cum_reward = 0
        solved_cubes = 0
        while i < number_runs:
            j = 0
            last_state = None
            last_action = None
            last_reward = None
            last_solved = None
            while not self.cube.solved() and j < self.actions_reset_threshold:
                action = agent.get_action(self.cube.cube)
                state, action, reward, next_state, solved = self.take_action(action)
                if j > 0:
                    agent.get_feedback(state=last_state, action=last_action, reward=last_reward, next_state=state,
                                       next_action=action, finished=last_solved)

                last_state = state
                last_action = action
                last_reward = reward
                last_solved = solved
                cum_reward += reward

                if i % (number_runs / logging_frequency) == 0:
                    logging.info("Mean reward per rotation is {} at iteraton: {}".format(cum_reward / (i + 1), i))
                    logging.info("Solved {} of {} cubes, ratio: {}".format(solved_cubes, k, (solved_cubes/(k+1))*100))
                if solved:
                    agent.get_feedback(state=state, action=action, reward=reward, next_state=next_state,
                                       next_action=0, finished=solved)
                    solved_cubes += 1

                if self.wrapper_callback is not None:
                    self.wrapper_callback(self)
                if self.agent_callback is not None:
                    self.agent_callback(agent)

                j += 1
                i += 1
                self.all_iterations = i

            k += 1
            self.inner_iterations = k

            self.cube.init_random_cube(number_rotations=self.number_shuffles)

    def take_action(self, action):
        """
        Actions are decoded as the following

        :param action:
        :return:
        """
        side, direction = action
        state = self.cube.cube.copy()
        self.cube.rotate(side=side, direction=direction)

        solved = self.cube.solved()
        reward = self.rewards["solved"][solved]
        action = ActionSerializer.serialize(direction, side)
        return state, action, reward, self.cube.cube.copy(), solved


if __name__ == "__main__":
    def callback(number_shuffles, number_inner_iterations, all_iterations, number_runs):
        return number_shuffles
    logging.getLogger().setLevel(logging.DEBUG)
    wrapper = TrainWrapper(number_shuffles=2)
    wrapper.run_training(RandomAgent(), 2000000, logging_frequency=100)
