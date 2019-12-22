import logging

from rubiksolver.train_wrapper import TrainWrapper
from rubiksolver.agents.dqnagent import DQNAgent


def main():

    number_runs = 100000
    logging.getLogger().setLevel(logging.DEBUG)
    agent = DQNAgent(buffer_size=50000, freq_turns_train=4000, freq_turns_load=50000, epsilon=0)
    wrapper = TrainWrapper(number_shuffles=1)
    wrapper.run_training(agent, number_runs, logging_frequency=100)


if __name__ == "__main__":
    main()

