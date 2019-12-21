import tensorflow
import numpy as np

from rubiksolver.replay_buffer import ReplayBuffer
from rubiksolver.cube import Cube
from rubiksolver.cube import Direction, Side



class DQNAgent:

    def __init__(self, epsilon: float, buffer_size: int = 100000,  intervall_turns_train: int = 500,
                 intervall_turns_load: int = 10000, save_path: str = "../data/modeldata/q/model.ckpt",
                 caching: bool = False, name: str = "Rubik Q Learning"):
        """
        Agent which implements Q Learning
        :param state_shape: shape of state
        :param action_shape: shape of actions
        :param name: name of agent
        :param side: is he going to start at the top or at the bottom?
        :param epsilon: exploration factor
        """
        # tensorflow related stuff
        self.name = name
        self._batch_size = 64
        self._learning_rate = 0.3
        self._gamma = 0.99

        # calculate number actions from actionshape
        self._intervall_actions_train = intervall_turns_train
        self._intervall_turns_load = intervall_turns_load

        self.network = self._configure_network( self.name)
        self.target_network = self._configure_network("target_{}".format(name))

        self.epsilon = epsilon
        self.exp_buffer = ReplayBuffer(buffer_size)

        self._save_path = save_path
        if os.path.isfile(self._save_path + ".index"):
            self.network.load_weights(self._save_path)

        # copy weight to target weights
        self.load_weigths_into_target_network()

    def _train_network(self):
        pass

    def get_action(self, state: np.ndarray):
        actions = self.network(state)

    def load_weigths_into_target_network(self):
        """ assign target_network.weights variables to their respective agent.weights values. """
        logging.debug("Transfer Weight!")
        self.network.save_weights(self._save_path)
        self.target_network.load_weights(self._save_path)

    def _configure_network(self, state_shape: tuple, name: str):
        network = tf.keras.models.Sequential([
            Dense(512, activation="relu", input_shape=(multiply(*state_shape), )),
            # Dense(1024, activation="relu"),
            # Dense(2048, activation="relu"),
            # Dense(4096, activation="relu"),
            Dense(2048, activation="relu"),
            Dense(self.number_actions, activation="linear")])

        self.optimizer = tf.optimizers.Adam(self._learning_rate)
        return network

