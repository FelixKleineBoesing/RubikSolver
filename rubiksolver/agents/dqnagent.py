import tensorflow as tf
from tensorflow.python.keras.layers import Dense, LSTM
import os
import logging
import numpy as np
import random

from rubiksolver.replay_buffer import ReplayBuffer
from rubiksolver.agents.agent import Agent
from rubiksolver.helpers import min_max_scaling


class DQNAgent(Agent):
    """
    Usedd neural network to approximate the q table for q learning

    """

    def __init__(self, epsilon: float = 0.99, gamma: float = 0.99, buffer_size: int = 100000,
                 freq_turns_train: int = 500, freq_turns_load: int = 10000,
                 save_path: str = "../data/weights/dqn/model.ckpt", name: str = "Rubik Q Learning",
                 moving_average_length: int = 10, epsilon_decrease_factor: float = 0.999999):
        """
        Agent which implements Deep Q Learning

        :param epsilon: exploration factor. 1 means only exploration, while 0 means no exploration
        :param buffer_size: size of buffer class that stores states, actions, etc.
        :param freq_turns_train: frequency in which the network will be trained
        :param freq_turns_load: frequency in which the agent newtwork will be transferred to the target network
        :param save_path: path in which the network will be saved
        :param name: name of the model
        """
        super().__init__()
        # tensorflow related stuff
        self.name = name
        self._batch_size = 4096
        self._learning_rate = 0.3
        self.state_size = 3 * 3 * 6 * 6
        self.action_size = 12

        self._gamma = gamma
        self._epsilon_decrease_factor = epsilon_decrease_factor

        # loss logging
        self.td_loss_history = []
        self.moving_average_loss = []
        self.moving_average_length = moving_average_length

        # update frequencies
        self._freq_actions_train = freq_turns_train
        self._freq_turns_load = freq_turns_load

        # configure networks
        self.network = self._configure_network()
        self.target_network = self._configure_network()

        self.epsilon = epsilon
        self.exp_buffer = ReplayBuffer(buffer_size)

        self._save_path = save_path
        if os.path.isfile(self._save_path + ".index"):
            self.network.load_weights(self._save_path)

        # copy weight to target weights
        self.load_weigths_into_target_network()

    def _get_action(self, state: np.ndarray):
        """
        predicts the action based on the state

        :param state:
        :return:
        """
        if random.random() < self.epsilon:
            possible_actions = list(range(12))
            index = np.random.choice(possible_actions)
        else:
            state = _transform_state(state)
            state = state.reshape(1, state.shape[0])
            qvalues = self.network(state)
            index = np.argmax(qvalues)

        return int(index)

    def load_weigths_into_target_network(self):
        """ assign target_network.weights variables to their respective agent.weights values. """
        logging.debug("Transfer Weight!")
        logging.debug("Epsilon: {}".format(self.epsilon))
        self.network.save_weights(self._save_path)
        self.target_network.load_weights(self._save_path)

    def _configure_network(self):
        """
        initializes the network

        :return:
        """
        network = tf.keras.models.Sequential([
            Dense(512, activation="relu", input_shape=(self.state_size, )),
            # Dense(1024, activation="relu"),
            # Dense(2048, activation="relu"),
            # Dense(4096, activation="relu"),
            Dense(2048, activation="relu"),
            Dense(self.action_size, activation="linear")])

        self.optimizer = tf.optimizers.Adam(self._learning_rate)
        return network

    def _train_network(self, obs, actions, next_obs, rewards, is_done):
        """
        defines the loss functions and optimizes the weights

        :param obs: list of observations
        :param actions: list of actions
        :param next_obs: list of next_states
        :param rewards: list of rewards
        :param is_done: list of is_done
        :return:
        """

        # Decorator autographs the function
        @tf.function
        def td_loss():
            """
            temporal differences loss which should be minimized by the optimizer

            :return:
            """
            current_qvalues = self.network(obs)
            current_action_qvalues = tf.reduce_sum(tf.one_hot(actions, self.action_size) * current_qvalues, axis=1)

            next_qvalues_target = self.target_network(next_obs)
            next_state_values_target = tf.reduce_max(next_qvalues_target, axis=-1)
            reference_qvalues = rewards + self._gamma * next_state_values_target * (1 - is_done)
            return tf.reduce_mean(current_action_qvalues - reference_qvalues) ** 2

        with tf.GradientTape() as tape:
            loss = td_loss()

        grads = tape.gradient(loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))

        return loss

    def train_network(self):
        """
        trains the agent network based on the gathered samples from buffer

        :return:
        """
        logging.debug("Train Network!")
        loss_t = self._train_network(**self._sample_batch(batch_size=self._batch_size))
        self.td_loss_history.append(loss_t)
        self.moving_average_loss.append(np.mean([self.td_loss_history[max([0, len(self.td_loss_history) -
                                                                           self.moving_average_length]):]]))
        ma = self.moving_average_loss[-1]
        relative_ma = self.moving_average_loss[-1] / self._batch_size
        logging.info("Loss: {},     relative Loss: {}".format(ma, relative_ma))

    def _sample_batch(self, batch_size):
        """
        samples a batch from the buffer

        :param batch_size: size of the sample
        :return:
        """
        obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = self.exp_buffer.sample(batch_size)
        obs_batch = min_max_scaling(obs_batch)
        next_obs_batch = min_max_scaling(next_obs_batch)
        is_done_batch = is_done_batch.astype("float32")
        reward_batch = reward_batch.astype("float32")
        return {"obs": obs_batch, "actions": act_batch, "rewards": reward_batch,
                "next_obs": next_obs_batch, "is_done": is_done_batch}

    def get_feedback(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, next_action: float,
                     finished: bool):
        state = _transform_state(state)
        next_state = _transform_state(next_state)

        self.exp_buffer.add(state, action, reward, next_state, finished)

        if self.number_turns % self._freq_actions_train == 0:
            self.train_network()

        if self.number_turns % self._freq_turns_load == 0:
            self.load_weigths_into_target_network()

        self.epsilon *= self._epsilon_decrease_factor
        self.number_turns += 1


def _transform_state(state: np.ndarray):
    state = np.eye(6)[state]
    state = state.flatten()
    return state

