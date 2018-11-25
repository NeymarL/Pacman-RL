# Base controller

import gym
import numpy as np
import tensorflow as tf

from logging import getLogger
from collections import defaultdict
from keras.models import Sequential
from keras.layers import Dense, Flatten
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.util import Buffer
from src.config import Config, ControllerType

logger = getLogger(__name__)


class BaseController:
    def __init__(self):
        self.model = None
        self.epsilon = 0
        self.env = None
        self.max_workers = 1

    def build_model(self):
        model = Sequential()
        model.add(Dense(self.env.action_space.n, input_shape=self.env.observation_space.shape,
                        kernel_initializer='uniform', activation='linear'))
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        self.graph = tf.get_default_graph()
        return model

    def action(self, observation, predict=False, return_q=False, epsilon=None):
        '''epsilon-greedy policy with epsilon decay

        Choose an action according to epsilon-greedy policy.

        Args:
            observation: An observation from the environment
            predict: Boolean value. Set true to become greedy policy (no random action)
            retuen_q: Boolean value. Set true to return the action as well as the original Q-value

        Return:
            The action choosed according to epsilon-greedy policy
        '''
        if not epsilon:
            # epsilon may decay
            epsilon = self.epsilon
        if np.random.rand() <= self.epsilon and not predict:
            a = self.env.action_space.sample()
        else:
            with self.graph.as_default():
                Q = self.model.predict(observation)
                a = np.argmax(Q)
                if return_q:
                    return (a, Q)
        return a

    def train(self, batch_buffers, i=None):
        '''Update parameters

        Args:
            batch_buffers = [buf1, buf2, ...]
        '''
        x = None
        y = None
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.build_training_set, buf)
                       for buf in batch_buffers]
            for future in futures:
                if x is None:
                    x, y = future.result()
                else:
                    inputs, targets = future.result()
                    x = np.concatenate((x, inputs), axis=0)
                    y = np.concatenate((y, targets), axis=0)
        self.model.train_on_batch(x, y)

    def build_training_set(self, buf):
        raise NotImplementedError

    def save(self, path):
        self.model.save_weights(path)
        logger.info(f"Save weight to {path}")

    def load(self, path):
        self.model.load_weights(path)
        self.graph = tf.get_default_graph()
        logger.info(f"Load weight from {path}")
