# Base controller

import gym
import numpy as np
import tensorflow as tf

from logging import getLogger
from collections import defaultdict
from keras.models import Sequential
from keras.layers import Dense, Flatten
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config import Config, ControllerType

logger = getLogger(__name__)

class BaseController:
    def __init__(self):
        self.model = None
        self.epsilon = 0

    def build_model(self):
        model = Sequential()
        model.add(Dense(self.env.action_space.n, input_shape=self.env.observation_space.shape, 
                             kernel_initializer='uniform', activation='linear'))
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        self.graph = tf.get_default_graph()
        return model

    def epsilon_greedy_action(self, observation, predict=False, return_q=False):
        '''epsilon-greedy policy

        Choose an action according to epsilon-greedy policy.

        Args:
            observation: An observation from the environment
            predict: Boolean value. Set true to become greedy policy (no random action)
            retuen_q: Boolean value. Set true to return the action as well as the original Q-value

        Return:
            The action choosed according to epsilon-greedy policy
        '''
        if np.random.rand() <= self.epsilon and not predict:
            a = self.env.action_space.sample()
        else:
            with self.graph.as_default():
                Q = self.model.predict(observation)
                a = np.argmax(Q)
                if return_q:
                    return (a, Q)
        return a

    def save(self, path):
        self.model.save_weights(path)
        logger.info(f"Save weight to {path}")

    def load(self, path):
        self.model.load_weights(path)
        self.graph = tf.get_default_graph()
        logger.info(f"Load weight from {path}")
