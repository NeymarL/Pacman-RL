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

    def build_model(self):
        model = Sequential()
        model.add(Dense(self.env.action_space.n, input_shape=self.env.observation_space.shape, 
                             kernel_initializer='uniform', activation='linear'))
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        self.graph = tf.get_default_graph()
        return model

    def save(self, path):
        self.model.save_weights(path)
        logger.info(f"Save weight to {path}")

    def load(self, path):
        self.model.load_weights(path)
        self.graph = tf.get_default_graph()
        logger.info(f"Load weight from {path}")
