# Q-learning control with Q-value function approximation
# Policy evaluation: Q(s, a) <- Q(s, a) + 𝜶 * (R + 𝜸 maxQ(s', a') - Q(s, a))
# Policy improvement: epsilon-greedy exploration
# Q-value function approximation: Two-layer perception (input layer and output layer only)

import gym
import numpy as np
import tensorflow as tf

from logging import getLogger
from collections import defaultdict
from keras.models import Sequential
from keras.layers import Dense, Flatten
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.util import Buffer
from src.base import BaseController
from src.config import Config, ControllerType

logger = getLogger(__name__)


class QlearningControl(BaseController):
    def __init__(self, env, config: Config):
        super().__init__()
        self.env = env
        self.epsilon = config.controller.epsilon
        self.gamma = config.controller.gamma
        self.model = self.build_model()
        self.max_workers = config.controller.max_workers

    def build_training_set(self, buf):
        '''Q-learning evaluation

        Q(s, a) <- Q(s, a) + 𝜶 * (R + 𝜸 maxQ(s', a') - Q(s, a))

        Args:
            buf.states = [s1, s2, ..., sT-1]
            buf.actions = [a1, a2, ..., aT-1]
            buf.rewards = [r2, r3, ..., rT]

        Return:
            (inputs, targets): 
                inputs is a state list; 
                targets contains lists of action-values for each state in inputs
        '''
        Q_ = dict()
        history = [(s, a) for s, a in zip(buf.states, buf.actions)]
        his1 = history.copy()
        his2 = history
        del(his2[0])
        his2.append((0, 0))
        inputs = np.zeros((len(buf.rewards), ) +
                          self.env.observation_space.shape)
        targets = np.zeros((len(buf.rewards), self.env.action_space.n))

        for i, ((s, a), r, (s_, a_)) in enumerate(zip(his1, buf.rewards, his2)):
            inputs[i] = np.array(s)
            targets[i] = self.model.predict(np.expand_dims(inputs[i], axis=0))
            if i + 1 == len(buf.rewards):
                targets[i, a] = r
            else:
                if tuple(s_) not in Q_:
                    Q_[tuple(s_)] = self.model.predict(
                        np.expand_dims(s_, axis=0))[0]
                targets[i, a] = r + self.gamma * np.max(Q_[tuple(s_)])
        return inputs, targets
