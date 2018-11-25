# Monte-Carlo Control with Q-value function approximation
# Policy evaluation: Q(s, a) <- Q(s, a) + (G - Q(s, a)) / N(s, a)
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


class MonteCarloControl(BaseController):
    def __init__(self, env, config: Config):
        super().__init__()
        self.env = env
        self.epsilon = config.controller.epsilon
        self.gamma = config.controller.gamma
        self.model = self.build_model()
        self.max_workers = config.controller.max_workers

    def build_training_set(self, buf):
        '''Monte-Carlo evaluation

        Q(s, a) <- Q(s, a) + (G - Q(s, a)) / N(s, a)

        Args:
            buf.states = [s1, s2, ..., sT-1]
            buf.actions = [a1, a2, ..., aT-1]
            buf.rewards = [r2, r3, ..., rT]

        Return:
            (inputs, targets): 
                inputs is a state list; 
                targets contains lists of action-values for each state in inputs
        '''
        N = defaultdict(int)
        Q = defaultdict(float)
        for i, (s, a) in enumerate(zip(buf.states, buf.actions)):
            s = tuple(s)
            N[(s, a)] += 1
            G = self.compute_return(buf.rewards, i)
            Q[(s, a)] += (G - Q[(s, a)]) / N[(s, a)]

        data = defaultdict(list)
        for (s, a), q in Q.items():
            data[s].append((a, q))

        inputs = np.zeros((len(data), ) + self.env.observation_space.shape)
        targets = np.zeros((len(data), self.env.action_space.n))
        for i, (s, item) in enumerate(data.items()):
            inputs[i] = np.array(s)
            targets[i] = self.model.predict(np.expand_dims(inputs[i], axis=0))
            for a, q in item:
                targets[i, a] = q
        return (inputs, targets)

    def compute_return(self, rewards, t):
        '''Compute return G_t

        G_t = R_t+1 + 𝜸R_t+2 + 𝜸^2R_t+3 + ... + 𝜸^(T-1)R_T

        Args:
            rewards = [r_1, r_2, ...., r_T]
            t: timestep
        '''
        G = 0
        for idx, j in enumerate(range(t, len(rewards))):
            G += (self.gamma ** idx) * rewards[j]
        return G
