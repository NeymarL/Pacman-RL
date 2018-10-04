# Sarsa control with Q-value function approximation
# Policy evaluation: Q(s, a) <- Q(s, a) + alpha * (R + gamma * Q(s', a') - Q(s, a))
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

from src.base import BaseController
from src.config import Config, ControllerType

logger = getLogger(__name__)

class SarsaControl(BaseController):
    def __init__(self, env, config: Config):
        super().__init__()
        self.env = env
        self.epsilon = config.controller.epsilon
        self.gamma = config.controller.gamma
        self.model = self.build_model()
        self.max_workers = config.controller.max_workers

    def action(self, observation, predict=False, return_q=False):
        '''
        epsilon-greedy policy
        '''
        return self.epsilon_greedy_action(observation, predict, return_q)

    def update_q_value_on_batch(self, batch_history, batch_rewards):
        '''Sarsa evaluation

        Q(s, a) <- Q(s, a) + alpha * (R + gamma * Q(s', a') - Q(s, a))

        Args:
            batch_history = [[(s1, a1), (s2, a2), ...,(sT-1, aT-1)], ...]
            batch_rewards = [[r2, r3, ..., rT], ...]
        '''
        x = None
        y = None
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.build_training_set, history, rewards) 
                       for history, rewards in zip(batch_history, batch_rewards)]
        for future in futures:
            if x is None:
                x, y = future.result()
            else:
                inputs, targets = future.result()
                x = np.concatenate((x, inputs), axis=0)
                y = np.concatenate((y, targets), axis=0)
        self.model.train_on_batch(x, y)

    def build_training_set(self, history, rewards):
        '''
        history = [(s1, a1), (s2, a2), ...,(sT-1, aT-1)]
        rewards = [r2, r3, ..., rT]
        '''
        Q_ = dict()
        his1 = history.copy()
        his2 = history
        del(his2[0])
        his2.append((0, 0))
        inputs = np.zeros((len(rewards), ) + self.env.observation_space.shape)
        targets = np.zeros((len(rewards), self.env.action_space.n))

        for i, ((s, a), r, (s_, a_)) in enumerate(zip(his1, rewards, his2)):
            inputs[i] = np.array(s)
            targets[i] = self.model.predict(np.expand_dims(inputs[i], axis=0))
            if i + 1 == len(rewards):
                targets[i, a] = r
            else:
                if tuple(s_) not in Q_:
                    Q_[tuple(s_)] = self.model.predict(np.expand_dims(s_, axis=0))[0]
                targets[i, a] = r + self.gamma * Q_[tuple(s_)][a_]
        return inputs, targets
