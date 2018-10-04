# Forward-view Sarsa(𝝀) control with Q-value function approximation
# Policy evaluation: Q(s, a) <- Q(s, a) + 𝜶 * (q_t_𝝀 - Q(s, a))
#                    q_t_𝝀 = (1 - 𝝀) * ∑ 𝝀^(n - 1)q_t_n
#                    q_t_n = R_t+1 + 𝜸R_t+2 + 𝜸^2R_t+3 + ... + 𝜸^(n-1)R_t+n + 𝜸^nQ(s_t+n, a_t+n)
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

class SarsaLambdaControl(BaseController):
    def __init__(self, env, config: Config):
        super().__init__()
        self.env = env
        self.epsilon = config.controller.epsilon
        self.gamma = config.controller.gamma
        self.lambda_ = config.controller.lambda_
        self.model = self.build_model()
        self.max_workers = config.controller.max_workers
        
    def action(self, observation, predict=False, return_q=False):
        '''
        epsilon-greedy policy
        '''
        return self.epsilon_greedy_action(observation, predict, return_q)

    def build_training_set(self, history, rewards):
        '''Sarsa(𝝀) evaluation

        Q(s, a) <- Q(s, a) + 𝜶 * (q_t_𝝀 - Q(s, a))

        Args:
            history = [(s1, a1), (s2, a2), ...,(sT-1, aT-1)]
            rewards = [r2, r3, ..., rT]

        Return:
            (inputs, targets): 
                inputs is a state list; 
                targets contains lists of action-values for each state in inputs
        '''
        Q = dict()
        inputs = np.zeros((len(rewards), ) + self.env.observation_space.shape)
        targets = np.zeros((len(rewards), self.env.action_space.n))
        for t, ((s, a), r) in enumerate(zip(history, rewards)):
            inputs[t] = np.array(s)
            if tuple(s) not in Q:
                Q[tuple(s)] = self.model.predict(np.expand_dims(inputs[t], axis=0))
            targets[t] = Q[tuple(s)]
            targets[t, a] = self.q_lambda(t, rewards, history, Q)
        return inputs, targets

    def q_lambda(self, t, rewards, history, Q):
        '''Compute q_t_𝝀

        q_t_𝝀 = (1 - 𝝀) * ∑ 𝝀^(n - 1)q_t_n

        Args:
            t: timestep
            rewards = [r_1, r_2, ...., r_T]
            history = [(s1, a1), (s2, a2), ...,(sT-1, aT-1)]
            Q: A dict mapping s and Q(s), Q(s) is a list contains all action-values 
                of taking each action a from state s
        '''
        q_t_lambda = 0
        qtns = []
        for n in range(1, len(rewards) - t):
            # q_t_n = q_t_n-1 + 𝜸^n-1(r_t+n - Q(s_t+n-1, a_t+n-1)) + 𝜸^n * Q(s_t+n, a_t+n)
            if len(qtns) == 0:
                qtns.append(self.q_t_n(t, n, rewards, history, Q))
                qtn = qtns[n - 1]
            else:
                qtn1 = qtns[len(qtns) - 1]
                s_tn_1 = history[t + n - 1][0]
                a_tn_1 = history[t + n - 1][1]
                s_tn = history[t + n][0]
                a_tn = history[t + n][1]
                if tuple(s_tn) not in Q:
                    Q[tuple(s_tn)] = self.model.predict(np.expand_dims(s_tn, axis=0))[0]
                qtn = qtn1 - (self.gamma ** (n - 1)) * (rewards[t + n] - Q[tuple(s_tn_1)][a_tn_1]) \
                      + (self.gamma ** n) * Q[tuple(s_tn)][a_tn]
            q_t_lambda += (self.lambda_ ** (n - 1)) * qtn
        return q_t_lambda

    def q_lambda_slow(self, t, rewards, history, Q):
        '''Compute q_t_𝝀

        q_t_𝝀 = (1 - 𝝀) * ∑ 𝝀^(n - 1)q_t_n

        Args:
            t: timestep
            rewards = [r_1, r_2, ...., r_T]
            history = [(s1, a1), (s2, a2), ...,(sT-1, aT-1)]
            Q: A dict mapping s and Q(s), Q(s) is a list contains all action-values 
                of taking each action a from state s
        '''
        q_t_lambda = 0
        qtns = []
        for n in range(1, len(rewards) - t):
            q_t_lambda += (self.lambda_ ** (n - 1)) * self.q_t_n(t, n, rewards, history, Q)
        return q_t_lambda

    def q_t_n(self, t, n, rewards, history, Q):
        '''Compute q_t_n

        q_t_n = R_t+1 + 𝜸R_t+2 + 𝜸^2R_t+3 + ... + 𝜸^(n-1)R_t+n + 𝜸^n * Q(s_t+n, a_t+n)

        Args:
            t: timestep
            n: farest timestep
            rewards = [r_1, r_2, ...., r_T]
            history = [(s1, a1), (s2, a2), ...,(sT-1, aT-1)]
            Q: A dict mapping s and Q(s), Q(s) is a list contains all action-values 
                of taking each action a from state s
        '''
        G = 0
        for j in range(t, t + n):
            G += (self.gamma ** (j - t)) * rewards[j]
        if t + n < len(history):
            s_n = history[t + n][0]
            a_n = history[t + n][1]
            if tuple(s_n) not in Q:
                Q[tuple(s_n)] = self.model.predict(np.expand_dims(s_n, axis=0))[0]
            G += (self.gamma ** n) * Q[tuple(s_n)][a_n]
        return G

