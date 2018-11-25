# Sarsa(𝝀) control with Q-value function approximation
# Policy evaluation:
#   Forward-view:
#       Q(s, a) <- Q(s, a) + 𝜶 * (q_t_𝝀 - Q(s, a))
#       q_t_𝝀 = (1 - 𝝀) * ∑ 𝝀^(n - 1)q_t_n
#       q_t_n = R_t+1 + 𝜸R_t+2 + 𝜸^2R_t+3 + ... + 𝜸^(n-1)R_t+n + 𝜸^n * Q(s_t+n, a_t+n)
#   Backward-view:
#       Q(s, a) <- Q(s, a) + 𝜶 * (R + 𝜸Q(s', a') - Q(s, a))E_t(s, a)
#       E_0(s, a) = 0
#       E_t(s, a) = 𝜸𝝀E_t-1(s, a) + 1(S_t = s, A_t = a)
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
        self.forward = config.controller.forward

    def build_training_set(self, buf):
        if self.forward:
            return self.build_training_set_forward(buf)
        else:
            return self.build_training_set_backward(buf)

    def build_training_set_forward(self, buf):
        '''Forward-view Sarsa(𝝀) evaluation

        Q(s, a) <- Q(s, a) + 𝜶 * (q_t_𝝀 - Q(s, a))

        Args:
            buf.states = [s1, s2, ..., sT-1]
            buf.actions = [a1, a2, ..., aT-1]
            buf.rewards = [r2, r3, ..., rT]

        Return:
            (inputs, targets): 
                inputs is a state list; 
                targets contains lists of action-values for each state in inputs
        '''
        Q = dict()
        inputs = np.zeros((len(buf.rewards), ) +
                          self.env.observation_space.shape)
        targets = np.zeros((len(buf.rewards), self.env.action_space.n))
        history = [(s, a) for s, a in zip(buf.states, buf.actions)]
        for t, (s, a, r) in enumerate(zip(buf.states, buf.actions, buf.rewards)):
            inputs[t] = np.array(s)
            if tuple(s) not in Q:
                Q[tuple(s)] = self.model.predict(
                    np.expand_dims(inputs[t], axis=0))
            targets[t] = Q[tuple(s)]
            targets[t, a] = self.q_lambda(t, buf.rewards, history, Q)
        return inputs, targets

    def build_training_set_backward(self, buf):
        '''Backward-view Sarsa(𝝀) evaluation

        Q(s, a) <- Q(s, a) + 𝜶 * (R + 𝜸Q(s', a') - Q(s, a))E_t(s, a)
        E_0(s, a) = 0
        E_t(s, a) = 𝜸𝝀E_t-1(s, a) + 1(S_t = s, A_t = a)

        Args:
            buf.states = [s1, s2, ..., sT-1]
            buf.actions = [a1, a2, ..., aT-1]
            buf.rewards = [r2, r3, ..., rT]

        Return:
            (inputs, targets): 
                inputs is a state list; 
                targets contains lists of action-values for each state in inputs
        '''
        Q = dict()              # store Q-value
        E = defaultdict(int)    # eligibility trace
        history = [(s, a) for s, a in zip(buf.states, buf.actions)]
        his1 = history.copy()
        his2 = history
        del(his2[0])
        his2.append((0, 0))

        for i, ((s, a), r, (s_, a_)) in enumerate(zip(his1, buf.rewards, his2)):
            s = tuple(s)
            E[(s, a)] += 1
            if s not in Q:
                Q[s] = self.model.predict(
                    np.expand_dims(np.asarray(s), axis=0))[0]
            if i + 1 == len(buf.rewards):
                td_error = r - Q[s][a]
            else:
                s_ = tuple(s_)
                if s_ not in Q:
                    Q[s_] = self.model.predict(
                        np.expand_dims(np.array(s_), axis=0))[0]
                td_error = r + self.gamma * Q[s_][a_] - Q[s][a]
            # update Q and E
            deletion = []
            for (s, a), e in E.items():
                if s not in Q:
                    Q[s] = self.model.predict(
                        np.expand_dims(np.asarray(s), axis=0))[0]
                Q[s][a] += td_error * E[(s, a)]
                E[(s, a)] *= self.lambda_
                if E[(s, a)] < 0.01:
                    deletion.append((s, a))
            # delete small eligibility traces
            for s, a in deletion:
                del E[(s, a)]

        inputs = np.zeros((len(Q), ) + self.env.observation_space.shape)
        targets = np.zeros((len(Q), self.env.action_space.n))
        for i, (s, target) in enumerate(Q.items()):
            inputs[i] = np.asarray(s)
            targets[i] = target
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
                    Q[tuple(s_tn)] = self.model.predict(
                        np.expand_dims(s_tn, axis=0))[0]
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
            q_t_lambda += (self.lambda_ ** (n - 1)) * \
                self.q_t_n(t, n, rewards, history, Q)
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
                Q[tuple(s_n)] = self.model.predict(
                    np.expand_dims(s_n, axis=0))[0]
            G += (self.gamma ** n) * Q[tuple(s_n)][a_n]
        return G
