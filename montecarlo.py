# Monte-Carlo Control with Q-value function approximation
# Policy evaluation: Q(s, a) <- Q(s, a) + (G - Q(s, a)) / N(s, a)
# Policy improvement: epsilon-greedy exploration
# Q-value function approximation: Two-layer perception (input layer and output layer only)

import gym
import numpy as np
from collections import defaultdict
from keras.models import Sequential
from keras.layers import Dense, Flatten
from concurrent.futures import ThreadPoolExecutor, as_completed

class MonteCarloControl:
    def __init__(self, env, epsilon=0.5, gamma=0.9, max_workers=8):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.model = self.build_model()
        self.max_workers = max_workers

    def build_model(self):
        model = Sequential()
        model.add(Dense(self.env.action_space.n, input_shape=self.env.observation_space.shape, 
                             kernel_initializer='uniform', activation='linear'))
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        return model

    def action(self, observation, predict=False, return_q=False):
        '''
        epsilon-greedy policy
        '''
        if np.random.rand() <= self.epsilon and not predict:
            a = self.env.action_space.sample()
        else:
            Q = self.model.predict(observation)
            a = np.argmax(Q)
            if return_q:
                return (a, Q)
        return a

    def update_q_value(self, history, rewards):
        '''
        Monte-Carlo evaluation
        history = [(s1, a1), (s2, a2), ...,(sT-1, aT-1)]
        rewards = [r2, r3, ..., rT]
        '''
        inputs, targets = self.build_training_set(history, rewards)
        self.model.train_on_batch(inputs, targets)

    def update_q_value_on_batch(self, batch_history, batch_rewards):
        '''
        Monte-Carlo evaluation
        batch_history = [[(s1, a1), (s2, a2), ...,(sT-1, aT-1)], ...]
        batch_rewards = [[r2, r3, ..., rT], ...]
        '''
        x = None
        y = None
        N = defaultdict(int)
        Q = defaultdict(float)
        for i, (history, rewards) in enumerate(zip(batch_history, batch_rewards)):
            for i, ((s, a), r) in enumerate(zip(history, rewards)):
                s = tuple(s)
                N[(s, a)] += 1
                G = self.compute_return(rewards, i)
                Q[(s, a)] += (G - Q[(s, a)]) / N[(s, a)]    # track the mean return

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
        self.model.train_on_batch(inputs, targets)

    def update_q_value_on_batch_multithreads(self, batch_history, batch_rewards):
        '''
        Monte-Carlo evaluation
        batch_history = [[(s1, a1), (s2, a2), ...,(sT-1, aT-1)], ...]
        batch_rewards = [[r2, r3, ..., rT], ...]
        '''
        x = None
        y = None
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.build_training_set, history, rewards) 
                       for history, rewards in zip(batch_history, batch_rewards)]
            for future in as_completed(futures):
                if x is None:
                    x, y = future.result()
                else:
                    inputs, targets = future.result()
                    x = np.concatenate((x, inputs), axis=0)
                    y = np.concatenate((y, targets), axis=0)
        self.model.train_on_batch(x, y)

    def build_training_set(self, history, rewards):
        N = defaultdict(int)
        Q = defaultdict(float)
        for i, ((s, a), r) in enumerate(zip(history, rewards)):
            s = tuple(s)
            N[(s, a)] += 1
            G = self.compute_return(rewards, i)
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

    def compute_return(self, rewards, i):
        G = 0
        for j in range(i, len(rewards)):
            G += (self.gamma ** j) * rewards[j]
        return G

    def save(self, path):
        self.model.save_weights(path)
        print(f"Save weight to {path}")

    def load(self, path):
        self.model.load_weights(path)
        print(f"Load weight from {path}")



