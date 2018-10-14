# Actor Critic Policy Gradient using advantage function
#   Actor: 𝜽 = 𝜽 + 𝜶 𝝯log(pi(s_t, a_t)) * TD_error
#   Critic: TD evaluation (estimate V(s))
# Policy approximation: softmax policy

import gym
import numpy as np
import tensorflow as tf

from logging import getLogger
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.base import BaseController
from src.config import Config, ControllerType

logger = getLogger(__name__)

class ActorCriticControl(BaseController):
    def __init__(self, env, config: Config):
        self.env = env
        self.config = config
        self.epsilon = config.controller.epsilon
        self.gamma = config.controller.gamma
        self.lr = config.trainer.lr
        self.max_workers = config.controller.max_workers
        self.sess = tf.Session()
        self.actor = Actor(self.sess, self.env.observation_space.shape[0], 
                           self.env.action_space.n, self.config.trainer.lr)
        self.critic = Critic(self.sess, self.env.observation_space.shape[0],
                            self.config.trainer.lr)
        self.build_model()

    def build_model(self):
        self.actor.build_model()
        self.critic.build_model()
        self.sess.run(tf.global_variables_initializer())

    def action(self, observation, predict=False, return_q=False, epsilon=None):
        if return_q:
            v = self.critic.value_of(observation)
            return self.actor.action(observation), [v]
        return self.actor.action(observation)

    def train(self, batch_history, batch_rewards, i):
        '''Update parameters

        Args:
            batch_history = [[(s1, a1), (s2, a2), ...,(sT-1, aT-1)], ...]
            batch_rewards = [[r2, r3, ..., rT], ...]
        '''
        batch_states = []
        batch_actions = []
        batch_tdtargets = []
        batch_tderror = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.build_training_set, history, rewards) 
                       for history, rewards in zip(batch_history, batch_rewards)]
            for future in futures:
                states, actions, td_targets, td_errors = future.result()
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_tdtargets.extend(td_targets)
                batch_tderror.extend(td_errors)

        self.actor.train(batch_states, batch_actions, batch_tderror, i)
        self.critic.train(batch_states, batch_tdtargets, i)

    def build_training_set(self, history, rewards):
        states = [x[0] for x in history]
        actions = [x[1] for x in history]
        td_targets = []
        td_errors = []
        V = defaultdict(float)
        T = len(rewards)
        for i, (s, r) in enumerate(zip(states, rewards)):
            if i < T - 1:
                s_ = states[i + 1]
                if tuple(s_) not in V:
                    V[tuple(s_)] = self.critic.value_of(np.expand_dims(s_, axis=0))
                td_target = r + self.gamma * V[tuple(s_)]
            else:
                td_target = r
            if tuple(s) not in V:
                V[tuple(s)] = self.critic.value_of(np.expand_dims(s, axis=0))
            td_error = td_target - V[tuple(s)]
            td_targets.append(td_target)
            td_errors.append(td_error)
        return states, actions, td_targets, td_errors

    def save(self, path):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, path)
        logger.info(f"Save weight to {save_path}")

    def load(self, path):
        try:
            saver = tf.train.Saver()
            saver.restore(self.sess, path)
            logger.info(f"Load weight from {path}")
        except Exception:
            pass

class Actor:
    def __init__(self, sess, n_features, n_actions, lr):
        self.sess = sess
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = lr

    def build_model(self):
        # Input placeholder
        self.s = tf.placeholder(tf.float32, [None, self.n_features])
        self.a = tf.placeholder(tf.int32, [None, 1])
        self.td_error = tf.placeholder(tf.float32, [None])
        # Construct model
        self.acts_prob = tf.layers.dense(
            inputs=self.s,
            units=self.n_actions,    # output units
            activation=tf.nn.softmax,   # get action probabilities
            kernel_initializer=tf.random_normal_initializer(0., 0.0001),  # weights
            bias_initializer=tf.constant_initializer(0.0001),  # biases
            name='acts_prob'
        )
        # cost = log(pi(s_t, a_t)) * TD_error
        log_prob = tf.squeeze(tf.log(tf.batch_gather(self.acts_prob, self.a)))
        self.cost = tf.reduce_mean(tf.multiply(self.td_error, log_prob))
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(-self.cost)
        

    def action(self, observation):
        '''
        Choose an action according to approximated softmax policy.

        Args:
            observation: An observation from the environment

        Return:
            The action choosed according to softmax policy
        '''
        probs = self.sess.run(self.acts_prob, feed_dict={self.s: observation})
        my_action = int(np.random.choice(range(self.n_actions), p=probs.ravel()))
        return my_action

    def train(self, batch_states, batch_actions, batch_tderror, i):
        '''Update parameters

        Args:
            batch_states = [s1, s2, ..., sn]
            batch_actions = [a1, a2, ..., an]
            batch_tderror = [e1, e2, ..., en]
            i: episode number
        '''
        batch_actions = np.asarray(batch_actions)
        batch_actions = np.expand_dims(batch_actions, axis=1)
        _, cost = self.sess.run([self.optimizer, self.cost], feed_dict={self.td_error: batch_tderror,
                                                              self.s: batch_states,
                                                              self.a: batch_actions})
        logger.info(f"Episode {i}, Actor loss = {-cost:.2f}")

class Critic:
    def __init__(self, sess, n_features, lr):
        self.sess = sess
        self.n_features = n_features
        self.lr = lr
        self.model = None

    def build_model(self):
        self.s = tf.placeholder(tf.float32, [None, self.n_features])
        self.td_targets = tf.placeholder(tf.float32, [None])
        self.value = tf.squeeze(tf.layers.dense(
            inputs=self.s,
            units=1,    # output units
            activation=None,   # get action probabilities
            kernel_initializer=tf.random_normal_initializer(0., 0.0001),  # weights
            bias_initializer=tf.constant_initializer(0.0001),  # biases
            name='value'
        ))
        self.cost = tf.reduce_mean(tf.losses.mean_squared_error(self.td_targets, self.value))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def value_of(self, state):
        v = self.sess.run(self.value, feed_dict={self.s: state})
        return v

    def train(self, batch_states, batch_tdtargets, i):
        _, cost = self.sess.run([self.optimizer, self.cost], feed_dict={self.s: batch_states,
                                                            self.td_targets: batch_tdtargets})
        logger.info(f"Episode {i}, Critic loss = {cost:.2f}")


