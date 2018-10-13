# REINFORCE (Monte-Carlo Policy Gradient)
# 𝜽 = 𝜽 + 𝜶 𝝯log(pi(s_t, a_t))G_t
# G_t = R_t+1 + 𝜸R_t+2 + 𝜸^2R_t+3 + ... + 𝜸^(T-1)R_T
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

class ReinforceControl(BaseController):
    def __init__(self, env, config: Config):
        super().__init__()
        self.env = env
        self.config = config
        self.epsilon = config.controller.epsilon
        self.gamma = config.controller.gamma
        self.lr = config.trainer.lr
        self.max_workers = config.controller.max_workers
        self.sess = tf.Session()
        self.build_model()

    def build_model(self):
        # Input placeholder
        self.s = tf.placeholder(tf.float32, [None, self.env.observation_space.shape[0]])
        self.a = tf.placeholder(tf.int32, [None, 1])
        self.G = tf.placeholder(tf.float32, [None])
        # Construct model
        self.acts_prob = tf.layers.dense(
            inputs=self.s,
            units=self.env.action_space.n,    # output units
            activation=tf.nn.softmax,   # get action probabilities
            kernel_initializer=tf.random_normal_initializer(0., 0.0001),  # weights
            bias_initializer=tf.constant_initializer(0.0001),  # biases
            name='acts_prob'
        )
        # cost = log(pi(s_t, a_t))G_t
        log_prob = tf.squeeze(tf.log(tf.batch_gather(self.acts_prob, self.a)))
        self.cost = tf.reduce_mean(tf.multiply(self.G, log_prob))
        tf.summary.scalar('Log prob return', self.cost)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(-self.cost)
        self.summary = tf.summary.merge_all()
        self.sess.run(tf.global_variables_initializer())
        self.train_writer = tf.summary.FileWriter(self.config.resource.graph_dir,
                                                self.sess.graph)

    def action(self, observation, predict=False, return_q=False, epsilon=None):
        '''

        Choose an action according to approximated softmax policy.

        Args:
            observation: An observation from the environment

        Return:
            The action choosed according to softmax policy
        '''
        probs = self.sess.run(self.acts_prob, feed_dict={self.s: observation})
        my_action = int(np.random.choice(range(self.env.action_space.n), p=probs.ravel()))
        return my_action

    def train(self, batch_history, batch_rewards, i):
        '''Update parameters

        Args:
            batch_history = [[(s1, a1), (s2, a2), ...,(sT-1, aT-1)], ...]
            batch_rewards = [[r2, r3, ..., rT], ...]
        '''
        batch_states, batch_actions, batch_return = self.build_training_set_on_batch(batch_history, 
                                                                                     batch_rewards)
        batch_actions = np.asarray(batch_actions)
        batch_actions = np.expand_dims(batch_actions, axis=1)
        _, summary, cost = self.sess.run([self.optimizer, self.summary, self.cost], feed_dict={self.G: batch_return,
                                                              self.s: batch_states,
                                                              self.a: batch_actions})
        self.train_writer.add_summary(summary, i)
        logger.info(f"Episode {i}, logPi(s, a)G = {cost:.2f}")

    def build_training_set_on_batch(self, batch_history, batch_rewards):
        batch_states = []
        batch_actions = []
        batch_return = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.build_training_set, history, rewards) 
                       for history, rewards in zip(batch_history, batch_rewards)]
            for future in futures:
                states, actions, returns = future.result()
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_return.extend(returns)
        return batch_states, batch_actions, batch_return

    def build_training_set(self, history, rewards):
        states = []
        actions = []
        returns = []
        for i in range(0, len(history), 2):
            states.append(history[i][0])
            actions.append(history[i][1])
            returns.append(self.compute_return(rewards, i))
        return states, actions, returns

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

    def save(self, path):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, path)
        logger.info(f"Save weight to {save_path}")

    def load(self, path):
        try:
            saver = tf.train.Saver()
            saver.restore(self.sess, path)
            logger.info(f"Load weight from {path}")
        except Exception as e:
            pass
