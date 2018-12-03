# Proximal Policy Optimization
# https://arxiv.org/abs/1707.06347
# https://www.52coding.com.cn/2018/11/25/RL%20-%20PPO/

import gym
import numpy as np
import tensorflow as tf

from logging import getLogger
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.base import BaseController
from src.util import discount_cumsum, mlp, cnn
from src.config import Config, ControllerType

logger = getLogger(__name__)


class PPOControl(BaseController):
    def __init__(self, env, config: Config):
        self.env = env
        self.epsilon = config.controller.epsilon    # clip ratio
        self.gamma = config.controller.gamma
        self.lam = config.controller.lambda_
        self.pi_lr = config.trainer.lr   # 1e-4
        self.v_lr = 1e-3
        self.max_workers = config.controller.max_workers
        tfconfig = tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                allow_growth=True,
                visible_device_list='0'
            )
        )
        self.sess = tf.Session(config=tfconfig)
        self.raw_pixels = config.controller.raw_pixels
        if self.raw_pixels:
            state_space = [84, 84, 2]
        else:
            state_space = self.env.observation_space.shape
        self.actor = PPOActor(self.sess, state_space, self.env.action_space.n,
                              self.pi_lr, self.epsilon, self.raw_pixels)
        self.critic = PPOCritic(self.sess, state_space,
                                self.v_lr, self.raw_pixels)
        self.build_model()

    def build_model(self):
        self.actor.build_model()
        self.critic.build_model()
        self.sess.run(tf.global_variables_initializer())

    def action(self, observation, predict=False, return_q=False, epsilon=None, return_logp=True):
        if return_q:
            v = self.critic.value_of(observation)
            return self.actor.action(observation), [v]
        return self.actor.action(observation)[0]

    def train(self, batch_buffers, i):
        '''Update parameters

        Args:
            batch_buffers = [buf1, buf2, ...]
        '''
        batch_states = []
        batch_actions = []
        batch_rets = []
        batch_advs = []
        batch_logp_old = []
        total_rewards = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.build_training_set, buf)
                       for buf in batch_buffers]
            for future in futures:
                states, actions, returns, advantages, logps, r = future.result()
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_rets.extend(returns)
                batch_advs.extend(advantages)
                batch_logp_old.extend(logps)
                total_rewards.append(r)

        self.actor.train(batch_states, batch_actions,
                         batch_advs, batch_logp_old, np.mean(total_rewards), i)
        self.critic.train(batch_states, batch_rets, i)

    def build_training_set(self, buf):
        rewards_to_go = discount_cumsum(buf.rewards, self.gamma)
        buf.values = np.array(buf.values)
        # compute GAE
        deltas = buf.rewards[:-1] + self.gamma * \
            buf.values[1:] - buf.values[:-1]
        advs = discount_cumsum(deltas, self.gamma * self.lam)
        # advantage normalization trick
        adv_mean, adv_std = np.mean(advs), np.std(advs)
        advs = (advs - adv_mean) / adv_std
        # squeeze
        actions = np.squeeze(buf.actions)
        logps = np.squeeze(buf.logps)
        if self.raw_pixels:
            states = np.array(buf.states[:-1])
            states = np.reshape(states, (-1, 84, 84, 2))
        else:
            states = buf.states[:-1]
        return states, actions[:-1], rewards_to_go[:-1], advs, logps[:-1], sum(buf.rewards)

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
            logger.error(e)


class PPOActor:
    def __init__(self, sess, n_features, n_actions, lr, epsilon, raw_pixels):
        self.sess = sess
        if raw_pixels:
            self.n_features = n_features
        else:
            self.n_features = n_features[0]
        self.n_actions = n_actions
        self.lr = lr
        self.epsilon = epsilon
        self.train_policy_iter = 80
        self.target_kl = 0.01
        self.raw_pixels = raw_pixels

    def build_model(self):
        clip_ratio = self.epsilon
        # Input placeholder
        if self.raw_pixels:
            self.s_ph = tf.placeholder(tf.float32, [None] + self.n_features)
        else:
            self.s_ph = tf.placeholder(tf.float32, [None, self.n_features])
        self.a_ph = tf.placeholder(tf.int32, [None])
        self.logp_old_ph = tf.placeholder(tf.float32, [None])
        self.adv_ph = tf.placeholder(tf.float32, [None])
        # Construct model
        with tf.variable_scope('pi'):
            if self.raw_pixels:
                logits = mlp(cnn(self.s_ph), [256, self.n_actions], tf.tanh)
            else:
                logits = mlp(self.s_ph, [128, 64, self.n_actions], tf.tanh)
        self.logp_all = tf.nn.log_softmax(logits)
        self.pi = tf.squeeze(tf.multinomial(logits, 1), axis=1)
        self.logp_pi = tf.reduce_sum(tf.one_hot(
            self.pi, depth=self.n_actions) * self.logp_all, axis=1)
        logp = tf.reduce_sum(tf.one_hot(
            self.a_ph, depth=self.n_actions) * self.logp_all, axis=1)
        # PPO objectives
        # pi(a|s) / pi_old(a|s)
        ratio = tf.exp(logp - self.logp_old_ph)
        min_adv = tf.where(self.adv_ph > 0, (1+clip_ratio)
                           * self.adv_ph, (1-clip_ratio)*self.adv_ph)
        self.pi_loss = - \
            tf.reduce_mean(tf.minimum(ratio * self.adv_ph, min_adv))
        self.approx_kl = tf.reduce_mean(self.logp_old_ph - logp)
        self.approx_ent = tf.reduce_mean(-logp)
        self.pi_loss -= 0.01 * self.approx_ent
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.pi_loss)

    def action(self, observation):
        '''
        Choose an action according to approximated softmax policy.

        Args:
            observation: An observation from the environment

        Return:
            The action choosed according to the policy
        '''
        my_action, logp = self.sess.run(
            [self.pi, self.logp_pi], feed_dict={self.s_ph: observation})
        return my_action, logp

    def train(self, states, actions, advs, logp_old, avg_reward, i):
        '''Update parameters

        Args:
            states = [s1, s2, ..., sn]
            actions = [a1, a2, ..., an]
            advs = [adv1, adv2, ..., advn]
            logp_old = [logp1, logp2, ..., logpn]
            i: episode number
        '''
        inputs = {
            self.s_ph: states,
            self.a_ph: actions,
            self.adv_ph: advs,
            self.logp_old_ph: logp_old
        }
        pi_loss_old, ent = self.sess.run(
            [self.pi_loss, self.approx_ent], feed_dict=inputs)
        for j in range(self.train_policy_iter):
            _, kl = self.sess.run(
                [self.optimizer, self.approx_kl], feed_dict=inputs)
            kl = kl.mean()
            if kl > 1.5 * self.target_kl:
                logger.info(
                    'Early stopping at step %d due to reaching max kl.' % j)
                break
        pi_loss_new, kl = self.sess.run(
            [self.pi_loss, self.approx_kl], feed_dict=inputs)
        logger.info(
            f"\n\tEpisode: {i}\n\tAvg Reward: {avg_reward:.2f}\n\t"
            f"Loss_pi: {pi_loss_old:.3e}\n\tEntropy: {ent:.2f}\n\t"
            f"KL: {kl:.2f}\n\tDelta_Pi_Loss: {(pi_loss_new - pi_loss_old):.2e}")


class PPOCritic:
    def __init__(self, sess, n_features, lr, raw_pixels):
        self.sess = sess
        if raw_pixels:
            self.n_features = n_features
        else:
            self.n_features = n_features[0]
        self.lr = lr
        self.model = None
        self.train_value_iter = 80
        self.raw_pixels = raw_pixels

    def build_model(self):
        with tf.variable_scope('v'):
            if self.raw_pixels:
                self.s_ph = tf.placeholder(
                    tf.float32, [None] + self.n_features)
                x = cnn(self.s_ph)
                hidden_sizes = [256, 1]
            else:
                self.s_ph = tf.placeholder(tf.float32, [None, self.n_features])
                x = self.s_ph
                hidden_sizes = [64, 64, 1]
        self.ret_ph = tf.placeholder(tf.float32, [None])
        self.value = tf.squeeze(mlp(x, hidden_sizes, tf.tanh), axis=1)
        self.v_loss = tf.reduce_mean(
            tf.losses.mean_squared_error(self.ret_ph, self.value))
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.v_loss)

    def value_of(self, state):
        v = self.sess.run(self.value, feed_dict={self.s_ph: state})
        return v

    def train(self, states, rets, i):
        inputs = {
            self.s_ph: states,
            self.ret_ph: rets
        }
        for _ in range(self.train_value_iter):
            _, loss = self.sess.run(
                [self.optimizer, self.v_loss], feed_dict=inputs)
        print(f"\tLoss_v = {loss:.2e}")
