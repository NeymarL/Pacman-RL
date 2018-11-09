# Asychronous Advantage Actor Critic (A3C)

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F

from logging import getLogger
from collections import defaultdict
from torch.autograd import Variable

from src.config import Config, ControllerType

logger = getLogger(__name__)


class A3CControl():
    def __init__(self, env, config: Config):
        super().__init__()
        self.env = env
        self.lr = config.trainer.lr
        self.gamma = config.controller.gamma
        self.input_shape = self.env.observation_space.shape
        self.model = A3CModel(
            self.env.observation_space.shape[2],
            self.env.action_space.n,
        )
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr)
        self.hx = None
        self.cx = None

    def action(self, state, training=False):
        '''
        Choose an action according to policy network.

        Args:
            state: An observation from the environment
            training: bool, whether is training or not

        Return:
            The action choosed according to softmax policy
            If training == True, it will also return value,
            log probability and entropy.
        '''
        state = torch.tensor(state)
        state = state.view(
            1, self.input_shape[2], self.input_shape[0], self.input_shape[1])
        state = state.type(torch.FloatTensor)
        self.cx = Variable(self.cx.data)
        self.hx = Variable(self.hx.data)
        # inference
        value, logit, (self.hx, self.cx) = self.model(
            (state, (self.hx, self.cx)))
        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        action = prob.multinomial(num_samples=1).data
        log_prob = log_prob.gather(1, Variable(action))
        action = action.numpy()
        if training:
            return action, value, log_prob, entropy
        else:
            return action, value.data.numpy()[0][0]

    def update(self, rewards, values, log_probs, entropies, R):
        '''
        Compute gradients and backpropagation

        Args:
            rewards = [r1, r2, ..., r_t-1]
            log_probs = [p1, p2, ..., p_t-1]
            entropies = [e1, e2, ..., e_t-1]
            R: 0 for terminal s_t otherwise V(s_t)
        '''
        value_loss = 0
        policy_loss = 0
        for t in reversed(range(len(rewards))):
            R = self.gamma * R + rewards[t]
            td_error = R - values[t]
            # maximize return
            with torch.no_grad():
                advantage = td_error
            policy_loss = policy_loss - \
                (log_probs[t] * advantage + 0.01 * entropies[t])
            # minimize
            value_loss = value_loss + td_error.pow(2)
        self.optimizer.zero_grad()
        loss = policy_loss + 0.5 * value_loss
        loss.backward()

    def refresh_state(self):
        self.cx = Variable(torch.zeros(1, 256))
        self.hx = Variable(torch.zeros(1, 256))


class A3CModel(torch.nn.Module):
    def __init__(self, n_channels, n_actions):
        super(A3CModel, self).__init__()
        self.n_channels = n_channels
        self.n_actions = n_actions
        # 210 x 160 x 3
        self.conv1 = nn.Conv2d(n_channels, 16, 8, stride=4)
        self.bn1 = nn.BatchNorm2d(16)
        # 51 x 39 x 16
        self.conv2 = nn.Conv2d(16, 32, 6, stride=3, padding=(0, 2))
        self.bn2 = nn.BatchNorm2d(32)
        # 16 x 13 x 32
        self.conv3 = nn.Conv2d(32, 32, 4, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        # 7 x 5 x 32
        self.lstm = nn.LSTMCell(1120, 256)

        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, n_actions)

        # initialization
        self.apply(weights_init)
        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

    def forward(self, x):
        x, (hx, cx) = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(1, -1)
        hx, cx = self.lstm(x, (hx, cx))

        x = hx
        return self.critic_linear(x), self.actor_linear(x), (hx, cx)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    # x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    y = (x ** 2)
    y = torch.sum(y, 1, keepdim=True)
    x *= std / torch.sqrt(y)
    return x
