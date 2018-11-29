import scipy.signal
import tensorflow as tf
import numpy as np
import cv2


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def preprocess(img):
    '''
    Preprocess the origin screen input from Atari games roughly like the DQN paper does
    Assume the image shape is (210, 160, 3)
    '''
    try:
        assert img.shape == (210, 160, 3)
    except AssertionError:
        print(img.shape, " is not equal to (210, 160, 3)")
        exit(1)
    # gray scale
    gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
    # Canny edge
    canny = cv2.Canny(gray.astype('uint8'), 50, 200)

    def downsampling(img):
        # padding to (218, 168)
        img = np.pad(img, 4, mode='edge')
        # downsampling to (109, 84)
        img = img[::2, ::2]
        # crop to (84, 84)
        img = img[3:87][:]
        return img
    top = downsampling(gray)
    down = downsampling(canny)
    img = np.concatenate((top, down))
    return np.reshape(img, (84, 84, 2))


def mlp(x, hidden_sizes, activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=tf.tanh)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)


def cnn(x):
    # (90, 90, 4) -> (20, 20, 16)
    x = tf.layers.conv2d(x, 16, [8, 8], [4, 4], activation=tf.nn.relu)
    # (20, 20, 16) -> (8, 8, 32)
    x = tf.layers.conv2d(x, 32, [5, 5], [2, 2], activation=tf.nn.relu)
    # (8, 8, 32) -> (2048)
    x = tf.layers.flatten(x)
    return x


class Buffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.logps = []

    def add(self, s, a, r, v=None, logp=None):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        if v:
            self.values.append(v)
        if logp:
            self.logps.append(logp)
