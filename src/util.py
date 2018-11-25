import scipy.signal


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
