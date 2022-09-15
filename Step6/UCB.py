import numpy as np

from Learner import Learner


class UCBArm:

    def __init__(self, c_value):
        self.n = 0  # the number of times this arm has been pulled
        self.x = []  # list of all rewards ?
        self.empirical_means = 0
        self.confidence = np.inf
        self.c = c_value

    def sample(self):
        """ compute upper confidence bound """
        return self.empirical_means + self.c * self.confidence

    def update_empirical_means(self, reward):
        self.n += 1
        self.x.append(reward)  # needed?
        self.empirical_means = (self.empirical_means * (self.n - 1) + reward) / self.n  # n or t?

    def update_confidence(self, t):
        self.confidence = (2 * np.log(t) / self.n) ** 0.5 if self.n > 0 else np.inf


class UCB(Learner):

    def __init__(self, arms_shape, c_value=200):
        arms = []
        n_arms = np.zeros(arms_shape).size
        for _ in range(n_arms):
            arms.append(UCBArm(c_value))
        arms = np.array(arms).reshape(arms_shape)
        super().__init__(arms)

    def update(self, configuration, reward):
        self.t += 1
        self.arms[configuration].update_empirical_means(reward)
        for arm in self.arms:
            arm.update_confidence(self.t)

