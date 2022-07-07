import numpy as np


class Environment:
    def __init__(self, n_products, n_arms, probabilities):
        self.n_products = n_products
        self.n_arm = n_arms
        self.probabilities = probabilities

    def round(self, pulled_arms):
        rewards = np.zeros(self.n_products)
        for i in range(self.n_products):
            rewards[i] = np.random.binomial(1, self.probabilities[i, pulled_arms[i]])
        return rewards
