import numpy as np


class Learner:

    def __init__(self, n_products, n_arms):
        self.n_products = n_products
        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = [[[] for _ in range(self.n_arms)] for _ in range(self.n_products)]
        self.collected_rewards = [[] for _ in range(self.n_products)]

    def update_observations(self, pulled_arms, rewards):
        for i in range(self.n_products):
            self.rewards_per_arm[i][pulled_arms[i]].append(rewards[i])
            self.collected_rewards[i].append(rewards[i])
