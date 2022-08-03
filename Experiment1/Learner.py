import numpy as np


class Learner:

    def __init__(self, n_products, n_arms, n_user_types):

        self.first_iteration = True

        self.n_products = n_products
        self.n_arms = n_arms
        self.n_user_types = n_user_types

        self.rewards_per_arm = [[[] for _ in range(self.n_arms)] for _ in range(self.n_products)]
        self.collected_rewards = [[] for _ in range(self.n_products)]

        self.counters = np.zeros(self.n_products, dtype=int)

    # TODO: add function to estimate node-arrival probabilities

    # TODO: add function to estimate expected rewards

    def pull_arms(self):  # TODO: remove pull_arms function

        if self.first_iteration:
            self.first_iteration = False
            return [np.zeros(self.n_products)]

        mask = self.counters < self.n_arms - 1
        configurations = []
        for i in range(self.n_products):
            if mask[i]:
                arms = self.counters
                arms[i] += 1
                configurations.append(arms)

        return configurations

    """
    def update(self, pulled_arms, rewards):

        if np.shape(self.collected_rewards)[-1] == 0:
            improvements = True
        else:
            improvements = any(rewards > np.array([elem[-1] for elem in self.collected_rewards]))

        for i in range(self.n_products):
            self.rewards_per_arm[i][pulled_arms[i]].append(rewards[i])
            self.collected_rewards[i].append(rewards[i])

        stop = not improvements or all(self.counters == self.n_arms - 1)

        return stop
    """
