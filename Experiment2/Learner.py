import numpy as np


class Learner:

    def __init__(self, n_products, n_arms, n_user_types):

        self.n_products = n_products
        self.n_arms = n_arms
        self.n_user_types = n_user_types

        self.rewards_per_arm = [[[] for _ in range(self.n_arms)] for _ in range(self.n_products)]
        self.collected_rewards = [[] for _ in range(self.n_products)]

        self.counters = np.zeros(self.n_products, dtype=int)

    def pull_arm(self):

        mask = self.counters < self.n_arms - 1
        choice = np.random.choice(np.arange(self.n_products)[mask])
        self.counters[choice] += 1
        pulled_arms = self.counters
        return pulled_arms

    def update(self, pulled_arms, rewards):

        if np.shape(self.collected_rewards)[-1] == 0:
            improvements = True
        else:
            improvements = any(rewards > np.array([elem[-1] for elem in self.collected_rewards]))

        for i in range(self.n_products):
            self.rewards_per_arm[i][pulled_arms[i]].append(rewards[i])
            self.collected_rewards[i].append(rewards[i])

        if not improvements:
            return 1
        elif all(self.counters == self.n_arms - 1):
            return 2
        else:
            return 0
