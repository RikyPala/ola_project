import numpy as np


class Learner:

    def __init__(self, n_products, n_arms):
        self.n_products = n_products
        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = [[[] for _ in range(self.n_arms)] for _ in range(self.n_products)]
        self.collected_rewards = [[] for _ in range(self.n_products)]
        self.expected_rewards = np.zeros((n_products, n_arms))
        self.counters = np.zeros(self.n_products, dtype=int)

    def update_observations(self, pulled_arms, rewards):
        for i in range(self.n_products):
            self.rewards_per_arm[i][pulled_arms[i]].append(rewards[i])
            self.collected_rewards[i].append(rewards[i])

    def pull_arm(self):
        mask = self.counters < self.n_arms - 1
        choice = np.random.choice(np.arange(self.n_products)[mask])
        self.counters[choice] += 1
        pulled_arms = self.counters
        return pulled_arms

    def update(self, pulled_arms, rewards):
        self.t += 1
        if np.shape(self.collected_rewards)[-1] == 0:
            improvements = True
        else:
            improvements = any(rewards > np.array([elem[-1] for elem in self.collected_rewards]))
        self.update_observations(pulled_arms, rewards)
        for i in range(self.n_products):
            self.expected_rewards[i][pulled_arms[i]] =\
                (self.expected_rewards[i][pulled_arms[i]] * (self.t - 1) + rewards[i]) / self.t
        stop = all(self.counters == self.n_arms - 1) or not improvements
        return stop
