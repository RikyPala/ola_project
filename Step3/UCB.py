
import numpy as np


class UCB():

    def __init__(self, n_products, n_arms):

        self.n_products = n_products
        self.n_arms = n_arms
        self.rewards_per_arm = [[[] for _ in range(self.n_arms)] for _ in range(self.n_products)]
        self.collected_rewards = [[[]for _ in range(self.n_arms)] for _ in range(self.n_products)]
        self.counters = np.zeros((self.n_products, self.n_arms), dtype=int)
        self.collected_realization_per_arm = [[[]for _ in range(self.n_arms)] for _ in range(self.n_products)]
        self.pulled_rounds = np.zeros((n_products, n_arms)) # how many round a specific arm has been pulled
        self.empirical_means = np.zeros((n_products, n_arms))
        self.confidence = np.ones((n_products, n_arms))*np.inf
        self.t = 0

    def pull_arm(self):

        while self.t < 4:
            return [self.t, self.t, self.t, self.t, self.t]

        upper_conf = self.empirical_means + self.confidence
        superarm = []
        for i in range(0, self.n_products):
            print(np.argwhere(upper_conf[i] == upper_conf[i].max()).reshape(-1))
            j = np.random.choice(np.argwhere(upper_conf[i] == upper_conf[i].max()).reshape(-1))
            self.pulled_rounds[i, j] += 1
            superarm.append(j)
            print(self.pulled_rounds)
        return superarm

    def update(self, pulled_arm, conversion_rates):
        if self.t < 4:
            for i in range(self.n_products):
                self.empirical_means[i, self.t] = conversion_rates[i]
        print(self.empirical_means)
        self.t += 1
        """self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm] * (self.t - 1) + reward) / self.t
        for a in range(self.n_arms):
            n_samples = len(self.rewards_per_arm[a])
            self.confidence[a] = (2 * np.log(self.t) / n_samples) ** 0.5 if n_samples > 0 else np.inf
        self.update_observations(pulled_arm, reward)"""


    def update_observations(self, pulled_arms, rewards):
        for i in range(self.n_products):
            self.rewards_per_arm[i][pulled_arms[i]].append(rewards[i])
            self.collected_rewards[i].append(rewards[i])
