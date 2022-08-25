
import numpy as np


class UCB():

    def __init__(self, n_products, n_arms):

        self.n_products = n_products
        self.n_arms = n_arms
        self.rewards_per_arm = [[[] for _ in range(self.n_arms)] for _ in range(self.n_products)]
        self.collected_rewards = [[[]for _ in range(self.n_arms)] for _ in range(self.n_products)]
        self.counters = np.zeros((self.n_products, self.n_arms), dtype=int)
        self.collected_realization_per_arm = [[[]for _ in range(self.n_arms)] for _ in range(self.n_products)]
        self.pulled_rounds = np.zeros((n_products, n_arms)) # how many round a specific arm has been pulled, one for the initialization
        self.empirical_means = np.zeros((n_products, n_arms))
        self.confidence = np.ones((n_products, n_arms))*np.inf
        self.t = 1

    def pull_arm(self, prices, products_sold, expected_ratios):
        upper_conf = self.empirical_means + self.confidence
        superarm = []
        for i in range(self.n_products):
            dot = upper_conf[i]*prices
            dot = dot * products_sold[i]
            dot = dot * expected_ratios[i]
            j = np.random.choice(np.where(dot == dot.max())[0])
            self.pulled_rounds[i][j] += 1
            superarm.append(j)
        print(self.pulled_rounds)
        return superarm

    def update(self, pulled_arms, conversion_rates):

        for i in range(self.n_products):
            self.empirical_means[i][pulled_arms[i]] = (self.empirical_means[i][pulled_arms[i]] * (self.pulled_rounds[i][pulled_arms[i]]-1) + conversion_rates[i]) / self.pulled_rounds[i][pulled_arms[i]]
        print("EMPIRICAL MEAN")
        print(self.empirical_means)
        for i in range(self.n_products):
            for a in range(self.n_arms):
                n_samples = self.pulled_rounds[i][a]
                self.confidence[i][a] = (2*np.log(self.t+1)/n_samples)**0.5 if n_samples > 0 else np.inf
        print("CONFIDENCE")
        print(self.confidence)
        self.t += 1

