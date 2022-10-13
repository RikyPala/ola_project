import numpy as np
from Learner import Learner
from Environment import Environment, RoundData


class UCB(Learner):

    def __init__(self, env: Environment):
        super().__init__(env)
        self.t = 0
        self.confidence = np.ones((self.n_products, self.n_arms)) * np.inf
        self.empirical_means = np.zeros((self.n_products, self.n_arms))

    def update(self, results: RoundData):
        self.t += 1

        configuration = results.configuration
        conversion_rates = results.conversions / results.visits
        idxs = np.arange(self.n_products)
        n_pulls = self.pulled_rounds[idxs, configuration]

        self.empirical_means[idxs, configuration] = \
            (self.empirical_means[idxs, configuration] * (n_pulls - 1) + conversion_rates) / \
            self.pulled_rounds[idxs, configuration]
        self.confidence[idxs, configuration] = (2 * np.log(self.t) / n_pulls) ** 0.5
        self.update_marginal_reward(configuration)

    def sample(self):
        return self.empirical_means + self.confidence

    def get_means(self):
        return self.empirical_means
