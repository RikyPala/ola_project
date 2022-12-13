import numpy as np
from Learner import Learner
from Environment import Environment
from RoundData import RoundData


class UCB(Learner):

    def __init__(self, env: Environment, feature_1=None, feature_2=None):
        super().__init__(env, feature_1, feature_2)
        self.t = 0
        self.confidence = np.ones((self.n_products, self.n_arms)) * np.inf
        self.empirical_means = np.zeros((self.n_products, self.n_arms))
        self.c = 0.5

    def update(self, round_data: RoundData):
        self.t += 1
        configuration = self.get_configuration_by_agg_classes(round_data.ctx_configs)
        conversions = np.sum(round_data.conversions[self.agg_classes], axis=0)
        visits = np.sum(round_data.visits[self.agg_classes], axis=0)
        conversion_rates = conversions / visits
        idxs = np.arange(self.n_products)
        n_pulls = self.pulled_rounds[idxs, configuration]
        self.empirical_means[idxs, configuration] = \
            (self.empirical_means[idxs, configuration] * (n_pulls - 1) + conversion_rates) / n_pulls
        for prod in range(self.n_products):
            for arm in range(self.n_arms):
                n = self.pulled_rounds[prod, arm]
                self.confidence[prod, arm] = self.c * (np.log(self.t) / n) ** 0.5 if n > 0 else np.inf
        self.update_estimates(configuration, round_data)

    def sample(self):
        return self.empirical_means + self.confidence

    def get_means(self):
        return self.empirical_means
