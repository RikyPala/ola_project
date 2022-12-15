import numpy as np

from Learner import Learner
from Environment import Environment
from RoundData import RoundData
from RoundsHistory import RoundsHistory


class TS(Learner):

    def __init__(self, env: Environment, feature_1=None, feature_2=None):
        super().__init__(env, feature_1, feature_2)
        self.beta_parameters = np.ones((self.n_products, self.n_arms, 2))
        for round_data in RoundsHistory.history[RoundsHistory.TS_index]:
            self.update(round_data)

    def update(self, round_data: RoundData):
        configuration = self.get_configuration_by_agg_classes(round_data.ctx_configs)
        self.pulled_rounds[np.arange(self.n_products), configuration] += 1
        idxs = np.arange(self.n_products)
        conversions = np.sum(round_data.conversions[self.agg_classes], axis=0)
        visits = np.sum(round_data.visits[self.agg_classes], axis=0)
        self.beta_parameters[idxs, configuration, 0] += conversions
        self.beta_parameters[idxs, configuration, 1] += visits - conversions
        self.update_estimates(configuration, round_data)

    def sample(self):
        return np.random.beta(self.beta_parameters[:, :, 0], self.beta_parameters[:, :, 1])

    def get_means(self):
        return self.beta_parameters[:, :, 0] / (self.beta_parameters[:, :, 0] + self.beta_parameters[:, :, 1])
