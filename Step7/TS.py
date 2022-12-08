import numpy as np

from Learner import Learner
from Environment import Environment, RoundData


class TS(Learner):

    def __init__(self, env: Environment):
        super().__init__(env)
        self.beta_parameters = np.ones((self.n_products, self.n_arms, 2))

    def update(self, round_data: RoundData):
        configuration = self.get_configuration_by_agg_classes(round_data.ctx_configs)
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
