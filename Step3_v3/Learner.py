from abc import abstractmethod

import numpy as np

from Step3_v3.Environment import Environment


class Learner:

    def __init__(self, env: Environment):
        self.t = 0
        self.n_products = env.n_products
        self.n_arms = env.n_arms
        self.prices = env.prices
        self.max_products_sold = np.mean(
            env.max_products_sold * np.expand_dims(env.user_probabilities, axis=(1, 2)),
            axis=0)
        self.lambda_p = env.lambda_p
        self.alpha_ratios_parameters = np.sum(env.alpha_ratios_parameters, axis=0)
        self.graph_probabilities = np.sum(
            env.graph_probabilities * np.expand_dims(env.user_probabilities, axis=(1, 2)),
            axis=0
        )

    def pull(self):
        best_configuration = np.zeros(self.n_products)
        best_sample = 0
        conversion_rates = self.sample()
        products_sold = np.random.randint(self.max_products_sold + 1)
        # TODO: Compute contribution
        conversion_rates * self.prices * products_sold + contribution

        return best_configuration

    @abstractmethod
    def update(self, results):
        pass

    @abstractmethod
    def sample(self):
        pass