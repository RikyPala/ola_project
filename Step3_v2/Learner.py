import numpy as np
from Environment import Environment


class Learner:

    def __init__(self, env: Environment):

        # Parameters
        self.n_products = env.n_products
        self.n_arms = env.n_arms
        self.prices = env.prices
        self.max_products_sold = np.mean(
            env.max_products_sold * env.user_probabilities,
            axis=1
        )
        self.lambda_p = env.lambda_p
        self.alpha_ratios_parameters = np.sum(env.alpha_ratios_parameters, axis=0)
        self.graph_probabilities = np.mean(
            env.graph_probabilities * np.expand_dims(env.user_probabilities, axis=(1, 2)),
            axis=0
        )
        self.secondaries = env.secondaries
        self.estimated_conversion_rates = np.zeros((self.n_products, self.n_arms))

        self.collected_rewards = [[] for _ in range(self.n_products)]

    def update_observations(self, rewards):
        for i in range(self.n_products):
            self.collected_rewards[i].append(rewards[i])
