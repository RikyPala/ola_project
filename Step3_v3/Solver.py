import numpy as np
from tqdm.auto import tqdm

from Environment import Environment


class Solver:

    def __init__(self, env: Environment):
        self.n_products = env.n_products
        self.n_arms = env.n_arms

        self.prices = env.prices
        self.conversion_rates = np.sum(
            env.conversion_rates * np.expand_dims(env.user_probabilities, axis=(1, 2)),
            axis=0)
        self.lambda_p = env.lambda_p
        self.avg_products_sold = np.sum(
            (env.max_products_sold + 1) / 2 * np.expand_dims(env.user_probabilities,  axis=(1, 2)),
            axis=0)

        alpha_ratios_parameters = np.sum(env.alpha_ratios_parameters, axis=0)
        self.expected_alpha_ratios = self.compute_expected_alpha_ratios(alpha_ratios_parameters)

        graph_probabilities = np.sum(
            env.graph_probabilities * np.expand_dims(env.user_probabilities, axis=(1, 2)),
            axis=0)
        secondaries = env.secondaries
        self.inverse_graph = self.compute_inverse_graph(secondaries, graph_probabilities)

    def compute_expected_alpha_ratios(self, alpha_ratios_parameters):
        alpha_ratios_avg = alpha_ratios_parameters[:, 0] / \
                           (alpha_ratios_parameters[:, 0] + alpha_ratios_parameters[:, 1])
        norm_factors = np.sum(alpha_ratios_avg, axis=0)
        return alpha_ratios_avg / norm_factors

    def compute_inverse_graph(self, secondaries, graph_probabilities):
        inverse_graph = np.zeros((self.n_products, self.n_products))
        inverse_graph[np.arange(self.n_products), secondaries[:, 0]] = \
            graph_probabilities[np.arange(self.n_products), secondaries[:, 0]]  # primaries
        inverse_graph[np.arange(self.n_products), secondaries[:, 1]] = \
            graph_probabilities[np.arange(self.n_products), secondaries[:, 1]] * self.lambda_p  # secondaries
        return inverse_graph.T

    def find_optimal(self):
        arms_shape = (self.n_arms,) * self.n_products
        expected_reward_per_configuration = np.zeros(arms_shape)
        for configuration, _ in tqdm(np.ndenumerate(expected_reward_per_configuration)):
            rewards = np.zeros(self.n_products)
            for start in range(self.n_products):
                common_term = self.conversion_rates[start, configuration[start]] * \
                              self.prices[start, configuration[start]] * \
                              self.avg_products_sold[start, configuration[start]]
                rewards[start] = common_term \
                                 * (self.expected_alpha_ratios[start] +\

                                self.compute_children_contribute([start], configuration))
            expected_reward_per_configuration[configuration] = np.sum(rewards)
        optimal_configuration = np.unravel_index(np.argmax(expected_reward_per_configuration),
                                                 expected_reward_per_configuration.shape)
        optimal_reward = np.max(expected_reward_per_configuration)
        return optimal_configuration, optimal_reward

    def compute_children_contribute(self, predecessors, configuration):
        root = predecessors[-1]
        contribute = 0
        for child, prob in enumerate(self.inverse_graph[root]):
            if prob == 0 or child in predecessors:
                continue
            contribute += prob * self.conversion_rates[child, configuration[child]] * (
                    self.expected_alpha_ratios[child] +
                    self.compute_children_contribute([*predecessors, child], configuration)
            )
        return contribute
