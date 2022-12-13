import numpy as np
from Environment import Environment


class Solver:

    def __init__(self, env: Environment):
        self.n_products = env.n_products
        self.n_arms = env.n_arms
        self.n_user_types = env.n_user_types

        self.prices = env.prices
        self.conversion_rates = env.conversion_rates
        self.lambda_p = env.lambda_p
        self.avg_products_sold = env.max_products_sold / 2

        alpha_ratios_parameters = env.alpha_ratios_parameters
        self.expected_alpha_ratios = self.compute_expected_alpha_ratios(alpha_ratios_parameters)

        graph_probabilities = env.graph_probabilities
        secondaries = env.secondaries
        self.inverse_graph = self.compute_inverse_graph(secondaries, graph_probabilities)

    def compute_expected_alpha_ratios(self, alpha_ratios_parameters):
        alpha_ratios_avg = alpha_ratios_parameters[:, :, 0] / \
                           (alpha_ratios_parameters[:, :, 0] + alpha_ratios_parameters[:, :, 1])
        norm_factors = np.sum(alpha_ratios_avg, axis=1)
        return alpha_ratios_avg / norm_factors[:, np.newaxis]

    def compute_inverse_graph(self, secondaries, graph_probabilities):
        inverse_graph = np.zeros((self.n_user_types, self.n_products, self.n_products))
        inverse_graph[:, np.arange(self.n_products), secondaries[:, 0]] = \
            graph_probabilities[:, np.arange(self.n_products), secondaries[:, 0]]  # primaries
        inverse_graph[:, np.arange(self.n_products), secondaries[:, 1]] = \
            graph_probabilities[:, np.arange(self.n_products), secondaries[:, 1]] * self.lambda_p  # secondaries
        return inverse_graph.T

    def find_optimal(self):
        optimal_configurations = np.zeros((self.n_user_types, self.n_products))
        optimal_rewards = np.zeros(self.n_user_types)
        for user_type in range(self.n_user_types):
            arms_shape = (self.n_arms,) * self.n_products
            expected_reward_per_configuration = np.zeros(arms_shape)
            for configuration, _ in np.ndenumerate(expected_reward_per_configuration):
                rewards = np.zeros(self.n_products)
                for start in range(self.n_products):
                    common_term = self.conversion_rates[user_type, start, configuration[start]] * \
                                  self.prices[start, configuration[start]] * \
                                  self.avg_products_sold[user_type, start, configuration[start]]
                    rewards[start] = common_term * (self.expected_alpha_ratios[user_type, start] +
                                                    self.compute_children_contribute([start], configuration, user_type))
                expected_reward_per_configuration[configuration] = np.sum(rewards)
            optimal_configurations[user_type] = np.unravel_index(np.argmax(expected_reward_per_configuration),
                                                                 expected_reward_per_configuration.shape)
            optimal_rewards[user_type] = np.max(expected_reward_per_configuration)
        return optimal_configurations, optimal_rewards

    def compute_children_contribute(self, predecessors, configuration, user_type):
        root = predecessors[-1]
        contribute = 0
        for child, prob in enumerate(self.inverse_graph[user_type, root]):
            if prob == 0 or child in predecessors:
                continue
            contribute += prob * self.conversion_rates[user_type, child, configuration[child]] * (
                    self.expected_alpha_ratios[user_type, child] +
                    self.compute_children_contribute([*predecessors, child], configuration, user_type)
            )
        return contribute
