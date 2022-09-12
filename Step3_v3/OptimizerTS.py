import numpy as np
from Environment import Environment, RoundData
from TS import TS


class OptimizerTS:

    def __init__(self, env: Environment):
        self.n_products = env.n_products
        self.n_arms = env.n_arms
        self.prices = env.prices
        self.avg_products_sold = np.sum(
            (env.max_products_sold + 1) / 2 * np.expand_dims(env.user_probabilities, axis=1),
            axis=0)
        self.lambda_p = env.lambda_p
        self.alpha_ratios = np.sum(
            self.avg_alpha_ratios(env.alpha_ratios_parameters) * np.expand_dims(env.user_probabilities, axis=1),
            axis=0)
        self.graph_probabilities = np.sum(
            env.graph_probabilities * np.expand_dims(env.user_probabilities, axis=(1, 2)),
            axis=0)
        self.secondaries = env.secondaries

        self.conversion_rates_data = np.full((self.n_products, self.n_arms, 2), 0.)
        self.conversion_rates_est = np.full((self.n_products, self.n_arms), 1.)

        self.learner = TS((self.n_arms,) * self.n_products, gamma_rate=50000., prior_mean=500.)
        self.best_reward = 0.

    def avg_alpha_ratios(self, alpha_ratios_parameters):
        alpha_ratios_avg = alpha_ratios_parameters[:, :, 0] / \
                           (alpha_ratios_parameters[:, :, 0] + alpha_ratios_parameters[:, :, 1])
        norm_factors = np.sum(alpha_ratios_avg, axis=1)
        return (alpha_ratios_avg.T / norm_factors).T

    def draw_starting_page(self, alpha_ratios):
        product = np.random.choice(self.n_products + 1, p=alpha_ratios)
        return product

    def compute_node_probabilities(self, configuration):
        daily_users = 500

        idxs1 = np.arange(self.n_products)
        idxs2 = self.secondaries[:, 0]
        idxs3 = self.secondaries[:, 1]

        # all graph probabilities to 0 except those of the secondary products
        adj_graph_probabilities = np.zeros((self.n_products, self.n_products))
        adj_graph_probabilities[idxs1, idxs2] = self.graph_probabilities[idxs1, idxs2]
        adj_graph_probabilities[idxs1, idxs3] = self.graph_probabilities[idxs1, idxs3] * self.lambda_p

        node_visitors = np.zeros(self.n_products)

        for _ in range(daily_users):
            live_edge_graph = np.random.binomial(
                1, adj_graph_probabilities * self.conversion_rates_est[idxs1, configuration])
            start = self.draw_starting_page(self.alpha_ratios)
            if start == 5:  # competitors' page
                continue
            visited = []
            to_visit = [start]

            while to_visit:
                current_node = to_visit.pop(0)
                visited.append(current_node)
                node_visitors[current_node] += 1

                secondary_1 = self.secondaries[current_node, 0]
                if live_edge_graph[current_node, secondary_1] and \
                        secondary_1 not in visited and secondary_1 not in to_visit:
                    to_visit.append(secondary_1)

                secondary_2 = self.secondaries[current_node, 1]
                if live_edge_graph[current_node, secondary_2] and \
                        secondary_2 not in visited and secondary_2 not in to_visit:
                    to_visit.append(secondary_2)

        node_probabilities = node_visitors / daily_users

        return node_probabilities

    def evaluate_configuration(self, configuration):
        node_probabilities = self.compute_node_probabilities(configuration)
        print(node_probabilities)
        return np.sum(
            node_probabilities *
            self.conversion_rates_est[np.arange(self.n_products), configuration] *
            self.avg_products_sold *
            self.prices[np.arange(self.n_products), configuration])

    def update_conversion_rates(self, prod, price, visits, conversions):
        self.conversion_rates_data[prod, price, 0] += conversions
        self.conversion_rates_data[prod, price, 1] += visits
        self.conversion_rates_est[prod, price] = \
            self.conversion_rates_data[prod, price, 0] / self.conversion_rates_data[prod, price, 1]

    """ OPTIMIZATION """

    def initialize_configuration(self):
        return (1,) * self.n_products

    def update(self, round_data: RoundData):
        for prod in range(self.n_products):
            self.update_conversion_rates(prod, round_data.configuration[prod],
                                         round_data.visits[prod], round_data.conversions[prod])
        self.learner.update(round_data.configuration, round_data.reward)
        if round_data.reward > self.best_reward:
            self.best_reward = round_data.reward

    def optimize_round(self):
        best_configuration = (0,)
        best_reward = 0.

        for _ in range(self.n_arms ** self.n_products):
            configuration = self.learner.pull()
            print(configuration)
            reward = self.evaluate_configuration(configuration)
            print(reward)
            if reward > 0.85 * self.best_reward:
                return configuration
            self.learner.update(configuration, reward)
            if reward > best_reward:
                best_configuration = configuration
                best_reward = reward

        return best_configuration
