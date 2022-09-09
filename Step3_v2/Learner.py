import itertools

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
        self.graph_probabilities = np.sum(
            env.graph_probabilities * np.expand_dims(env.user_probabilities, axis=(1, 2)),
            axis=0
        )
        self.secondaries = env.secondaries
        self.estimated_conversion_rates = np.zeros((self.n_products, self.n_arms))

        self.collected_rewards = [[] for _ in range(self.n_products)]

        self.last_configuration = np.zeros(self.n_products)

    def draw_alpha_ratios(self):
        alpha_ratios = np.random.beta(self.alpha_ratios_parameters[:, 0], self.alpha_ratios_parameters[:, 1])
        norm_factors = np.sum(alpha_ratios)
        alpha_ratios = alpha_ratios / norm_factors
        return alpha_ratios

    def draw_starting_page(self, alpha_ratios):
        product = np.random.choice(self.n_products + 1, p=alpha_ratios)
        return product

    def optimize1(self, candidate_arms):

        best_reward = 0
        best_configuration = np.zeros(self.n_products)

        for configuration in itertools.product(*candidate_arms):
            daily_users = 100
            reward = 0
            alpha_ratios = self.draw_alpha_ratios()

            for _ in range(daily_users):
                product = self.draw_starting_page(alpha_ratios)
                if product == 5:  # competitors' page
                    continue
                visited = []
                to_visit = [product]

                while to_visit:
                    current_product = to_visit.pop(0)
                    visited.append(current_product)

                    product_price = self.prices[configuration[current_product]]

                    buy = np.random.binomial(1, self.estimated_conversion_rates[
                        current_product, configuration[current_product]])
                    if not buy:
                        continue

                    products_sold = np.random.randint(0, self.max_products_sold[current_product])
                    reward += product_price * products_sold

                    secondary_1 = self.secondaries[current_product, 0]
                    success_1 = np.random.binomial(1, self.graph_probabilities[current_product, secondary_1])
                    if success_1 and secondary_1 not in visited and secondary_1 not in to_visit:
                        to_visit.append(secondary_1)

                    secondary_2 = self.secondaries[current_product, 1]
                    success_2 = np.random.binomial(
                        1, self.lambda_p * self.graph_probabilities[current_product, secondary_2])
                    if success_2 and secondary_2 not in visited and secondary_2 not in to_visit:
                        to_visit.append(secondary_2)

            if reward > best_reward:
                best_reward = reward
                best_configuration = configuration

        return best_configuration

    def optimize2(self, candidate_arms):

        best_reward = 0
        best_configuration = np.zeros(self.n_products)
        """
        print("\nCandidate Arms:")
        print(candidate_arms)
        """
        for configuration in itertools.product(*candidate_arms):
            node_probabilities = self.compute_node_probabilities(configuration)
            """
            print("\nConfiguration: " + str(configuration))
            print("\nPrices:")
            print(self.prices[np.array(configuration)])
            print("\nConversion Rates:")
            print(self.estimated_conversion_rates[np.arange(self.n_products), configuration])
            print("\nNode Probabilities:")
            print(node_probabilities)
            """
            reward = np.sum(
                self.prices[np.array(configuration)] *
                self.estimated_conversion_rates[np.arange(self.n_products), configuration] *
                node_probabilities
            )
            # print("\nReward: " + str(reward))
            # print("-----------------")

            if reward > best_reward:
                best_reward = reward
                best_configuration = configuration
        """
        print("\nBest Configuration:")
        print(best_configuration)
        print("\n####################################################################\n\n")
        """
        return best_configuration

    def compute_node_probabilities(self, configuration):

        daily_users = 100
        alpha_ratios = self.draw_alpha_ratios()

        idxs1 = np.arange(self.n_products)
        idxs2 = self.secondaries[:, 0]
        idxs3 = self.secondaries[:, 1]

        # all graph probabilities to 0 except those of the secondary products
        adj_graph_probabilities = np.zeros((self.n_products, self.n_products))
        adj_graph_probabilities[idxs1, idxs2] = self.graph_probabilities[idxs1, idxs2]
        adj_graph_probabilities[idxs1, idxs3] = self.graph_probabilities[idxs1, idxs3] * self.lambda_p
        """
        print("\nidxs1:")
        print(idxs1)
        print("\nidxs2:")
        print(idxs2)
        print("\nidxs3:")
        print(idxs3)
        print("\nGraph Probabilities:")
        print(self.graph_probabilities)
        print("\nAdjusted Graph Probabilities:")
        print(adj_graph_probabilities)
        """
        node_visitors = np.zeros(self.n_products)

        for _ in range(daily_users):
            live_edge_graph = np.random.binomial(
                1, adj_graph_probabilities * self.estimated_conversion_rates[idxs1, configuration]
            )
            # print("\nLive Edge Graph:")
            # print(live_edge_graph)
            start = self.draw_starting_page(alpha_ratios)
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

    def update_observations(self, rewards):
        for i in range(self.n_products):
            self.collected_rewards[i].append(rewards[i])
