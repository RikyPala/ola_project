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
        self.n_simulations = 100
        self.marginal_rewards = np.zeros(env.n_products, env.n_arms)

    def pull(self, results):
        pulled_arms = []
        upper_conf = self.update(results)
        for prod in range(self.n_products):
            candidates = upper_conf[prod]
            j = np.random.choice(np.where(candidates == candidates.max())[0])
            self.pulled_rounds[prod][j] += 1
            pulled_arms.append(j)

        return pulled_arms

    @abstractmethod
    def update(self, results):
        pass

    @abstractmethod
    def sample(self):
        pass

    def compute_nearby_contribution(self, configuration):

        visits_starting_node = np.zeros(self.n_products, self.n_arms)
        for prod in range(self.n_products):
            for i in range(self.n_simulations):
                visited = self.simulation(prod, configuration)
                for j in range(self.n_products):
                    if (visited[j] == 1) and j != prod:
                        visits_starting_node[prod][j] += 1

        return visits_starting_node / self.n_simulations

    def simulation(self, starting_node, pulled_arms):
        visited = np.zeros(self.n_products)
        to_visit = [starting_node]

        while to_visit:
            current_product = to_visit.pop(0)
            visited[current_product] += 1

            buy = np.random.binomial(1, self.empirical_means[current_product][pulled_arms[current_product]])
            if not buy:
                continue

            secondary_1 = self.secondaries[current_product, 0]
            success_1 = np.random.binomial(1, self.graph_probabilities[current_product, secondary_1])
            if success_1 and visited[secondary_1] == 0 and secondary_1 not in to_visit:
                to_visit.append(secondary_1)

            secondary_2 = self.secondaries[current_product, 1]
            success_2 = np.random.binomial(
                1, self.lambda_p * self.graph_probabilities[current_product, secondary_2])
            if success_2 and visited[secondary_2] == 0 and secondary_2 not in to_visit:
                to_visit.append(secondary_2)

        return visited



