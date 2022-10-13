from abc import abstractmethod

import numpy as np

from Step3.Environment import Environment


class Learner:

    def __init__(self, env: Environment):

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
        self.n_simulations = 300
        self.marginal_rewards = np.zeros((env.n_products, env.n_arms))
        self.secondaries = env.secondaries
        self.pulled_rounds = np.zeros((self.n_products, self.n_arms))

    def pull(self):
        exp_conversion_rates = self.sample()
        exp_rewards = exp_conversion_rates * self.prices * self.max_products_sold + self.marginal_rewards
        configuration = np.argmax(exp_rewards, axis=1)
        self.pulled_rounds[np.arange(self.n_products), configuration] += 1
        return configuration

    @abstractmethod
    def update(self, results):
        pass

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def get_means(self):
        pass

    def compute_reaching_probabilities(self, configuration):
        reaches = np.zeros((self.n_products, self.n_products))
        for prod in range(self.n_products):
            for _ in range(self.n_simulations):
                visited = self.simulation(prod, configuration)
                visited[prod] = 0
                reaches[prod] += visited
        return reaches / self.n_simulations

    def simulation(self, starting_node, configuration):
        visited = np.zeros(self.n_products)
        to_visit = [starting_node]

        while to_visit:
            current_product = to_visit.pop(0)
            visited[current_product] += 1

            buy = np.random.binomial(1, self.get_means()[current_product, configuration[current_product]])
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

    def update_marginal_reward(self, configuration):
        reaching_probabilities = self.compute_reaching_probabilities(configuration)
        idxs = np.arange(self.n_products)
        for prod in range(self.n_products):
            old_marginal_reward = self.marginal_rewards[prod, configuration[prod]]
            marginal_reward = np.sum(
                self.get_means()[idxs, configuration[prod]] *
                reaching_probabilities[prod] *
                self.get_means()[idxs, configuration] *
                self.prices[idxs, configuration] *
                self.max_products_sold[idxs, configuration]
            )
            n_pulls = self.pulled_rounds[prod, configuration[prod]]
            self.marginal_rewards[prod, configuration[prod]] = \
                (old_marginal_reward * (n_pulls - 1) + marginal_reward) / n_pulls
