from abc import abstractmethod

import numpy as np

from Environment import Environment
from RoundData import RoundData


class Learner:
    TYPE0_0 = 0
    TYPE0_1 = 1
    TYPE1 = 2
    TYPE2 = 3

    def __init__(self, env: Environment, feature_1=None, feature_2=None):

        self.n_products = env.n_products
        self.n_arms = env.n_arms
        self.prices = env.prices
        self.lambda_p = env.lambda_p

        self.agg_classes = self.assign_agg_classes(feature_1, feature_2)

        fp_1 = env.feature_probabilities[0]
        fp_2 = env.feature_probabilities[1]
        self.classes_probabilities = np.array(
            [(1 - fp_1) * (1 - fp_2), (1 - fp_1) * fp_2, fp_1 * (1 - fp_2), fp_1 * fp_2])

        gp = np.array([env.graph_probabilities[0], env.graph_probabilities[0], *env.graph_probabilities[1:]])
        self.graph_probabilities = np.sum(
            gp[self.agg_classes] * np.expand_dims(self.classes_probabilities[self.agg_classes], axis=(1, 2)),
            axis=0) / np.sum(self.classes_probabilities[self.agg_classes])

        self.alpha_ratios_data = np.full((self.n_products + 1, 2), 0)
        self.alpha_ratios_est = np.full(self.n_products + 1, 1. / (self.n_products + 1))

        self.initialize = True
        self.avg_products_sold_data = np.full((self.n_products, self.n_arms, 2), 0)
        self.avg_products_sold_est = np.full((self.n_products, self.n_arms), np.inf)

        self.n_simulations = 300
        self.marginal_rewards = np.zeros((env.n_products, env.n_arms))
        self.secondaries = env.secondaries
        self.pulled_rounds = np.zeros((self.n_products, self.n_arms))

    def pull(self):
        exp_conversion_rates = self.sample()
        alpha_ratios = np.array([self.alpha_ratios_est[:self.n_products]] * self.n_arms).transpose()
        exp_rewards = (exp_conversion_rates * self.prices * self.avg_products_sold_est
                       + self.marginal_rewards) * alpha_ratios
        configuration = np.argmax(exp_rewards, axis=1)
        if self.initialize:
            configuration = np.zeros_like(configuration)
        return configuration

    def assign_agg_classes(self, feature_1, feature_2):
        agg_classes = []
        if feature_1 is None:
            if feature_2 is None:
                agg_classes = [self.TYPE0_0, self.TYPE0_1, self.TYPE1, self.TYPE2]
            elif not feature_2:
                agg_classes = [self.TYPE0_0, self.TYPE1]
            elif feature_2:
                agg_classes = [self.TYPE0_1, self.TYPE2]
        elif not feature_1:
            if feature_2 is None:
                agg_classes = [self.TYPE0_0, self.TYPE0_1]
            elif not feature_2:
                agg_classes = [self.TYPE0_0]
            elif feature_2:
                agg_classes = [self.TYPE0_1]
        elif feature_1:
            if feature_2 is None:
                agg_classes = [self.TYPE1, self.TYPE2]
            elif not feature_2:
                agg_classes = [self.TYPE1]
            elif feature_2:
                agg_classes = [self.TYPE2]
        if not agg_classes:
            raise NotImplementedError('Feature passed in the constructor are of the wrong type')
        return agg_classes

    @abstractmethod
    def update(self, round_data):
        pass

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def get_means(self):
        pass

    def get_configuration_by_agg_classes(self, ctx_configs):
        for ctx_config in ctx_configs:
            if all(clss in ctx_config.agg_classes for clss in self.agg_classes):
                return ctx_config.configuration
        raise AssertionError('Aggregated classes not found in the contexts')

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

    def update_alpha_ratios(self, results: RoundData):
        self.alpha_ratios_data[:self.n_products, 0] += np.sum(results.first_clicks[self.agg_classes], axis=0)
        self.alpha_ratios_data[self.n_products, 0] += \
            np.sum(results.users[self.agg_classes]) - np.sum(results.first_clicks[self.agg_classes])
        self.alpha_ratios_data[:, 1] += np.sum(results.users[self.agg_classes], axis=0, dtype=np.int32)
        self.alpha_ratios_est = self.alpha_ratios_data[:, 0] / self.alpha_ratios_data[:, 1]

    def update_avg_products_sold(self, configuration, results: RoundData):
        max_found = 0.
        for prod in range(self.n_products):
            sales = self.avg_products_sold_data[prod, configuration[prod], 0]
            conversions = self.avg_products_sold_data[prod, configuration[prod], 1]
            sales += np.sum(results.sales[self.agg_classes, prod], axis=0)
            conversions += np.sum(results.conversions[self.agg_classes, prod], axis=0)
            if conversions > 0:
                est = sales / conversions
                self.avg_products_sold_est[prod, configuration[prod]] = est
                if self.initialize:
                    max_found = max(max_found, est)
        if self.initialize:
            self.avg_products_sold_est = np.clip(self.avg_products_sold_est, None, 0.8*max_found)

    def update_marginal_reward(self, configuration):
        reaching_probabilities = self.compute_reaching_probabilities(configuration)
        idxs = np.arange(self.n_products)
        for prod in range(self.n_products):
            old_marginal_reward = self.marginal_rewards[prod, configuration[prod]]
            marginal_reward = np.sum(
                self.get_means()[prod, configuration[prod]] *
                reaching_probabilities[prod] *
                self.get_means()[idxs, configuration] *
                self.prices[idxs, configuration] *
                self.avg_products_sold_est[idxs, configuration]
            )
            n_pulls = self.pulled_rounds[prod, configuration[prod]]
            self.marginal_rewards[prod, configuration[prod]] = \
                (old_marginal_reward * (n_pulls - 1) + marginal_reward) / n_pulls

    def update_estimates(self, configuration, results: RoundData):
        self.update_alpha_ratios(results)
        self.update_avg_products_sold(configuration, results)
        self.update_marginal_reward(configuration)
        if self.initialize:
            self.initialize = False
