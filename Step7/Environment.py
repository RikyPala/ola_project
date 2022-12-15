import json

from collections import namedtuple
from typing import List

import numpy as np

from RoundData import RoundData
from RoundsHistory import RoundsHistory


ContextConfig = namedtuple('ContextConfig', ['configuration', 'agg_classes'])


class Environment:
    def __init__(self):
        filepath = '../json/c_environment.json'
        with open(filepath, 'r', encoding='utf_8') as stream:
            env_features = json.load(stream)

        self.n_products = env_features['num_products']
        self.n_arms = env_features['num_arms']
        self.n_user_types = env_features['num_user_types']

        # REWARDS VARIABLES
        self.feature_probabilities = np.array(env_features['feature_probabilities'])
        self.n_features = len(self.feature_probabilities)
        self.user_probabilities = np.array([
            (1 - self.feature_probabilities[0]),
            (self.feature_probabilities[0]) * (1 - self.feature_probabilities[1]),
            (self.feature_probabilities[0]) * (self.feature_probabilities[1])])
        self.prices = np.array(env_features['prices'])
        self.conversion_rates = np.array(env_features['conversion_rates'])
        self.max_products_sold = np.array(env_features['max_products_sold'])

        # GRAPH VARIABLES
        self.lambda_p = env_features['lambda']
        self.alpha_ratios_parameters = np.array(env_features['alpha_ratios_parameters'])
        self.graph_probabilities = np.array(env_features['graph_probabilities'])
        self.secondaries = np.array(env_features['secondaries'])

    def draw_user_type(self):
        """
        Example
            - feature_1:
                TRUE -> Young
                FALSE -> Old
            - feature_2:
                TRUE -> Rich
                FALSE -> Poor
            - user_type:
                0 -> Old Rich/Poor
                1 -> Young Poor
                2 -> Young Rich
        :return: user_type
        """
        feature_1 = np.random.binomial(1, self.feature_probabilities[0])
        feature_2 = np.random.binomial(1, self.feature_probabilities[1])
        if not feature_1:
            user_type = 0
            if not feature_2:
                class_type = 0
            else:
                class_type = 1
        else:
            if not feature_2:
                user_type = 1
                class_type = 2
            else:
                user_type = 2
                class_type = 3
        return user_type, class_type

    def draw_starting_page(self, user_type, alpha_ratios):
        product = np.random.choice(6, p=alpha_ratios[user_type])
        return product

    def draw_alpha_ratios(self):
        alpha_ratios = np.random.beta(self.alpha_ratios_parameters[:, :, 0], self.alpha_ratios_parameters[:, :, 1])
        norm_factors = np.sum(alpha_ratios, axis=1)
        alpha_ratios = (alpha_ratios.T / norm_factors).T
        return alpha_ratios

    def get_pulled_arms_by_user_type(self, user_type, ctx_configs):
        for ctx_config in ctx_configs:
            if user_type in ctx_config.agg_classes:
                return ctx_config.configuration
        raise AssertionError('User class not found in the contexts')

    def round(self, ctx_configs: List[ContextConfig], learner_class=None, seed=0):
        s = seed
        if seed == 0:
            s = np.random.randint(1, 2**30)
        np.random.seed(s)

        result = RoundData(self.n_products, self.n_features, self.n_user_types)
        result.ctx_configs = ctx_configs

        daily_users = np.random.randint(3000, 4000)
        alpha_ratios = self.draw_alpha_ratios()
        rewards = np.zeros((self.n_user_types, self.n_products), dtype=int)

        for _ in range(daily_users):
            user_type, class_type = self.draw_user_type()
            result.users[class_type] += 1
            product = self.draw_starting_page(user_type=user_type, alpha_ratios=alpha_ratios)
            if product == 5:  # competitors' page
                continue
            visited = []
            to_visit = [product]
            result.first_clicks[class_type, product] += 1
            pulled_arms = self.get_pulled_arms_by_user_type(user_type, ctx_configs)
            while to_visit:
                current_product = to_visit.pop(0)
                visited.append(current_product)
                result.visits[class_type, current_product] += 1

                product_price = self.prices[current_product, pulled_arms[current_product]]

                buy = np.random.binomial(1, self.conversion_rates[
                    user_type, current_product, pulled_arms[current_product]])
                if not buy:
                    continue

                result.conversions[class_type, current_product] += 1
                products_sold = np.random.randint(
                    1, self.max_products_sold[user_type, current_product, pulled_arms[current_product]] + 1)
                result.sales[class_type, current_product] += products_sold
                rewards[user_type, current_product] += product_price * products_sold

                secondary_1 = self.secondaries[current_product, 0]
                success_1 = np.random.binomial(1, self.graph_probabilities[user_type, current_product, secondary_1])
                if success_1 and secondary_1 not in visited and secondary_1 not in to_visit:
                    to_visit.append(secondary_1)

                secondary_2 = self.secondaries[current_product, 1]
                success_2 = np.random.binomial(
                    1, self.lambda_p * self.graph_probabilities[user_type, current_product, secondary_2])
                if success_2 and secondary_2 not in visited and secondary_2 not in to_visit:
                    to_visit.append(secondary_2)

        result.prod_rewards = rewards / daily_users
        result.rewards = np.sum(result.prod_rewards, axis=1)

        if learner_class is not None:
            RoundsHistory.append(result, learner_class)

        return result
