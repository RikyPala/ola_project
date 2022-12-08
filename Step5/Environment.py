import json

import numpy as np


class RoundData:
    def __init__(self, n_products):
        self.configuration = np.zeros(n_products, dtype=int)
        self.users = 0
        self.first_clicks = np.zeros(n_products, dtype=int)
        self.visits = np.zeros(n_products, dtype=int)
        self.secondary_visits = np.zeros((n_products, n_products, 2), dtype=int)
        self.conversions = np.zeros(n_products, dtype=int)
        self.reward = 0
        self.sales = np.zeros(n_products, dtype=int)
        self.prod_rewards = np.zeros(n_products)


class Environment:
    def __init__(self):
        filepath = '../json/environment.json'
        with open(filepath, 'r', encoding='utf_8') as stream:
            env_features = json.load(stream)

        self.n_products = env_features['num_products']
        self.n_arms = env_features['num_arms']
        self.n_user_types = env_features['num_user_types']

        # REWARDS VARIABLES
        self.feature_probabilities = np.array(env_features['feature_probabilities'])
        """
        User types:
            0 -> FALSE *
            1 -> TRUE FALSE
            2 -> TRUE TRUE
        """
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
        elif not feature_2:
            user_type = 1
        else:
            user_type = 2
        return user_type

    def draw_starting_page(self, user_type, alpha_ratios):
        product = np.random.choice(6, p=alpha_ratios[user_type])
        return product

    def draw_alpha_ratios(self):
        alpha_ratios = np.random.beta(self.alpha_ratios_parameters[:, :, 0], self.alpha_ratios_parameters[:, :, 1])
        norm_factors = np.sum(alpha_ratios, axis=1)
        alpha_ratios = (alpha_ratios.T / norm_factors).T
        return alpha_ratios

    def round(self, pulled_arms, seed=0):
        s = seed
        if seed == 0:
            s = np.random.randint(1, 2 ** 30)
        np.random.seed(s)

        result = RoundData(self.n_products)
        result.configuration = pulled_arms

        daily_users = np.random.randint(500, 1000)
        result.users = daily_users
        alpha_ratios = self.draw_alpha_ratios()
        rewards = np.zeros(self.n_products, dtype=int)

        for _ in range(daily_users):
            user_type = self.draw_user_type()
            product = self.draw_starting_page(user_type=user_type, alpha_ratios=alpha_ratios)
            if product == 5:  # competitors' page
                continue
            visited = []
            to_visit = [product]
            result.first_clicks[product] += 1

            while to_visit:
                current_product = to_visit.pop(0)
                visited.append(current_product)
                result.visits[current_product] += 1

                product_price = self.prices[current_product, pulled_arms[current_product]]

                buy = np.random.binomial(1, self.conversion_rates[
                    user_type, current_product, pulled_arms[current_product]])
                if not buy:
                    continue

                result.conversions[current_product] += 1
                products_sold = np.random.randint(
                    1, self.max_products_sold[user_type, current_product, pulled_arms[current_product]] + 1)
                result.sales[current_product] += products_sold
                rewards[current_product] += product_price * products_sold

                secondary_1 = self.secondaries[current_product, 0]
                if secondary_1 not in visited and secondary_1 not in to_visit:
                    result.secondary_visits[current_product, secondary_1, 1] += 1
                    success_1 = np.random.binomial(1, self.graph_probabilities[user_type, current_product, secondary_1])
                    if success_1:
                        result.secondary_visits[current_product, secondary_1, 0] += 1
                        to_visit.append(secondary_1)

                secondary_2 = self.secondaries[current_product, 1]
                if secondary_2 not in visited and secondary_2 not in to_visit:
                    result.secondary_visits[current_product, secondary_2, 1] += 1
                    success_2 = np.random.binomial(
                        1, self.lambda_p * self.graph_probabilities[user_type, current_product, secondary_2])
                    if success_2:
                        result.secondary_visits[current_product, secondary_2, 0] += 1
                        to_visit.append(secondary_2)

        result.prod_rewards = rewards / daily_users
        result.reward = np.sum(result.prod_rewards)

        return result
