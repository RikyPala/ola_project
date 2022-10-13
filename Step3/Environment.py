import numpy as np


class RoundData:
    def __init__(self, n_products):
        self.configuration = np.zeros(n_products, dtype=int)
        self.users = 0
        self.visits = np.zeros(n_products, dtype=int)
        self.conversions = np.zeros(n_products, dtype=int)
        self.reward = 0
        self.sales = np.zeros(n_products, dtype=int)
        self.prod_rewards = np.zeros(n_products)


class Environment:
    def __init__(self):
        self.n_products = 5
        self.n_arms = 4
        self.n_user_types = 3

        # REWARDS VARIABLES

        self.feature_probabilities = np.array([0.45, 0.65])
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

        self.prices = np.array([
            [80, 140, 180, 250],
            [40, 50, 80, 100],
            [160, 220, 300, 700],
            [100, 150, 250, 900],
            [35, 70, 84, 100]
        ])
        self.conversion_rates = np.array([
            [[0.77, 0.72, 0.56, 0.37],
             [0.80, 0.58, 0.45, 0.00],
             [0.62, 0.40, 0.00, 0.00],
             [0.67, 0.62, 0.51, 0.00],
             [0.55, 0.53, 0.00, 0.00]],

            [[0.81, 0.75, 0.72, 0.68],
             [0.68, 0.49, 0.32, 0.00],
             [0.58, 0.31, 0.00, 0.00],
             [0.77, 0.65, 0.60, 0.57],
             [0.62, 0.58, 0.40, 0.00]],

            [[0.60, 0.38, 0.00, 0.00],
             [0.71, 0.68, 0.65, 0.60],
             [0.81, 0.76, 0.72, 0.68],
             [0.76, 0.63, 0.54, 0.00],
             [0.53, 0.22, 0.00, 0.00]]
        ])
        self.max_products_sold = np.array([
            [[50, 40, 35, 30],
             [50, 40, 35, 30],
             [50, 40, 35, 30],
             [50, 40, 35, 30],
             [50, 40, 35, 30]],

            [[50, 40, 35, 30],
             [50, 40, 35, 30],
             [50, 40, 35, 30],
             [50, 40, 35, 30],
             [50, 40, 35, 30]],

            [[50, 40, 35, 30],
             [50, 40, 35, 30],
             [50, 40, 35, 30],
             [50, 40, 35, 30],
             [50, 40, 35, 30]],
        ])

        # GRAPH VARIABLES
        self.lambda_p = 0.8
        self.alpha_ratios_parameters = np.array([
            [[3, 7], [10, 2], [5, 6], [3, 3], [25, 13], [13, 2]],
            [[13, 15], [12, 20], [8, 6], [9, 3], [10, 12], [10, 3]],
            [[10, 7], [3, 12], [14, 17], [15, 8], [11, 6], [8, 3]]
        ])
        self.graph_probabilities = np.array([
            [[0, 0.40, 0.35, 0.40, 0.10],
             [0.30, 0, 0.25, 0.10, 0.15],
             [0.35, 0.15, 0, 0.20, 0.30],
             [0.10, 0.05, 0.20, 0, 0.15],
             [0.10, 0.30, 0.25, 0.10, 0]],

            [[0, 0.20, 0.50, 0.15, 0.45],
             [0.25, 0, 0.30, 0.40, 0.10],
             [0.45, 0.15, 0, 0.65, 0.15],
             [0.55, 0.50, 0.35, 0, 0.65],
             [0.15, 0.60, 0.35, 0.40, 0]],

            [[0, 0.25, 0.20, 0.45, 0.15],
             [0.35, 0, 0.10, 0.55, 0.50],
             [0.60, 0.45, 0, 0.50, 0.10],
             [0.15, 0.15, 0.35, 0, 0.10],
             [0.15, 0.25, 0.10, 0.55, 0]]
        ])
        self.secondaries = np.array([
            [1, 4],
            [2, 0],
            [0, 3],
            [4, 1],
            [3, 2]
        ])

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
            s = np.random.randint(1, 2**30)
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
                success_1 = np.random.binomial(1, self.graph_probabilities[user_type, current_product, secondary_1])
                if success_1 and secondary_1 not in visited and secondary_1 not in to_visit:
                    to_visit.append(secondary_1)

                secondary_2 = self.secondaries[current_product, 1]
                success_2 = np.random.binomial(
                    1, self.lambda_p * self.graph_probabilities[user_type, current_product, secondary_2])
                if success_2 and secondary_2 not in visited and secondary_2 not in to_visit:
                    to_visit.append(secondary_2)

        result.prod_rewards = rewards / daily_users
        result.reward = np.sum(result.prod_rewards)

        return result
