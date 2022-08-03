import numpy as np
import itertools

from queue import Queue


class Environment:
    def __init__(self):

        self.n_products = 5
        self.n_arms = 4
        self.n_user_types = 3

        # REWARDS VARIABLES
        self.prices = np.array([10, 20, 30, 40])
        self.daily_users_ratios = np.array([0.40, 0.35, 0.25])  # TODO: remove and add attributes distributions
        self.conversion_rates = np.array([
            [
                [0.80, 0.75, 0.90],
                [0.70, 0.65, 0.75],
                [0.65, 0.80, 0.60],
                [0.75, 0.70, 0.50],
                [0.50, 0.55, 0.65]
            ],
            [
                [0.70, 0.65, 0.65],
                [0.65, 0.60, 0.60],
                [0.60, 0.70, 0.30],
                [0.70, 0.65, 0.40],
                [0.40, 0.50, 0.30]
            ],
            [
                [0.65, 0.50, 0.55],
                [0.40, 0.40, 0.45],
                [0.45, 0.40, 0.25],
                [0.45, 0.35, 0.30],
                [0.35, 0.30, 0.25]
            ],
            [
                [0.30, 0.30, 0.50],
                [0.30, 0.25, 0.40],
                [0.35, 0.30, 0.20],
                [0.20, 0.30, 0.25],
                [0.30, 0.25, 0.10]
            ]
        ])
        self.max_products_sold = np.array([
            [40, 30, 20],
            [55, 35, 40],
            [50, 20, 60],
            [20, 40, 30],
            [50, 40, 50]
        ])

        expected_purchases = self.daily_users_ratios * self.conversion_rates * (self.max_products_sold / 2)
        expected_purchases = np.sum(expected_purchases, axis=2).T
        expected_rewards = expected_purchases * self.prices

        self.optimals = np.max(expected_rewards, axis=1)

        # GRAPH VARIABLES
        self.lambda_p = 0.8
        self.alpha_ratios = np.array([  # TODO: rewrite as dirichlet probability distributions
            # [product_type == 0, product_type == 1, product_type == 2,
            #  product_type == 3, product_type == 4, competitor_page]
            [0.10, 0.15, 0.05, 0.15, 0.10, 0.45],  # user_type == 0
            [0.05, 0.10, 0.20, 0.20, 0.20, 0.25],  # user_type == 1
            [0.15, 0.10, 0.15, 0.15, 0.10, 0.35]  # user_type == 2
        ])
        self.graph_probabilities = np.array([
            # [product_type == 0, product_type == 1, product_type == 2, product_type == 3, product_type == 4]
            [0, 0.40, 0.35, 0.40, 0.10],  # product_type == 0
            [0.30, 0, 0.25, 0.10, 0.15],  # product_type == 1
            [0.35, 0.15, 0, 0.20, 0.30],  # product_type == 2
            [0.10, 0.05, 0.20, 0, 0.15],  # product_type == 3
            [0.10, 0.30, 0.25, 0.10, 0]  # product_type == 4
        ])
        self.secondaries = np.array([
            [4, 2],  # product_type == 0
            [0, 2],  # product_type == 1
            [1, 3],  # product_type == 2
            [4, 0],  # product_type == 3
            [2, 3]  # product_type == 4
        ])

    def round(self, pulled_arms):

        daily_users = np.random.randint(10, 200)

        rewards = np.zeros(self.n_products, dtype=int)

        # TODO: Draw from attributes distributions and change for accordingly
        for (user_type, product) in itertools.product(range(self.n_user_types), range(self.n_products)):

            for user in range(round(daily_users * self.daily_users_ratios[user_type] * self.alpha_ratios[user_type, product])):

                visited = []
                to_visit = [product]

                while to_visit:
                    current_product = to_visit.pop(0)
                    visited.append(current_product)

                    product_price = self.prices[pulled_arms[current_product]]

                    buy = np.random.binomial(1, self.conversion_rates[pulled_arms[current_product], current_product, user_type])
                    if not buy:
                        continue

                    products_sold = np.random.randint(0, self.max_products_sold[current_product, user_type])
                    rewards[current_product] += product_price * products_sold

                    secondary_1 = self.secondaries[current_product, 0]
                    success_1 = np.random.binomial(1, self.graph_probabilities[current_product, secondary_1])
                    if success_1 and secondary_1 not in visited and secondary_1 not in to_visit:
                        to_visit.append(secondary_1)

                    secondary_2 = self.secondaries[current_product, 1]
                    success_2 = np.random.binomial(1, self.lambda_p * self.graph_probabilities[current_product, secondary_2])
                    if success_2 and secondary_2 not in visited and secondary_2 not in to_visit:
                        to_visit.append(secondary_2)

        return rewards / daily_users
