import numpy as np


class Environment:
    def __init__(self):

        self.n_products = 5
        self.n_arms = 4
        self.n_user_types = 3

        self.prices = [10, 20, 30, 40]

        self.lambda_p = 0.8

        self.alpha_ratios = np.array([
            # [product_type == 0, product_type == 1, product_type == 2,
            #  product_type == 3, product_type == 4, competitor_page]
            [0.10, 0.15, 0.05, 0.30, 0.10, 0.30],  # user_type == 0
            [0.05, 0.10, 0.20, 0.20, 0.20, 0.25],  # user_type == 1
            [0.15, 0.05, 0.15, 0.35, 0.10, 0.20]   # user_type == 2
        ])

        self.daily_users = np.array([
            150,  # user_type == 0
            120,  # user_type == 1
            90   # user_type == 2
        ])

        self.demand_curves = np.array([  # [price == 10, price == 20, price == 30, price == 40]
            [
                [0.80, 0.70, 0.65, 0.30],  # product_type == 0
                [0.70, 0.65, 0.40, 0.30],  # product_type == 1
                [0.65, 0.60, 0.55, 0.45],  # product_type == 2
                [0.75, 0.70, 0.65, 0.30],  # product_type == 3
                [0.50, 0.40, 0.35, 0.30]   # product_type == 4
            ],  # user_type == 0
            [
                [0.75, 0.65, 0.50, 0.30],  # product_type == 0
                [0.65, 0.60, 0.50, 0.25],  # product_type == 1
                [0.80, 0.70, 0.65, 0.50],  # product_type == 2
                [0.70, 0.65, 0.45, 0.40],  # product_type == 3
                [0.55, 0.50, 0.30, 0.25]   # product_type == 4
            ],  # user_type == 1
            [
                [0.90, 0.65, 0.55, 0.50],  # product_type == 0
                [0.75, 0.60, 0.55, 0.40],  # product_type == 1
                [0.60, 0.30, 0.25, 0.20],  # product_type == 2
                [0.50, 0.40, 0.30, 0.25],  # product_type == 3
                [0.65, 0.30, 0.25, 0.10]   # product_type == 4
            ]   # user_type == 2
        ])

        self.products_sold = np.array([
            # [product_type == 0, product_type == 1, product_type == 2, product_type == 3, product_type == 4]
            [40, 55, 50, 20, 50],  # user_type == 0
            [30, 80, 20, 25, 40],  # user_type == 1
            [20, 40, 70, 60, 50]   # user_type == 2
        ])

        self.graph_probabilities = np.array([
            # [product_type == 0, product_type == 1, product_type == 2, product_type == 3, product_type == 4]
            [0.20, 0.40, 0.35, 0.40, 0.10],  # product_type == 0
            [0.30, 0.05, 0.25, 0.10, 0.15],  # product_type == 1
            [0.35, 0.15, 0.10, 0.20, 0.30],  # product_type == 2
            [0.10, 0.05, 0.20, 0.25, 0.15],  # product_type == 3
            [0.10, 0.30, 0.25, 0.10, 0.20]   # product_type == 4
        ])

        self.secondaries = np.array([
            [4, 2],  # product_type == 0
            [0, 2],  # product_type == 1
            [1, 3],  # product_type == 2
            [4, 0],  # product_type == 3
            [2, 3]   # product_type == 4
        ])

    def round(self, pulled_arms):
        rewards = np.zeros(self.n_products)
        for i in range(self.n_products):
            rewards[i] = np.random.binomial(1, self.probabilities[i, pulled_arms[i]])
        return rewards
