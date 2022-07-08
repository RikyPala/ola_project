import numpy as np


class Environment:
    def __init__(self):

        self.n_products = 5
        self.n_arm = 4

        self.T = 300
        self.n_experiments = 100

        self.probabilities = []

        self.prices = [10, 20, 30, 40]

        self.alpha_ratios = [  # [alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_0]
            [0.10, 0.15, 0.05, 0.30, 0.10, 0.30],  # user_type == 0
            [0.05, 0.10, 0.20, 0.20, 0.20, 0.25],  # user_type == 1
            [0.15, 0.05, 0.15, 0.35, 0.10, 0.20]   # user_type == 2
        ]

        self.daily_users = [
            150,  # user_type == 0
            120,  # user_type == 1
            90    # user_type == 2
        ]

        self.demand_curves = np.array([
            [
                [0.80, 0.70, 0.65, 0.30],  # product_type == 1
                [0.70, 0.65, 0.40, 0.30],  # product_type == 2
                [0.65, 0.60, 0.55, 0.45],  # product_type == 3
                [0.75, 0.70, 0.65, 0.30],  # product_type == 4
                [0.50, 0.40, 0.35, 0.30]   # product_type == 5
            ],  # user_type == 0
            [
                [0.75, 0.65, 0.50, 0.30],  # product_type == 1
                [0.65, 0.60, 0.50, 0.25],  # product_type == 2
                [0.80, 0.70, 0.65, 0.50],  # product_type == 3
                [0.70, 0.65, 0.45, 0.40],  # product_type == 4
                [0.55, 0.50, 0.30, 0.25]   # product_type == 5
            ],  # user_type == 1
            [
                [0.80, 0.70, 0.65, 0.50],  # product_type == 1
                [0.75, 0.60, 0.55, 0.40],  # product_type == 2
                [0.60, 0.30, 0.25, 0.20],  # product_type == 3
                [0.50, 0.40, 0.30, 0.25],  # product_type == 4
                [0.65, 0.30, 0.25, 0.10]   # product_type == 5
            ]   # user_type == 2
        ])

    def round(self, pulled_arms):
        rewards = np.zeros(self.n_products)
        for i in range(self.n_products):
            rewards[i] = np.random.binomial(1, self.probabilities[i, pulled_arms[i]])
        return rewards
