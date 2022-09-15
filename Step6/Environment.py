import numpy as np


class RoundData:
    def __init__(self, n_products):
        self.configuration = np.zeros(n_products, dtype=int)
        self.users = 0
        self.visits = np.zeros(n_products, dtype=int)
        self.conversions = np.zeros(n_products, dtype=int)
        self.sales = np.zeros(n_products, dtype=int)
        self.prod_rewards = np.zeros(n_products)
        self.reward = 0

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
            [160, 220, 300, 400],
            [100, 200, 250, 300],
            [35, 70, 84, 100]
        ])
        self.conversion_rates = np.array([
            [#FIRST PHASE
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
            ],
            [#SECOND PHASE
            [[0.77, 0.60, 0.56, 0.37],
             [0.65, 0.46, 0.45, 0.00],
             [0.62, 0.40, 0.30, 0.00],
             [0.76, 0.62, 0.51, 0.00],
             [0.55, 0.20, 0.10, 0.00]],

            [[0.91, 0.75, 0.72, 0.68],
             [0.68, 0.49, 0.32, 0.15],
             [0.78, 0.31, 0.00, 0.00],
             [0.77, 0.65, 0.60, 0.20],
             [0.62, 0.45, 0.40, 0.00]],

            [[0.40, 0.38, 0.00, 0.00],
             [0.71, 0.68, 0.65, 0.60],
             [0.91, 0.75, 0.72, 0.68],
             [0.76, 0.63, 0.54, 0.00],
             [0.53, 0.32, 0.10, 0.005]]
            ],
            [#THIRD PHASE
            [[0.90, 0.85, 0.56, 0.37],
             [0.70, 0.58, 0.45, 0.05],
             [0.62, 0.50, 0.40, 0.00],
             [0.67, 0.62, 0.51, 0.00],
             [0.55, 0.53, 0.00, 0.00]],

            [[0.71, 0.60, 0.52, 0.00],
             [0.68, 0.49, 0.32, 0.00],
             [0.58, 0.42, 0.00, 0.00],
             [0.77, 0.65, 0.60, 0.57],
             [0.62, 0.00, 0.00, 0.00]],

            [[0.60, 0.50, 0.40, 0.00],
             [0.71, 0.38, 0.28, 0.60],
             [0.81, 0.76, 0.72, 0.68],
             [0.65, 0.63, 0.54, 0.00],
             [0.53, 0.22, 0.11, 0.00]]
            ],
        ])
        self.max_products_sold = np.array([
            [7, 10, 8, 6, 4],
            [11, 8, 4, 5, 3],
            [6, 7, 9, 9, 6]
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

