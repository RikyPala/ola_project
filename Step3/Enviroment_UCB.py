import numpy as np
import itertools

from queue import Queue


class Environment_UCB:
    def __init__(self):

        self.n_products = 5
        self.n_arms = 4

        self.prices = np.array([10, 12, 15, 17])

        self.conversion_rates = np.array([
                #PRODUCTS
                [0.80, 0.80, 0.80, 0.80, 0.80], #ARMS


                [0.70, 0.70, 0.70, 0.70, 0.70],


                [0.60, 0.60, 0.60, 0.60, 0.60],


                [0.50, 0.50, 0.50, 0.50, 0.50],

        ])
        self.products_sold = np.array([
            [20, 18, 15, 14],
            [30, 28, 25, 23],
            [35, 33, 32, 30],
            [60, 58, 55, 50],
            [18, 16, 15, 12]])

        # GRAPH VARIABLES
        self.lambda_p = 0.8
        self.alpha_ratios_parameters = np.array(
            [[[3, 7], [10, 2], [5, 6], [3, 3], [25, 13], [13, 2]]]

        )

        self.expected_alpha_ratios = np.array([0.3, 10/12, 5/11, 0.5, 25/38, 13/15])

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

    def draw_starting_page(self, alpha_ratios):
        product = np.random.choice(6, p=alpha_ratios)
        return product

    def draw_alpha_ratios(self):
        alpha_ratios = np.random.beta(self.alpha_ratios_parameters[:,:, 0], self.alpha_ratios_parameters[:,:, 1])
        print("ALPHA RATIO MEAN VALUES")
        print(np.random.beta(self.alpha_ratios_parameters[:,:, 0], self.alpha_ratios_parameters[:,:, 1]).mean())
        norm_factors = np.sum(alpha_ratios, axis=1)
        alpha_ratios = (alpha_ratios.T / norm_factors).T
        alpha_ratios = np.reshape(alpha_ratios, -1)
        return alpha_ratios

    # TODO: the user types are still present!

    def round(self, pulled_arms):

        daily_users = np.random.randint(400, 500)
        alpha_ratios = self.draw_alpha_ratios()
        print("ALPHA RATIOSS")
        print(alpha_ratios)
        buyers = np.zeros(len(pulled_arms))
        visitors = np.zeros(len(pulled_arms))

        for _ in range(daily_users):
            product = self.draw_starting_page(alpha_ratios=alpha_ratios)
            if product == 5:  # competitors' page
                continue
            visited = []
            to_visit = [product]

            while to_visit:
                current_product = to_visit.pop(0)
                visited.append(current_product)
                visitors[current_product] += 1
                buy = np.random.binomial(1, self.conversion_rates[pulled_arms[current_product], current_product])
                if not buy:
                    continue

                buyers[current_product] += 1

                secondary_1 = self.secondaries[current_product, 0]
                success_1 = np.random.binomial(1, self.graph_probabilities[current_product, secondary_1])
                if success_1 and secondary_1 not in visited and secondary_1 not in to_visit:
                    to_visit.append(secondary_1)

                secondary_2 = self.secondaries[current_product, 1]
                success_2 = np.random.binomial(
                    1, self.lambda_p * self.graph_probabilities[current_product, secondary_2])
                if success_2 and secondary_2 not in visited and secondary_2 not in to_visit:
                    to_visit.append(secondary_2)

        print("BUYERSSS")
        print(buyers)
        print("VISITORSSS")
        print(visitors)
        return np.nan_to_num(buyers/visitors), alpha_ratios
