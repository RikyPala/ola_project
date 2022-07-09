import numpy as np
import itertools

from queue import Queue


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
            [0.15, 0.05, 0.15, 0.35, 0.10, 0.20]  # user_type == 2
        ])

        self.daily_users_ratios = np.array([
            0.40,  # user_type == 0
            0.35,  # user_type == 1
            0.25  # user_type == 2
        ])

        self.products_sold = np.array([
            # [product_type == 0, product_type == 1, product_type == 2, product_type == 3, product_type == 4]
            [40, 55, 50, 20, 50],  # user_type == 0
            [30, 80, 20, 25, 40],  # user_type == 1
            [20, 40, 70, 60, 50]   # user_type == 2
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

        for (user_type, product) in itertools.product(range(self.n_user_types), range(self.n_products)):

            for user in range(round(daily_users * self.daily_users_ratios[user_type] * self.alpha_ratios[user_type, product])):
                # TODO: Eventually implement reservation ranges basing on user_type
                reservation_price = np.random.randint(0, 50)

                visited = []
                to_visit = [product]

                while to_visit:
                    current_product = to_visit.pop(0)
                    visited.append(current_product)

                    product_price = self.prices[pulled_arms[current_product]]
                    if product_price > reservation_price:
                        continue
                    rewards[current_product] += (
                            product_price *
                            self.products_sold[user_type, current_product] *
                            self.demand_curves[user_type, current_product, pulled_arms[current_product]]
                    )

                    secondary_1 = self.secondaries[current_product, 0]
                    success_1 = np.random.binomial(1, self.graph_probabilities[current_product, secondary_1])
                    if success_1 and secondary_1 not in visited and secondary_1 not in to_visit:
                        to_visit.append(secondary_1)

                    secondary_2 = self.secondaries[current_product, 1]
                    success_2 = np.random.binomial(1, self.lambda_p * self.graph_probabilities[current_product, secondary_2])
                    if success_2 and secondary_2 not in visited and secondary_2 not in to_visit:
                        to_visit.append(secondary_2)

        return rewards / daily_users
