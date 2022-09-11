import numpy as np

class Environment_UCB:


    def __init__(self):

        self.n_products = 5
        self.n_arms = 4
        self.prices = np.array([
            [8.0, 14.0, 18.0, 25.0],
            [4.0, 5.0, 8.0, 10.0],
            [16.0, 22.0, 30.0, 40.0],
            [10.0, 20.0, 25.0, 30.0],
            [3.5, 7.0, 8.4, 10.0],
        ])
        self.conversion_rates = np.array([
                #PRODUCTS
                [0.60, 0.50, 0.65, 0.65, 0.65],#ARMS
                [0.55, 0.40, 0.50, 0.60, 0.50],
                [0.42, 0.36, 0.39, 0.55, 0.38],
                [0.35, 0.24, 0.30, 0.43, 0.28]
        ])

        # GRAPH VARIABLES
        self.lambda_p = 0.6
        self.alpha_ratios_parameters = np.array(
            [[[25, 75], [15, 85], [10, 90], [5, 95], [15, 85], [30, 70]]]
        )


        self.secondaries = np.array([
            [1, 4],  # product_type == 0
            [2, 0],  # product_type == 1
            [0, 3],  # product_type == 2
            [4, 1],  # product_type == 3
            [3, 2]   # product_type == 4
        ])

        self.product_sold = np.array([
            [20, 15, 10, 7],
            [20, 18, 16, 13],
            [25, 22, 20, 16],
            [12, 10, 9, 7],
            [30, 25, 20, 15]
        ])

        self.graph_probabilities = np.array([
            [0, 0.24, 0.25, 0.450, 0.17],
            [0.30, 0, 0.55, 0.25, 0.40],
            [0.10, 0.24, 0, 0.55, 0.15],
            [0.55, 0.40, 0.30, 0, 0.55],
            [0.05, 0.40, 0.45, 0.30, 0]
        ])

        self.expected_alpha_ratios = np.array([0.25, 0.15, 0.1, 0.05, 0.15, 0.3])

    def draw_starting_page(self, alpha_ratios):
        product = np.random.choice(self.n_products+1, p= alpha_ratios)
        return product

    def draw_alpha_ratios(self):
        alpha_ratios = np.random.beta(self.alpha_ratios_parameters[:,:, 0], self.alpha_ratios_parameters[:,:, 1])
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
            if product == self.n_products:  # competitors' page
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
        return np.nan_to_num(buyers/visitors)
