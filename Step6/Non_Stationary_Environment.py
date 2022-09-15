from Environment import Environment, RoundData
import numpy as np

class Non_Stationary_Environment(Environment):

    def __init__(self, n_arms, horizon):
        super().__init__()
        self.t = 0
        self.n_phases = len(self.conversion_rates)
        self.phases_size = horizon/self.n_phases


    def round(self, pulled_arms, seed=0):
        s = seed
        if seed == 0:
            s = np.random.randint(1, 2 ** 30)
        np.random.seed(s)

        current_phase = int(self.t / self.phases_size)

        result = RoundData(self.n_products)
        result.configuration = pulled_arms

        daily_users = np.random.randint(500, 1000)
        result.users = daily_users
        rewards = np.zeros(self.n_products, dtype=int)
        alpha_ratios = self.draw_alpha_ratios()

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
                    current_phase, user_type, current_product, pulled_arms[current_product]])
                if not buy:
                    continue

                result.conversions[current_product] += 1
                products_sold = np.random.randint(1, self.max_products_sold[user_type, current_product] + 1)
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
