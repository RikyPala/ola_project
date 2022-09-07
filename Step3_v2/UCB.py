import itertools
import numpy as np

from Environment import Environment
from Learner import Learner


class UCB(Learner):

    def __init__(self, env: Environment):
        super().__init__(env)
        self.empirical_means = np.zeros((self.n_products, self.n_arms))
        self.confidence = np.ones((self.n_products, self.n_arms)) * np.inf
        self.c = 200
        self.pulled_rounds = np.zeros((self.n_products, self.n_arms))
        self.t = 1

    def pull_arms(self):
        upper_conf = self.empirical_means + self.c * self.confidence

        if self.t >= self.n_arms + 1:
            candidate_arms_1 = np.argmax(upper_conf, axis=1)
            upper_conf[np.arange(self.n_products), candidate_arms_1] = -np.inf
            candidate_arms_2 = np.argmax(upper_conf, axis=1)
            candidate_arms = np.dstack((candidate_arms_1, candidate_arms_2))[0]
            pulled_arms = self.optimize(candidate_arms)
        else:
            pulled_arms = np.argmax(upper_conf, axis=1)

        self.pulled_rounds[np.arange(self.n_products), pulled_arms] += 1
        return pulled_arms

    def draw_alpha_ratios(self):
        alpha_ratios = np.random.beta(self.alpha_ratios_parameters[:, 0], self.alpha_ratios_parameters[:, 1])
        norm_factors = np.sum(alpha_ratios)
        alpha_ratios = alpha_ratios / norm_factors
        return alpha_ratios

    def draw_starting_page(self, alpha_ratios):
        product = np.random.choice(self.n_products + 1, p=alpha_ratios)
        return product

    def optimize(self, candidate_arms):

        best_reward = 0
        best_configuration = np.zeros(self.n_products)

        for configuration in itertools.product(*candidate_arms):
            daily_users = 100
            reward = 0
            alpha_ratios = self.draw_alpha_ratios()

            for _ in range(daily_users):
                product = self.draw_starting_page(alpha_ratios)
                if product == 5:  # competitors' page
                    continue
                visited = []
                to_visit = [product]

                while to_visit:
                    current_product = to_visit.pop(0)
                    visited.append(current_product)

                    product_price = self.prices[configuration[current_product]]

                    buy = np.random.binomial(1, self.estimated_conversion_rates[
                        current_product, configuration[current_product]])
                    if not buy:
                        continue

                    products_sold = np.random.randint(0, self.max_products_sold[current_product])
                    reward += product_price * products_sold

                    secondary_1 = self.secondaries[current_product, 0]
                    success_1 = np.random.binomial(1, self.graph_probabilities[current_product, secondary_1])
                    if success_1 and secondary_1 not in visited and secondary_1 not in to_visit:
                        to_visit.append(secondary_1)

                    secondary_2 = self.secondaries[current_product, 1]
                    success_2 = np.random.binomial(
                        1, self.lambda_p * self.graph_probabilities[current_product, secondary_2])
                    if success_2 and secondary_2 not in visited and secondary_2 not in to_visit:
                        to_visit.append(secondary_2)

            if reward > best_reward:
                best_reward = reward
                best_configuration = configuration

        return best_configuration

    def update(self, rewards, conversion_rates, pulled_arms):

        idxs = np.arange(self.n_products)
        n_pulls = self.pulled_rounds[idxs, pulled_arms]

        old_em = self.empirical_means[idxs, pulled_arms]
        rewards[rewards == -1] = old_em[rewards == -1]
        self.empirical_means[idxs, pulled_arms] = (old_em * (n_pulls - 1) + rewards) / n_pulls

        self.confidence[idxs, pulled_arms] = (2 * np.log(self.t) / n_pulls) ** 0.5

        old_ecr = self.estimated_conversion_rates[idxs, pulled_arms]
        conversion_rates[conversion_rates == -1] = old_ecr[conversion_rates == -1]
        self.estimated_conversion_rates[idxs, pulled_arms] = (old_ecr * (n_pulls - 1) + conversion_rates) / n_pulls

        self.t += 1

        if (rewards == -1).any() or (conversion_rates == -1).any():
            print(rewards)
            print(self.empirical_means)
            print(conversion_rates)
            print(self.estimated_conversion_rates)

        self.update_observations(rewards)
