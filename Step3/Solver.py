import itertools
import numpy as np
import Environment as Environment


class Solver:

    def __init__(self, env: Environment):
        self.n_products = env.n_products
        self.n_arms = env.n_arms
        self.prices = env.prices
        self.conversion_rates = np.sum(
            env.conversion_rates * np.expand_dims(env.user_probabilities, axis=(1, 2)),
            axis=0)
        self.avg_products_sold = np.sum(
            (env.max_products_sold + 1) / 2 * np.expand_dims(env.user_probabilities, axis=1),
            axis=0)
        self.lambda_p = env.lambda_p
        self.alpha_ratios_parameters = np.sum(env.alpha_ratios_parameters, axis=0)
        self.graph_probabilities = np.sum(
            env.graph_probabilities * np.expand_dims(env.user_probabilities, axis=(1, 2)),
            axis=0)
        self.secondaries = env.secondaries

    def avg_alpha_ratios(self):
        alpha_ratios_avg = self.alpha_ratios_parameters[:, 0] / \
                           (self.alpha_ratios_parameters[:,0] + self.alpha_ratios_parameters[:,1])
        norm_factors = np.sum(alpha_ratios_avg, axis=0)
        print( alpha_ratios_avg / norm_factors)
        return alpha_ratios_avg / norm_factors

    def find_parents_lv(self, parents_lv, configuration, reward, level,  expected_alpha_ratios):
        parents_next = []
        if level == 1:
            for k in range(len(configuration)):
                if self.secondaries[k][0] == parents_lv[0]:
                    ca = self.conversion_rates[k][configuration[k]] * \
                         self.graph_probabilities[k][parents_lv[0]] * parents_lv[level]
                    reward[parents_lv[level-1]] += ca * expected_alpha_ratios[k]
                    parents_next.append([k, parents_lv[0], ca])
                elif self.secondaries[k][1] == parents_lv[0]:
                    ca = self.conversion_rates[k][configuration[k]] * \
                         self.lambda_p * self.graph_probabilities[k][parents_lv[0]] * parents_lv[level]
                    reward[parents_lv[level-1]] += ca * expected_alpha_ratios[k]
                    parents_next.append([k, parents_lv[0], ca])
            return parents_next
        elif level == 2:
            for parent in parents_lv:
                for k in range(len(configuration)):
                    if self.secondaries[k][0] == parent[0] and k != parent[1]:

                        ca = self.conversion_rates[k][configuration[k]] * self.graph_probabilities[k][parent[0]] * \
                             parent[level]
                        reward[parent[level-1]] += expected_alpha_ratios[k] * ca
                        parents_next.append([k, parent[0], parent[level-1], ca])

                    elif self.secondaries[k][1] == parent[0] and k != parent[1]:

                        ca = self.conversion_rates[k][configuration[k]] * self.graph_probabilities[k][
                            parent[0]] * self.lambda_p * parent[2]
                        reward[parent[level-1]] += expected_alpha_ratios[k] * ca
                        parents_next.append([k, parent[0], parent[level-1], ca])
            return parents_next
        elif level == 3:
            for parent in parents_lv:
                for k in range(len(configuration)):

                    if self.secondaries[k][0] == parent[0] and k != parent[1] and k != parent[2]:

                        ca = self.conversion_rates[k][configuration[k]] * self.graph_probabilities[k][parent[0]] * \
                             parent[level]
                        reward[parent[level-1]] += expected_alpha_ratios[k] * ca
                        parents_next.append([k, parent[0], parent[1], parent[level-1], ca])

                    elif self.secondaries[k][1] == parent[0] and k != parent[1] and k != parent[2]:

                        ca = self.conversion_rates[k][configuration[k]] * self.graph_probabilities[k][parent[0]] * self.lambda_p * parent[level]
                        reward[parent[level-1]] += expected_alpha_ratios[k] * ca
                        parents_next.append([k, parent[0], parent[1], parent[level-1], ca])
            return parents_next
        elif level == 4:
            for parent in parents_lv:
                for k in range(len(configuration)):

                    if self.secondaries[k][0] == parent[0] and k != parent[1] and k != parent[2] and k != parent[3]:

                        ca = self.conversion_rates[k][configuration[k]] * self.graph_probabilities[k][parent[0]] * \
                             parent[4]
                        reward[parent[level-1]] += expected_alpha_ratios[k] * ca
                        parents_next.append([k, parent[0], parent[1], parent[2], parent[3], ca])

                    elif self.secondaries[k][1] == parent[0] and k != parent[1] and k != parent[2] and k != parent[3]:

                        ca = self.conversion_rates[k][configuration[k]] * self.graph_probabilities[k][
                            parent[0]] * self.lambda_p * parent[level]
                        reward[parent[level-1]] += expected_alpha_ratios[k] * ca
                        parents_next.append([k, parent[0], parent[1], parent[2], parent[3], ca])
            return parents_next



    def find_optimal_arm(self):
        # print(''.join(map(str,configuration)))ù
        expected_alpha_ratios = self.avg_alpha_ratios()
        dict = {}
        for configuration in itertools.product([0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]):

            configuration = np.asarray(configuration)
            reward = np.zeros(5)

            for i in range(len(configuration)):  # number of product
                common_term = self.conversion_rates[i][configuration[i]] * self.prices[i][configuration[i]] * \
                              self.avg_products_sold[i]

                reward[i] += expected_alpha_ratios[i] * common_term
                level = 1
                parents_lv = [i, common_term]
                while parents_lv:
                    parents_lv = self.find_parents_lv(parents_lv, configuration, reward, level, expected_alpha_ratios)
                    level += 1

            dict[''.join(map(str, configuration))] = np.sum(reward)

        return dict



























