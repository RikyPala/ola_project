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

    def avg_alpha_ratios(self, alpha_ratios_parameters):
        alpha_ratios_avg = alpha_ratios_parameters[:, :, 0] / \
                           (alpha_ratios_parameters[:, :, 0] + alpha_ratios_parameters[:, :, 1])
        norm_factors = np.sum(alpha_ratios_avg, axis=1)
        return (alpha_ratios_avg.T / norm_factors).T

    def find_parents_lv1(self, i, configuration, reward, common_term):
        parents_lv1 = []
        expected_alpha_ratios = self.avg_alpha_ratios(self.alpha_ratios_parameters)
        for k in range(len(configuration)):
            if self.secondaries[k][0] == i:
                ca = self.conversion_rates[k][configuration[k]] * \
                     self.graph_probabilities[k][i] * common_term
                reward[i] += ca * expected_alpha_ratios[k]
                parents_lv1.append([k, i, ca])
            elif self.secondaries[k][1] == i:
                ca = self.conversion_rates[k][configuration[k]] * \
                     self.lambda_p * self.graph_probabilities[k][i] * common_term
                reward[i] += ca * expected_alpha_ratios[k]
                parents_lv1.append([k, i, ca])
        return parents_lv1

    def find_parents_lv2(self, parents_lv1, i, configuration, reward, reward_tot):

        parents_lv2 = []
        expected_alpha_ratios = self.avg_alpha_ratios(self.alpha_ratios_parameters)
        for parent in parents_lv1:
            for k in range(len(configuration)):

                if self.secondaries[k][0] == parent[0] and k != parent[1]:

                    ca = self.conversion_rates[k][configuration[k]] * self.graph_probabilities[k][parent[0]] * parent[2]
                    reward_tot[i] += ca
                    reward[i] += expected_alpha_ratios[k] * ca
                    parents_lv2.append([k, parent[0], i, ca])

                elif self.secondaries[k][1] == parent[0] and k != parent[1]:

                    ca = self.conversion_rates[k][configuration[k]] * self.graph_probabilities[k][
                        parent[0]] * self.lambda_p * parent[2]
                    reward[i] += expected_alpha_ratios[k] * ca
                    parents_lv2.append([k, parent[0], i, ca])
                    reward_tot[i] += ca

        return parents_lv2

    def find_parents_lv3(self, parents_lv2, i, configuration, reward, reward_tot):
        parents_lv3 = []
        expected_alpha_ratios = self.avg_alpha_ratios(self.alpha_ratios_parameters)
        for parent in parents_lv2:
            for k in range(len(configuration)):

                if self.secondaries[k][0] == parent[0] and k != parent[1] and k != parent[2]:

                    ca = self.conversion_rates[k][configuration[k]] * self.graph_probabilities[k][parent[0]] * parent[3]
                    reward_tot[i] += ca
                    reward[i] += expected_alpha_ratios[k] * ca
                    parents_lv3.append([k, parent[0], parent[1], i, ca])

                elif self.secondaries[k][1] == parent[0] and k != i and k != parent[1] and k != parent[2]:

                    ca = self.conversion_rates[k][configuration[k]] * self.graph_probabilities[k][
                        parent[0]] * self.lambda_p * parent[3]
                    reward[i] += expected_alpha_ratios[k] * ca
                    reward_tot[i] += ca
                    parents_lv3.append([k, parent[0], parent[1], i, ca])
        return parents_lv3

    def find_parents_lv4(self, parents_lv3, i, configuration, reward, reward_tot):
        parents_lv4 = []
        expected_alpha_ratios = self.avg_alpha_ratios(self.alpha_ratios_parameters)
        for parent in parents_lv3:
            for k in range(len(configuration)):

                if self.secondaries[k][0] == parent[0] and k != parent[1] and k != parent[2] and k != parent[3]:

                    ca = self.conversion_rates[k][configuration[k]] * self.graph_probabilities[k][parent[0]] * parent[4]
                    reward_tot[i] += ca
                    reward[i] += expected_alpha_ratios[k] * ca
                    parents_lv4.append([k, parent[0], parent[1], parent[2], parent[3], ca])

                elif self.secondaries[k][1] == parent[0] and k != parent[1] and k != parent[2] and k != parent[3]:

                    ca = self.conversion_rates[k][configuration[k]] * self.graph_probabilities[k][
                        parent[0]] * self.lambda_p * parent[4]
                    reward[i] += expected_alpha_ratios[k] * ca
                    reward_tot[i] += ca
                    parents_lv4.append([k, parent[0], parent[1], parent[2], parent[3], ca])
        return parents_lv4

    def compute_parents_lv5(self, parents_lv4, i, configuration, reward, reward_tot):
        expected_alpha_ratios = self.avg_alpha_ratios(self.alpha_ratios_parameters)
        for parent in parents_lv4:
            for k in range(len(configuration)):

                if self.secondaries[k][0] == parent[0] and k != parent[1] and k != parent[2] and k != parent[3] and k != \
                        parent[4]:

                    ca = self.conversion_rates[k][configuration[k]] * self.graph_probabilities[k][parent[0]] * parent[5]
                    reward_tot[i] += ca
                    reward[i] += expected_alpha_ratios[k] * ca


                elif self.secondaries[k][1] == parent[0] and k != parent[1] and k != parent[2] and k != parent[
                    3] and k != parent[4]:

                    ca = self.conversion_rates[k][configuration[k]] * self.graph_probabilities[k][
                        parent[0]] * self.lambda_p * parent[5]
                    reward[i] += expected_alpha_ratios[k] * ca
                    reward_tot[i] += ca

    def find_optimal_arm_base(self):
        # print(''.join(map(str,configuration)))
        dict = {}
        for configuration in itertools.product([0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]):
            configuration = np.asarray(configuration)
            reward = np.zeros(5)
            for i in range(len(configuration)):  # number of product
                common_term = self.conversion_rates[i][configuration[i]] * self.prices[configuration[i]] * \
                              self.avg_products_sold[i]
                reward[i] += self.expected_alpha_ratios[i] * common_term
            dict[''.join(map(str, configuration))] = np.sum(reward)

        return dict

    def find_optimal_arm_until_lv1(self):
        # print(''.join(map(str,configuration)))
        dict = {}
        for configuration in itertools.product([0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]):
            configuration = np.asarray(configuration)
            reward = np.zeros(5)
            for i in range(len(configuration)):  # number of product
                common_term = self.conversion_rates[i][configuration[i]] * self.prices[configuration[i]] * \
                              self.avg_products_sold[i]
                self.find_parents_lv1(i, configuration, reward, common_term)

                reward[i] += self.expected_alpha_ratios[i] * common_term
            dict[''.join(map(str, configuration))] = np.sum(reward)

        return dict

    def find_optimal_arm_until_lv2(self):
        # print(''.join(map(str,configuration)))
        dict = {}
        for configuration in itertools.product([0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]):
            configuration = np.asarray(configuration)
            reward = np.zeros(5)
            reward_tot = np.zeros(5)
            for i in range(len(configuration)):  # number of product
                common_term = self.conversion_rates[i][configuration[i]] * self.prices[configuration[i]] * \
                              self.avg_products_sold[i]
                parents_lv1 = self.find_parents_lv1(i, configuration, reward, common_term)
                self.find_parents_lv2(parents_lv1, i, configuration, reward, reward_tot)

                reward[i] += self.expected_alpha_ratios[i] * common_term
            dict[''.join(map(str, configuration))] = np.sum(reward)

        return dict

    def find_optimal_arm_until_lv3(self):
        # print(''.join(map(str,configuration)))
        dict = {}
        for configuration in itertools.product([0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]):

            configuration = np.asarray(configuration)
            print("CONFIGURATION")
            print(configuration)

            reward = np.zeros(5)
            reward_tot = np.zeros(5)

            for i in range(len(configuration)):  # number of product
                common_term = self.conversion_rates[i][configuration[i]] * self.prices[configuration[i]] * \
                              self.avg_products_sold[i]
                reward[i] += self.expected_alpha_ratios[i] * common_term
                parents_lv1 = self.find_parents_lv1(i, configuration, reward, common_term)
                parents_lv2 = self.find_parents_lv2(parents_lv1, i, configuration, reward, reward_tot)
                self.find_parents_lv3(parents_lv2, i, configuration, reward, reward_tot)

            dict[''.join(map(str, configuration))] = np.sum(reward)

        return dict

    def find_optimal_arm_until_lv4(self):
        # print(''.join(map(str,configuration)))
        dict = {}
        for configuration in itertools.product([0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]):

            configuration = np.asarray(configuration)
            print("CONFIGURATION")
            print(configuration)

            reward = np.zeros(5)
            reward_tot = np.zeros(5)

            for i in range(len(configuration)):  # number of product
                common_term = self.conversion_rates[i][configuration[i]] * self.prices[configuration[i]] * \
                              self.avg_products_sold[i]
                reward[i] += self.expected_alpha_ratios[i] * common_term
                parents_lv1 = self.find_parents_lv1(i, configuration, reward, common_term)
                parents_lv2 = self.find_parents_lv2(parents_lv1, i, configuration, reward, reward_tot)
                parents_lv3 = self.find_parents_lv3(parents_lv2, i, configuration, reward, reward_tot)
                self.find_parents_lv4(parents_lv3, i, configuration, reward, reward_tot)

            dict[''.join(map(str, configuration))] = np.sum(reward)

        return dict

    def find_optimal_arm_until_lv5(self):
        # print(''.join(map(str,configuration)))
        dict = {}
        for configuration in itertools.product([0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]):

            configuration = np.asarray(configuration)
            print("CONFIGURATION")
            print(configuration)

            reward = np.zeros(5)
            reward_tot = np.zeros(5)

            for i in range(len(configuration)):  # number of product
                common_term = self.conversion_rates[i][configuration[i]] * self.prices[configuration[i]] * \
                              self.avg_products_sold[i]
                reward[i] += self.expected_alpha_ratios[i] * common_term
                parents_lv1 = self.find_parents_lv1(i, configuration, reward, common_term)
                parents_lv2 = self.find_parents_lv2(parents_lv1, i, configuration, reward, reward_tot)
                parents_lv3 = self.find_parents_lv3(parents_lv2, i, configuration, reward, reward_tot)
                parents_lv4 = self.find_parents_lv4(parents_lv3, i, configuration, reward, reward_tot)
                self.compute_parents_lv5(parents_lv4, i, configuration, reward, reward_tot)

            dict[''.join(map(str, configuration))] = np.sum(reward)

        return dict



























