import numpy as np
from Environment import Environment


class Learner:

    def __init__(self, env: Environment):

        # Parameters
        self.n_products = env.n_products
        self.n_arms = env.n_arms
        self.n_user_types = env.n_user_types

        self.feature_probabilities = env.feature_probabilities
        self.user_probabilities = env.user_probabilities
        self.prices = env.prices
        self.conversion_rates = env.conversion_rates
        self.max_products_sold = env.max_products_sold
        self.lambda_p = env.lambda_p
        self.alpha_ratios_parameters = env.alpha_ratios_parameters
        self.graph_probabilities = env.graph_probabilities
        self.secondaries = env.secondaries

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
        product = np.random.choice(self.n_products + 1, p=alpha_ratios[user_type])
        return product

    def draw_alpha_ratios(self):
        alpha_ratios = np.random.beta(self.alpha_ratios_parameters[:, :, 0], self.alpha_ratios_parameters[:, :, 1])
        norm_factors = np.sum(alpha_ratios, axis=1)
        alpha_ratios = (alpha_ratios.T / norm_factors).T
        return alpha_ratios

    def greedy_optimization(self):

        iteration = 0
        current_configuration = [0] * self.n_products

        best_configuration = current_configuration
        best_reward = self.evaluate_configuration(current_configuration)

        while any(x < (self.n_arms - 1) for x in current_configuration):
            print(f"Best configuration n{iteration}: ", best_configuration)
            print(f"Best reward n{iteration}: ", best_reward, "\n")
            for i in range(self.n_products):
                new_configuration = current_configuration.copy()
                if new_configuration[i] >= (self.n_arms - 1):
                    continue
                new_configuration[i] += 1
                print("Testing configuration ", new_configuration, "...")
                reward = self.evaluate_configuration(new_configuration)
                print("Obtained reward:  ", reward)
                if reward > best_reward:
                    best_configuration = new_configuration
                    best_reward = reward
                    print("# BEST")
            if np.array_equal(current_configuration, best_configuration):
                break
            current_configuration = best_configuration
            iteration += 1
            print()

        print("#############\n")
        print("Best configuration: ", best_configuration)
        print("Best reward: ", best_reward)

        return best_configuration

    def evaluate_configuration(self, configuration):

        daily_users = 1000
        reward = 0
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

                product_price = self.prices[configuration[current_product]]

                buy = np.random.binomial(1, self.conversion_rates[
                    user_type, current_product, configuration[current_product]])
                if not buy:
                    continue

                products_sold = np.random.randint(0, self.max_products_sold[current_product, user_type])
                reward += product_price * products_sold

                secondary_1 = self.secondaries[current_product, 0]
                success_1 = np.random.binomial(1, self.graph_probabilities[user_type, current_product, secondary_1])
                if success_1 and secondary_1 not in visited and secondary_1 not in to_visit:
                    to_visit.append(secondary_1)

                secondary_2 = self.secondaries[current_product, 1]
                success_2 = np.random.binomial(
                    1, self.lambda_p * self.graph_probabilities[user_type, current_product, secondary_2])
                if success_2 and secondary_2 not in visited and secondary_2 not in to_visit:
                    to_visit.append(secondary_2)

        return reward

    """
    def evaluate_configuration2(self, configuration):

        reward = 0

        for prod in range(self.n_products):
            arm = configuration[prod]
            price = self.prices[configuration[prod]]
            for user in range(self.n_user_types):
                visits = self.user_probabilities[user] * self.node_probabilities[user][prod]
                conversions = visits * self.conversion_rates[arm][prod][user]
                sales = conversions * self.max_products_sold[prod][user] / 2
                reward += sales * price

        return reward

    def estimate_node_activation_fully_connected(self, graph_probabilities, feature_probabilities):

        tot_node_arrivals = np.empty((3, 5))
        tot_users = np.zeros(3)
        days = 100
        users_per_day = 10

        for d in range(days):

            alpha_ratios = self.draw_alpha_ratios()

            for user in range(users_per_day):
                user_type = self.draw_user_type()
                tot_users[user_type] += 1
                seed = self.draw_starting_page(user_type, alpha_ratios)

                if seed == 5:
                    continue

                flat_edge_weights = np.ravel(edge_weights)
                live_edge_graph = np.array([np.random.binomial(1, p) for p in flat_edge_weights])
                live_edge_graph = np.reshape(live_edge_graph, (5, 5))
                # DFS
                frontier = []
                already_visited = np.zeros(5)

                already_visited[seed] = 1
                frontier.extend([idx for idx, p in enumerate(live_edge_graph[seed]) if p > 0])  # argwhere problem

                while len(frontier) > 0:
                    seed = frontier.pop(0)
                    if already_visited[seed] > 0:
                        continue
                    already_visited[seed] = 1
                    frontier.extend([idx for idx, p in enumerate(live_edge_graph[seed]) if p > 0])

                tot_node_arrivals[user_type] += already_visited

        node_arrival_probabilities = np.array([tot_node_arrivals[i] / count for i, count in enumerate(counters)])
        return node_arrival_probabilities
    """
