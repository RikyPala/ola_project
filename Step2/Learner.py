import numpy as np
from Environment import Environment


class Learner:

    def __init__(self, env: Environment()):

        self.env = env
        self.node_probabilities = []

    def greedy_optimization(self):

        iteration = 0
        current_configuration = [0] * self.env.n_products

        best_configuration = current_configuration
        best_reward = self.evaluate_configuration(current_configuration)

        while any(x < (self.env.n_arms - 1) for x in current_configuration):
            print(f"Best configuration n{iteration}: ", best_configuration)
            print(f"Best reward n{iteration}: ", best_reward, "\n")
            for i in range(self.env.n_products):
                new_configuration = current_configuration.copy()
                if new_configuration[i] >= (self.env.n_arms - 1):
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

        daily_users = np.random.randint(10, 200)
        reward = 0
        alpha_ratios = self.env.draw_alpha_ratios()

        for _ in range(daily_users):

            user_type = self.env.draw_user_type()
            product = self.env.draw_starting_page(user_type=user_type, alpha_ratios=alpha_ratios)
            if product == 5:  # competitors' page
                continue
            visited = []
            to_visit = [product]

            while to_visit:
                current_product = to_visit.pop(0)
                visited.append(current_product)

                product_price = self.env.prices[configuration[current_product]]

                buy = np.random.binomial(1, self.env.conversion_rates[
                    configuration[current_product], current_product, user_type])
                if not buy:
                    continue

                products_sold = np.random.randint(0, self.env.max_products_sold[current_product, user_type])
                reward += product_price * products_sold

                secondary_1 = self.env.secondaries[current_product, 0]
                success_1 = np.random.binomial(1, self.env.graph_probabilities[user_type, current_product, secondary_1])
                if success_1 and secondary_1 not in visited and secondary_1 not in to_visit:
                    to_visit.append(secondary_1)

                secondary_2 = self.env.secondaries[current_product, 1]
                success_2 = np.random.binomial(
                    1, self.env.lambda_p * self.env.graph_probabilities[user_type, current_product, secondary_2])
                if success_2 and secondary_2 not in visited and secondary_2 not in to_visit:
                    to_visit.append(secondary_2)

        return reward / daily_users

    """
    def evaluate_configuration2(self, configuration):

        reward = 0

        for prod in range(self.env.n_products):
            arm = configuration[prod]
            price = self.env.prices[configuration[prod]]
            for user in range(self.env.n_user_types):
                visits = self.env.user_probabilities[user] * self.node_probabilities[user][prod]
                conversions = visits * self.env.conversion_rates[arm][prod][user]
                sales = conversions * self.env.max_products_sold[prod][user] / 2
                reward += sales * price

        return reward

    def estimate_node_activation_fully_connected(self, graph_probabilities, feature_probabilities):

        tot_node_arrivals = np.empty((3, 5))
        tot_users = np.zeros(3)
        days = 100
        users_per_day = 10

        for d in range(days):

            alpha_ratios = self.env.draw_alpha_ratios()

            for user in range(users_per_day):
                user_type = self.env.draw_user_type()
                tot_users[user_type] += 1
                seed = self.env.draw_starting_page(user_type, alpha_ratios)

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
