import numpy as np
from Environment import Environment


class Learner:

    def __init__(self, env: Environment()):

        self.env = env
        self.node_probabilities = []
        self.landing_probabilities = []

    # TODO: add function to estimate node-arrival probabilities

    # TODO: add function to estimate expected rewards

    def greedy_optimization(self):  # TODO: remove pull_arms function

        config = [0] * self.env.n_products

        self.node_probabilities = []
        # for i in range(self.env.n_user_types):
        self.node_probabilities = self.estimate_node_activation_fully_connected(self.env.graph_probabilities, self.env.feature_probabilities)
        self.landing_probabilities = self.estimate_landing_probabilities(self.env.alpha_ratios_parameters)

        best_configuration = config
        best_reward = self.evaluate_configuration(config)
        print(config)
        print(best_reward)

        while any(x < (self.env.n_arms - 1) for x in config):
            print()
            new_config = []
            for i in range(self.env.n_products):
                new_config = config.copy()
                if new_config[i] >= (self.env.n_arms - 1):
                    continue
                new_config[i] += 1
                print(new_config)
                reward = self.evaluate_configuration(new_config)
                print(reward)
                if reward > best_reward:
                    best_configuration = new_config
                    best_reward = reward
                    print("# BEST")
            if np.array_equal(config, best_configuration):
                break
            config = best_configuration

        print("\n########")
        print(best_configuration)
        print(best_reward)
        return best_configuration

    def evaluate_configuration(self, config):

        reward = 0

        for prod in range(self.env.n_products):
            arm = config[prod]
            price = self.env.prices[config[prod]]
            for user in range(self.env.n_user_types):
                visits = self.env.user_probabilities[user] * self.landing_probabilities[user][prod + 1] * \
                         self.node_probabilities[user][prod]
                conversions = visits * self.env.conversion_rates[arm][prod][user]
                sales = conversions * self.env.max_products_sold[prod][user]
                reward += sales * price

        return reward


    def estimate_node_activation_fully_connected(self, graph_probabilities, feature_probabilities):

        tot_node_arrivals = np.empty((3, 5))  # z_i
        counters = np.zeros(3)  # k_i
        n_simulation = 10
        n_day = 1
        for d in range(n_day):

            alpha_ratios = self.env.draw_alpha_ratios()

            for s in range(n_simulation):
                user_type_index_0 = np.random.binomial(1, feature_probabilities[0])
                user_type_index_1 = 0
                if user_type_index_0:
                    user_type_index_1 = np.random.binomial(1, feature_probabilities[1])
                    edge_weights = graph_probabilities[user_type_index_0 + user_type_index_1]
                    counters[user_type_index_0 + user_type_index_1] += 1
                    seed = self.env.draw_starting_page(user_type_index_0 + user_type_index_1, alpha_ratios)

                else:
                    edge_weights = graph_probabilities[user_type_index_0]
                    counters[user_type_index_0] += 1
                    seed = self.env.draw_starting_page(user_type_index_0, alpha_ratios)

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

                if user_type_index_0:
                    tot_node_arrivals[user_type_index_0 + user_type_index_1] += already_visited
                else:
                    tot_node_arrivals[user_type_index_0] += already_visited

        node_arrival_probabilities = np.array([tot_node_arrivals[i] / count for i, count in enumerate(counters)])
        return node_arrival_probabilities

    def estimate_landing_probabilities(self, params):

        probs = params[:, :, 0] / (params[:, :, 0] + params[:, :, 1])
        for a in probs:
            a /= a.sum()
        return probs
