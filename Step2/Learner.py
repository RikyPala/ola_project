import numpy as np


class Learner:

    def __init__(self, n_products, n_arms, n_user_types):

        self.first_iteration = True

        self.n_products = n_products
        self.n_arms = n_arms
        self.n_user_types = n_user_types

        self.rewards_per_arm = [[[] for _ in range(self.n_arms)] for _ in range(self.n_products)]
        self.collected_rewards = [[] for _ in range(self.n_products)]

        self.counters = np.zeros(self.n_products, dtype=int)

    # TODO: add function to estimate node-arrival probabilities

    # TODO: add function to estimate expected rewards

    def pull_arms(self):  # TODO: remove pull_arms function

        if self.first_iteration:
            self.first_iteration = False
            return [np.zeros(self.n_products)]

        mask = self.counters < self.n_arms - 1
        configurations = []
        for i in range(self.n_products):
            if mask[i]:
                arms = self.counters
                arms[i] += 1
                configurations.append(arms)

        return configurations

    """
    def update(self, pulled_arms, rewards):

        if np.shape(self.collected_rewards)[-1] == 0:
            improvements = True
        else:
            improvements = any(rewards > np.array([elem[-1] for elem in self.collected_rewards]))

        for i in range(self.n_products):
            self.rewards_per_arm[i][pulled_arms[i]].append(rewards[i])
            self.collected_rewards[i].append(rewards[i])

        stop = not improvements or all(self.counters == self.n_arms - 1)

        return stop
    """

    def estimate_node_activation_fully_connected(self, graph_probabilities, feature_probabilities):

        tot_node_arrivals = np.empty((3, 5))  # z_i
        counters = np.zeros(3)  # k_i
        n_simulation = 10

        for s in range(n_simulation):
            user_type_index_0 = np.random.binomial(1, feature_probabilities[0])
            user_type_index_1 = 0
            if user_type_index_0:
                user_type_index_1 = np.random.binomial(1, feature_probabilities[1])
                edge_weights = graph_probabilities[user_type_index_0 + user_type_index_1]
                counters[user_type_index_0 + user_type_index_1] += 1
            else:
                edge_weights = graph_probabilities[user_type_index_0]
                counters[user_type_index_0] += 1

            flat_edge_weights = np.ravel(edge_weights)
            live_edge_graph = np.array([np.random.binomial(1, p) for p in flat_edge_weights])
            live_edge_graph = np.reshape(live_edge_graph, (5, 5))

            # DFS
            frontier = []
            already_visited = np.zeros(5)
            seed = np.random.choice(5, 1).squeeze()

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