
import numpy as np


class UCB():

    def __init__(self, n_products, n_arms):

        self.n_products = n_products
        self.n_arms = n_arms
        self.rewards_per_arm = [[[] for _ in range(self.n_arms)] for _ in range(self.n_products)]
        self.collected_rewards = [[[]for _ in range(self.n_arms)] for _ in range(self.n_products)]
        self.counters = np.zeros((self.n_products, self.n_arms), dtype=int)
        self.collected_realization_per_arm = [[[]for _ in range(self.n_arms)] for _ in range(self.n_products)]
        self.pulled_rounds = np.zeros((n_products, n_arms)) # how many round a specific arm has been pulled, one for the initialization
        self.empirical_means = np.zeros((n_products, n_arms))
        self.confidence = np.ones((n_products, n_arms))*np.inf
        self.t = 1

    def pull_arm(self, prices, products_sold):


        upper_conf = self.empirical_means + self.confidence
        superarm = []
        for i in range(self.n_products):
            dot = upper_conf[i]*prices
            dot = dot * products_sold[i]
            j = np.random.choice(np.where(dot == dot.max())[0])
            self.pulled_rounds[i][j] += 1
            superarm.append(j)
        print(self.pulled_rounds)
        return superarm

    def update(self, pulled_arms, conversion_rates, alpha_ratios, graph_prob, secondaries, pulled, lamb):

        for i in range(self.n_products):
            self.empirical_means[i][pulled_arms[i]] = (self.empirical_means[i][pulled_arms[i]] *
                                                       (self.pulled_rounds[i][pulled_arms[i]]-1) + conversion_rates[i]) / self.pulled_rounds[i][pulled_arms[i]]
        print("EMPIRICAL MEAN")
        print(self.empirical_means)
        for i in range(self.n_products):
            for a in range(self.n_arms):
                n_samples = self.pulled_rounds[i][a]
                self.confidence[i][a] = (2*np.log(self.t+1)/n_samples)**0.5 if n_samples > 0 else np.inf
        print("CONFIDENCE")
        print(self.confidence)
        node_prob = self.node_probabilities(alpha_ratios, graph_prob, secondaries, lamb, pulled_arms)
        self.t += 1
        return node_prob



    def node_probabilities(self, alpha_ratios, graph_prob, secondaries, lamb, pulled_arms):

        tot_node_arrivals = np.empty(5)
        users_per_day = 10
        for user in range(users_per_day):
            seed = self.simulate_starting_page(alpha_ratios)
            if seed == 5:
                continue
            live_edge_graph = np.zeros((5, 5))

            start = seed
            already_visited = np.zeros(5)
            frontier = []
            frontier.append(start)
            print("Seed")
            print(seed)
            while len(frontier) > 0:
                start = frontier.pop(0)
                if already_visited[start] > 0:
                    continue

                already_visited[start] == 1

                for a in secondaries[start]:
                    print("aaa")
                    print(a)
                    if a == secondaries[start][0]:
                        print('conversion rate of product')
                        print(a)
                        print('with arm')
                        print(pulled_arms[a])
                        product = graph_prob[start][a]*self.empirical_means[a][pulled_arms[a]]
                        live_edge_graph[start][a] = np.random.binomial(1, product)
                    else:
                        live_edge_graph[start][a] = np.random.binomial(1, graph_prob[start][a] *
                                                                       self.empirical_means[a][pulled_arms[a]]*lamb)
                    candidates = np.where(live_edge_graph[start] == 1)
                    for i in candidates:
                        if already_visited[i] != 1:
                            frontier.append(i)


            print("Live edge graph")
            print(live_edge_graph)

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

            tot_node_arrivals += already_visited

        node_arrival_probabilities = np.array([tot_node_arrivals[i] / users_per_day])
        print("NODE PROB")
        print(node_arrival_probabilities)

        return node_arrival_probabilities


    def simulate_starting_page(self, alpha_ratios):
        product = np.random.choice(6, p=alpha_ratios)
        return product


