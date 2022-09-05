
import numpy as np


class UCB():

    def __init__(self, n_products, n_arms):

        self.n_products = n_products
        self.n_arms = n_arms
        self.counters = np.zeros((self.n_products, self.n_arms), dtype=int)
        self.pulled_rounds = np.zeros((n_products, n_arms)) # how many round a specific arm has been pulled, one for the initialization
        self.empirical_means = np.zeros((n_products, n_arms))
        self.confidence = np.ones((n_products, n_arms))*np.inf
        self.node_prob = np.zeros(3)
        self.previous_pulled = np.zeros(3)
        self.t = 1

    def pull_arm(self, prices, products_sold):

        upper_conf = self.empirical_means + self.confidence
        superarm = []
        for i in range(self.n_products):
            dot = upper_conf[i]*prices
            print("DOTTTTTTT")
            print(dot)
            dot = dot * products_sold[i]
            print("DOTTTTTTT")
            print(dot)
            print(products_sold[i])
            if self.t > 1:
                print("PREVIOUS PULLED ARM")
                print(self.previous_pulled[i])
                print("NODE PRBABILITIES")
                print(self.node_prob[0][i])
                dot *= self.node_prob[0][i]
                print("FINALL DOTTT")
                print(dot)
            j = np.random.choice(np.where(dot == dot.max())[0])
            self.pulled_rounds[i][j] += 1
            superarm.append(j)
        print("PULLED ROUNDS")
        print(self.pulled_rounds)
        self.previous_pulled = superarm
        print("SUPERARM")
        print(superarm)
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
        self.node_probabilities(alpha_ratios, graph_prob, secondaries, lamb, pulled_arms)
        self.t += 1




    def node_probabilities(self, alpha_ratios, graph_prob, secondaries, lamb, pulled_arms):

        tot_node_arrivals = np.zeros(3)
        users_per_day = 250
        for user in range(users_per_day):

            already_visited = np.zeros(3)
            seed = self.simulate_starting_page(alpha_ratios)
            if seed == 3:
                continue
            live_edge_graph = np.zeros((3, 3))

            start = seed
            frontier = []
            frontier.append(start)

            while len(frontier) > 0:
                start = frontier.pop(0)
                #print("Start")
                #print(start)
                if already_visited[start] > 0:
                    continue
                else:
                    already_visited[start] = 1
                for a in secondaries[start]:
                    #print("secondaries")
                    #print(a)
                    if a == secondaries[start][0]:
                        product = graph_prob[start][a]*self.empirical_means[a][pulled_arms[a]]
                        live_edge_graph[start][a] = np.random.binomial(1, product)
                        if live_edge_graph[start][a] > 0:
                            """
                            print("product between")
                            print(graph_prob[start][a])
                            print(self.empirical_means[a][pulled_arms[a]])
                            """

                    else:
                        live_edge_graph[start][a] = np.random.binomial(1, graph_prob[start][a] * self.empirical_means[a][pulled_arms[a]]*lamb)

                        if live_edge_graph[start][a] > 0:
                            """print("product between")
                            print(graph_prob[start][a])
                            print(self.empirical_means[a][pulled_arms[a]])
                            print(lamb)"""
                candidates = np.where(live_edge_graph[start] == 1)
                #print("CANDIDATES")
                #print(candidates)
                for i in candidates[0]:
                    #print(i)
                    if already_visited[i] != 1:
                        frontier.append(i)
                """
                print("FRONTIER")
                print(frontier)
                print("ALREADY VISITED")
                print(already_visited)
                """
            #print("Live edge graph")
            #print(live_edge_graph)

            tot_node_arrivals += already_visited
            #print("TOT NODE ARRIVALS")
            #print(tot_node_arrivals)

        self.node_prob = np.array([tot_node_arrivals / users_per_day])
        print("NODE PROB")
        print(self.node_prob)


    def simulate_starting_page(self, alpha_ratios):
        product = np.random.choice(4, p=alpha_ratios)
        return product


