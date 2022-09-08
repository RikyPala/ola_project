
import numpy as np


class UCB():

    def __init__(self, n_products, n_arms):

        self.n_products = n_products
        self.n_arms = n_arms
        self.counters = np.zeros((self.n_products, self.n_arms), dtype=int)
        self.pulled_rounds = np.zeros((n_products, n_arms))
        self.empirical_means = np.zeros((n_products, n_arms))
        self.confidence = np.ones((n_products, n_arms))*np.inf
        self.node_prob = np.zeros(3)
        self.previous_pulled = np.zeros(3)
        self.t = 1
        self.expected_alpha_ratios = np.array([0.2, 0.3, 0.1, 0.4])
        self.marginal_reward = np.zeros((3,2))
        self.prices = np.array([10, 20])
        self.product_sold = np.array([
            [40, 30],
            [40, 30],
            [40, 30]
        ])


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
                dot[self.previous_pulled[i]] += self.marginal_reward[i][self.previous_pulled[i]]
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



    def update(self, pulled_arms, conversion_rates, graph_prob, secondaries, lamb):

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
        self.node_probabilities(graph_prob, secondaries, lamb, pulled_arms)

        difference = self.node_prob - self.expected_alpha_ratios[:3]
        print("DIFFERENCEEEE")
        print(difference[0])

        for i in range(self.n_products):
            contr_1 = 0
            contr_2 = 0
            for j in range(self.n_products):
                if secondaries[i][0] == j and difference[0][j] > 0:
                    first = graph_prob[j, i]
                    contr_1 = self.compute_contribution(graph_prob, secondaries, i, j, lamb, first, difference[0], self.previous_pulled)
                elif secondaries[i][1] == j and difference[0][j] > 0:
                    second = graph_prob[j, i]*lamb
                    contr_2 = self.compute_contribution(graph_prob, secondaries, i, j, lamb, second, difference[0], self.previous_pulled)

            print("CONTRIBUTION1")
            print(contr_1)
            print("CONTRIBUTION2")
            print(contr_2)
            self.marginal_reward[i][pulled_arms[i]] = contr_1+contr_2

        self.t += 1


    def compute_contribution(self, graph_prob, secondaries, i, j, lamb, contribution, difference, pulled_arms):
        total_contribution = []
        for k in range(self.n_products):
            if secondaries[k][0] == j:
                total_contribution.append(graph_prob[k][j])
            elif secondaries[k][1] == j:
                total_contribution.append(graph_prob[k][j]*lamb)

            print("TOTAL CONTRIBUTIONNN")
            print(total_contribution)
            print(np.sum(total_contribution))
        margin_reward = difference[j]*self.empirical_means[j][pulled_arms[j]]*self.prices[pulled_arms[j]]* \
                        self.product_sold[j][pulled_arms[j]]
        print(margin_reward)

        return (contribution / np.sum(total_contribution))*margin_reward







    def node_probabilities(self,graph_prob, secondaries, lamb, pulled_arms):

        tot_node_arrivals = np.zeros(3)
        users_per_day = 100
        for user in range(users_per_day):

            already_visited = np.zeros(3)
            seed = self.simulate_starting_page()
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


    def simulate_starting_page(self):
        product = np.random.choice(4, p=self.expected_alpha_ratios)
        return product


