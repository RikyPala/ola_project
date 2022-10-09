import numpy as np
from Learner import Learner
from  Environment import Environment, RoundData


class UCB(Learner):

    def __init__(self, env: Environment):
        super().__init__(env)
        self.empirical_means = np.zeros((self.n_products, self.n_arms))
        self.confidence = np.ones((self.n_products, self.n_arms)) * np.inf
        self.pulled_rounds = np.zeros((self.n_products, self.n_arms))

    def update(self, results: RoundData):

        configuration = results.configuration
        for i in range(self.n_products):
            self.empirical_means[i][configuration[i]] = (self.empirical_means[i][configuration[i]] *
                                                       (self.pulled_rounds[i][configuration[i]]-1) + np.nan_to_num(results.conversions[i]/results.visits[i])) / self.pulled_rounds[i][configuration[i]]
        for i in range(self.n_products):
            for a in range(self.n_arms):
                n_samples = self.pulled_rounds[i][a]
                self.confidence[i][a] = 500*(2*np.log(self.t+1)/n_samples)**0.5 if n_samples > 0 else np.inf


    def sample(self, results: RoundData):

        superarm = []
        for i in range(self.n_products):
            dot = self.empirical_means[i] * self.prices[i]
            dot = dot * self.products_sold[i]
            if self.t > 1:
                print("PREVIOUS PULLED ARM")
                print(self.previous_pulled[i])
                print("NODE PRBABILITIES")
                print(self.node_prob[0][i])
                dot[self.previous_pulled[i]] += self.marginal_reward[i][self.previous_pulled[i]]
                print("FINALL DOTTT")
                print(dot)
                dot = dot + self.confidence[i]
            j = np.random.choice(np.where(dot == dot.max())[0])
            self.pulled_rounds[i][j] += 1
            superarm.append(j)
        print("PULLED ROUNDS")
        print(self.pulled_rounds)
        self.previous_pulled = superarm
        print("SUPERARM")
        print(superarm)
        return superarm



