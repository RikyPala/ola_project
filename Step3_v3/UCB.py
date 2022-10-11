import numpy as np
from Learner import Learner
from  Environment import Environment, RoundData


class UCB(Learner):

    def __init__(self, env: Environment):
        super().__init__(env)
        self.confidence = np.ones((self.n_products, self.n_arms)) * np.inf
        self.pulled_rounds = np.zeros((self.n_products, self.n_arms))

    def update(self, results: RoundData):
        self.t += 1
        configuration = results.configuration
        conversion_rate = results.conversions / results.visits

        for i in range(self.n_products):

            self.empirical_means[i][configuration[i]] = (self.empirical_means[i][configuration[i]] *
                                                       (self.pulled_rounds[i][configuration[i]]-1) + conversion_rate[i]) / self.pulled_rounds[i][configuration[i]]

        for i in range(self.n_products):
            for a in range(self.n_arms):
                n_samples = self.pulled_rounds[i][a]
                self.confidence[i][a] = 10*(2*np.log(self.t+1)/n_samples)**0.5 if n_samples > 0 else np.inf

        from_prod_to_node = self.compute_nearby_contribution(configuration)

        for prod in range(self.n_products):
            for temp in range(self.n_products):
                contribution = self.empirical_means[prod][configuration[prod]] * from_prod_to_node[prod][temp] \
                                                                     * self.empirical_means[temp][configuration[temp]] * self.prices[temp][configuration[temp]] \
                                                                     * self.max_products_sold[temp][configuration[temp]]
                self.marginal_rewards[prod][configuration[prod]] += (self.empirical_means[prod][configuration[prod]] * \
                                                                     (self.pulled_rounds[prod][configuration[prod]]-1) + contribution) / self.pulled_rounds[prod][configuration[prod]]
        print("MARGINAL REWARDS")
        print(self.marginal_rewards)

    def sample(self):
        print("Upper Confidence")
        print(self.confidence)
        return self.empirical_means + self.confidence



