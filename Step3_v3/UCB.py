import numpy as np
from Learner import Learner
from  Environment import Environment, RoundData


class UCB(Learner):

    def __init__(self, env: Environment):
        super().__init__(env)
        self.confidence = np.ones((self.n_products, self.n_arms)) * np.inf
        self.pulled_rounds = np.zeros((self.n_products, self.n_arms))
        self.empirical_means = np.zeros(env.n_products, env.n_products)

    def update(self, results: RoundData):

        configuration = results.configuration

        for i in range(self.n_products):
            self.empirical_means[i][configuration[i]] = (self.empirical_means[i][configuration[i]] *
                                                       (self.pulled_rounds[i][configuration[i]]-1) + np.nan_to_num(results.conversions[i]/results.visits[i])) / self.pulled_rounds[i][configuration[i]]
        for i in range(self.n_products):
            for a in range(self.n_arms):
                n_samples = self.pulled_rounds[i][a]
                self.confidence[i][a] = 500*(2*np.log(self.t+1)/n_samples)**0.5 if n_samples > 0 else np.inf

        from_prod_to_node = self.compute_nearby_contribution(configuration)

        for prod in range(self.n_products):
            for temp in range(self.n_products):
                self.marginal_rewards[prod][configuration[prod]] += self.empirical_means[prod][configuration[prod]] * from_prod_to_node[prod][temp] * \
                                                                    self.empirical_means[temp][configuration[temp]] * self.max_products_sold[temp][configuration[temp]] * \
                                                                    self.prices[temp][configuration[temp]]




    def sample(self):

        upper_conf = self.empirical_means + self.confidence
        products_sold = np.random.randint(self.max_products_sold + 1)
        for prod in range(self.n_products):
            for arm in range(self.n_arms):
                upper_conf[prod][arm] = upper_conf[prod][arm] * products_sold[prod][arm] * self.prices[prod][arm] + self.marginal_rewards
        return upper_conf



