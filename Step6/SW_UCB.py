import numpy as np
from Learner import Learner
from NonStationaryEnvironment import Environment, RoundData


class SW_UCB(Learner):

    def __init__(self, env: Environment, window_size=10):
        super().__init__(env)
        self.t = 0
        self.confidence = np.ones((self.n_products, self.n_arms)) * np.inf
        self.empirical_means = np.zeros((self.n_products, self.n_arms))
        self.c = 0.1
        self.window_size = window_size
        self.pulled_arms_timeline = np.array([])
        self.pulled_rounds = np.zeros((self.n_products, self.n_arms))
        self.rewards_per_arms = [[[] for _ in range(self.n_arms)] for _ in range(self.n_products)]

    def pull(self):
        exp_conversion_rates = self.sample()
        exp_rewards = exp_conversion_rates * self.prices * self.max_products_sold + self.marginal_rewards
        configuration = np.argmax(exp_rewards, axis=1)
        # pulled rounds non può essere aggiornato nello stesso modo dello step3
        # metterlo in update anzichè in pull per rendere il metodo pull generale
        return configuration

    def update(self, results: RoundData):
        self.t += 1
        configuration = results.configuration  # pulled arms last round
        conversion_rates = results.conversions / results.visits
        self.pulled_arms_timeline = np.append(self.pulled_arms_timeline, configuration)
        for prod in range(self.n_products):
            for arm in range(self.n_arms):
                # update_observation serve?
                self.rewards_per_arms[prod][configuration[prod]].append(conversion_rates[prod])  # prod x arms
        self.pulled_rounds = np.array([np.bincount(pulls_per_prod, minlength=self.n_arms)
                                       for pulls_per_prod in self.pulled_arms_timeline[-self.window_size:].T])  # prod x arms
        for prod in range(self.n_products):
            for arm in range(self.n_arms):
                n = self.pulled_rounds[prod][arm]
                self.empirical_means[prod, arm] = np.sum(self.rewards_per_arms[prod][arm][-n:]) / n if n > 0 else 0
                self.confidence[prod, arm] = self.c * (np.log(self.t) / n) ** 0.5 if n > 0 else np.inf

    def sample(self):
        return self.empirical_means + self.confidence

    def get_means(self):
        return self.empirical_means
