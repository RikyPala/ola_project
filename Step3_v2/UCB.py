import itertools
import numpy as np

from Environment import Environment
from Learner import Learner


class UCB(Learner):

    def __init__(self, env: Environment):
        super().__init__(env)
        self.empirical_means = np.zeros((self.n_products, self.n_arms))
        self.confidence = np.ones((self.n_products, self.n_arms)) * np.inf
        self.c = 200
        self.pulled_rounds = np.zeros((self.n_products, self.n_arms))
        self.t = 1

    def pull_arms(self):
        upper_conf = self.empirical_means + self.c * self.confidence

        if self.t >= self.n_arms + 1:
            candidate_arms_1 = np.argmax(upper_conf, axis=1)
            upper_conf[np.arange(self.n_products), candidate_arms_1] = -np.inf
            candidate_arms_2 = np.argmax(upper_conf, axis=1)
            candidate_arms = np.dstack((candidate_arms_1, candidate_arms_2))[0]
            pulled_arms = self.optimize2(candidate_arms)
        else:
            pulled_arms = np.argmax(upper_conf, axis=1)

        self.pulled_rounds[np.arange(self.n_products), pulled_arms] += 1

        self.last_configuration = pulled_arms

        return pulled_arms

    def update(self, rewards, conversion_rates, pulled_arms):

        idxs = np.arange(self.n_products)
        n_pulls = self.pulled_rounds[idxs, pulled_arms]

        old_em = self.empirical_means[idxs, pulled_arms]
        rewards[rewards == -1] = old_em[rewards == -1]
        self.empirical_means[idxs, pulled_arms] = (old_em * (n_pulls - 1) + rewards) / n_pulls

        self.confidence[idxs, pulled_arms] = (2 * np.log(self.t) / n_pulls) ** 0.5

        old_ecr = self.estimated_conversion_rates[idxs, pulled_arms]
        conversion_rates[conversion_rates == -1] = old_ecr[conversion_rates == -1]
        self.estimated_conversion_rates[idxs, pulled_arms] = (old_ecr * (n_pulls - 1) + conversion_rates) / n_pulls

        self.t += 1

        if (rewards == -1).any() or (conversion_rates == -1).any():
            print(rewards)
            print(self.empirical_means)
            print(conversion_rates)
            print(self.estimated_conversion_rates)

        self.update_observations(rewards)
