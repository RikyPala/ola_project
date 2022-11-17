from CUMSUM import CUMSUM
from Learner import Learner
import numpy as np
from scipy.optimize import linear_sum_assignment
from NonStationaryEnvironment import Environment, RoundData


class CUMSUM_UCB(Learner):

    def __init__(self, env: Environment, M=100, eps=0.05, h=20, alpha=0.1):
        super().__init__(env)
        self.confidence = np.ones((self.n_products, self.n_arms)) * np.inf
        self.empirical_means = np.zeros((self.n_products, self.n_arms))
        self.change_detection = self.init_change_detection(M, eps, h)
        self.valid_rewards_per_arms = [[[] for _ in range(self.n_arms)] for _ in range(self.n_products)]
        self.t = 0
        self.alpha = alpha

    def init_change_detection(self, M, eps, h):
        change_detection = [[] for _ in range(self.n_products)]
        for prod in range(self.n_products):
            for arm in range(self.n_arms):
                cum = CUMSUM(M, eps, h)
                change_detection[prod].append(cum)
        return change_detection

    def sample(self):
        return self.empirical_means + self.confidence

    def get_means(self):
        return self.empirical_means

    def update(self, data: RoundData):
        self.t += 1
        pulled_arm = data.configuration

        for prod in range(self.n_products):
            if self.change_detection[prod][pulled_arm[prod]].update(data, pulled_arm[prod]):
                # If this is okay, it means that there was an abrupt change.
                # then we detect it, finally we reset all the CUMSUM parameters
                self.valid_rewards_per_arms[prod][pulled_arm[prod]] = [[[] for _ in range(self.n_arms)]
                                                                       for _ in range(self.n_products)]
                self.pulled_rounds[prod][pulled_arm[prod]] = 0
                self.change_detection[prod][pulled_arm[prod]].reset()
                self.marginal_rewards[prod][pulled_arm[prod]] = 0

            self.valid_rewards_per_arms[prod][pulled_arm[prod]].append(data.conversions[prod]/data.visits[prod])
            self.empirical_means[prod][pulled_arm[prod]] = np.mean(self.valid_rewards_per_arms[prod][pulled_arm[prod]])
            total_valid_samples = np.sum(self.pulled_rounds[prod])
            for a in range(self.n_arms):
                n_samples = self.pulled_rounds[prod][a]
                self.confidence[prod][a] = (2 * np.log(total_valid_samples)/n_samples) ** 0.5 if n_samples > 0 \
                    else np.inf
        self.update_marginal_reward(pulled_arm)

    def pull(self):
        if np.random.binomial(1, 1 - self.alpha):
            exp_conversion_rates = self.sample()
            exp_rewards = exp_conversion_rates * self.prices * self.max_products_sold + self.marginal_rewards
            configuration = np.argmax(exp_rewards, axis=1)
            self.pulled_rounds[np.arange(self.n_products), configuration] += 1
            return configuration
        else:
            configuration = np.random.randint(0, self.n_arms, self.n_products)
            return configuration




