from CUMSUM import CUMSUM
from Learner import Learner
import numpy as np
from scipy.optimize import linear_sum_assignment
from NonStationaryEnvironment import Environment, RoundData


class CUMSUM_UCB(Learner):

    def __init__(self, env: Environment, M = 100, eps = 0.05, h=20):
        super().__init__(env)
        self.confidence = np.ones((self.n_products, self.n_arms)) * np.inf
        self.empirical_means = np.zeros((self.n_products, self.n_arms))
        self.change_detection = self.init_change_detection(M, eps, h)
        self.detections = [[[] for _ in range(self.n_arms)] for _ in range(self.n_products)]
        self.valid_rewards_per_arms = [[[] for _ in range(self.n_arms)] for _ in range(self.n_products)]
        self.t = 0

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
            if self.change_detection[prod][pulled_arm[prod]].update(data, pulled_arm[prod]): # If this if is okay it means that there was an abrupt change, so we detect it and we reset all the CUMSUM parameters
                self.detections[prod][pulled_arm[prod]].append(self.t)
                self.valid_rewards_per_arms[prod][pulled_arm[prod]] = [[[] for _ in range(self.n_arms)] for _ in range(self.n_products)]
                self.pulled_rounds = np.zeros((self.n_products, self.n_arms))
                self.change_detection[prod][pulled_arm[prod]].reset()

            mean_est = data.conversions[prod]/data.visits[prod]
            self.update_observations(pulled_arm, mean_est, prod)
            self.empirical_means[prod][pulled_arm[prod]] = np.mean(self.valid_rewards_per_arms[prod][pulled_arm[prod]])
            total_valid_samples = self.compute_total_valid_samples()
            for a in range(self.n_arms):
                n_samples = self.pulled_rounds[prod][a]
                self.confidence[prod][a] = (2*np.log(total_valid_samples)/n_samples)**0.5 if n_samples >0 else np.inf
        self.update_marginal_reward(pulled_arm)

    def compute_total_valid_samples(self):
        total_valid_samples = 0
        for prod in range(self.n_products):
            for arm in range(self.n_arms):
                total_valid_samples += len(self.valid_rewards_per_arms[prod][arm])
        return total_valid_samples

    def update_observations(self, pulled_arm, reward, prod):
        self.valid_rewards_per_arms[prod][pulled_arm[prod]].append(reward)

