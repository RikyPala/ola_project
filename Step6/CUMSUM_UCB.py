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
        self.change_detection, self.detections, self.valid_rewards_per_arms = self.init_change_detection(M, eps, h)
        self.t = 0


    def init_change_detection(self, M, eps, h):
        change_detection = [[] for _ in range(self.n_products)]
        valid_rewards_per_arm = [[] for _ in range(self.n_products)]
        detections = [[] for _ in range(self.n_products)]
        for prod in range(self.n_products):
            for arm in range(self.n_arms):
                li = []
                cum = CUMSUM(M, eps, h)
                change_detection[prod].append(cum)
                detections[prod].append(li)
                valid_rewards_per_arm[prod].append(li)
        return change_detection, detections, valid_rewards_per_arm

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
                self.valid_rewards_per_arms[prod][pulled_arm[prod]] = []
                self.change_detection[prod][pulled_arm[prod]].reset()

            mean_est = data.conversions[prod]/data.visits[prod]
            self.update_observations(pulled_arm, mean_est, prod)
            self.empirical_means[prod][pulled_arm[prod]] = np.mean(self.valid_rewards_per_arms[prod][pulled_arm[prod]])
            total_valid_samples = 0
            for a in range(self.n_arms):
                total_valid_samples += sum(self.valid_rewards_per_arms[prod][a])
            for a in range(self.n_arms):
                n_samples = 0
                for arms in range(self.n_arms):
                    n_samples += len(self.valid_rewards_per_arms[prod][arms])
                self.confidence[prod][a] = (2*np.log(total_valid_samples)/n_samples)**0.5 if n_samples >0 else np.inf
        self.update_marginal_reward(pulled_arm)

    def update_observations(self, pulled_arm, reward, prod):

        self.valid_rewards_per_arms[prod][pulled_arm[prod]].append(reward)

