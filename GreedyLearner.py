from Learner import *


class GreedyLearner(Learner):

    def __init__(self, n_products, n_arms):
        super().__init__(n_products, n_arms)
        self.expected_rewards = np.zeros((n_products, n_arms))

    def pull_arm(self):
        if self.t < self.n_arms:
            return [self.t for _ in range(self.n_products)]
        pulled_arms = np.empty(self.n_products, dtype=int)
        for i in range(self.n_products):
            maximum = np.amax(self.expected_rewards[i])
            idxs = np.argwhere(self.expected_rewards[i] == maximum).reshape(-1)
            pulled_arms[i] = np.random.choice(idxs)

        return pulled_arms

    def update(self, pulled_arms, rewards):
        self.t += 1
        self.update_observations(pulled_arms, rewards)
        for i in range(self.n_products):
            self.expected_rewards[i][pulled_arms[i]] =\
                (self.expected_rewards[i][pulled_arms[i]] * (self.t - 1) + rewards[i]) / self.t
