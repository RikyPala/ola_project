from Learner import *


class Learner1(Learner):

    def __init__(self, n_products, n_arms):
        super().__init__(n_products, n_arms)
        self.expected_rewards = np.zeros((n_products, n_arms))
        self.counters = np.zeros(self.n_products, dtype=int)

    def pull_arm(self):
        mask = self.counters < self.n_arms - 1
        choice = np.random.choice(np.arange(self.n_products)[mask])
        self.counters[choice] += 1
        pulled_arms = self.counters
        return pulled_arms

    def update(self, pulled_arms, rewards):
        self.t += 1
        if np.shape(self.collected_rewards)[-1] == 0:
            improvements = True
        else:
            improvements = any(rewards > np.array([elem[-1] for elem in self.collected_rewards]))
        self.update_observations(pulled_arms, rewards)
        for i in range(self.n_products):
            self.expected_rewards[i][pulled_arms[i]] =\
                (self.expected_rewards[i][pulled_arms[i]] * (self.t - 1) + rewards[i]) / self.t
        stop = all(self.counters == self.n_arms - 1) or not improvements
        return stop
