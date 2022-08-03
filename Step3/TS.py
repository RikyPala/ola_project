import numpy as np

from Step3.Learner import Learner


class TS(Learner):

    def __init__(self, n_products, n_arms, n_user_types):
        super().__init__(n_products, n_arms, n_user_types)
        self.beta_parameters = np.ones((n_products, n_arms, 2), dtype=int)

    def pull_arms(self):
        realizations = np.random.beta(self.beta_parameters[:, :, 0], self.beta_parameters[:, :, 1])
        arms = np.argmax(realizations, axis=1)
        return arms

    def update(self, pulled_arms, rewards):
        self.update_observations(pulled_arms, rewards)

        # TODO: When a reward is a success?!
        if np.shape(self.collected_rewards)[-1] == 0:
            successes = np.ones(self.n_products)
        else:
            successes = (rewards >= np.array([elem[-1] for elem in self.collected_rewards]))

        for product in range(self.n_products):
            self.beta_parameters[product, pulled_arms[product], 0] += successes[product]
            self.beta_parameters[product, pulled_arms[product], 1] += 1 - successes[product]
