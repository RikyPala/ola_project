from Experiment2.Learner import Learner


class UCB(Learner):

    def __init__(self, n_products, n_arms, n_user_types):
        super().__init__(n_products, n_arms, n_user_types)

        self.t = 0

    def pull_arms(self):
        pass

    def update(self, pulled_arms, rewards):
        pass
