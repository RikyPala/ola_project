import numpy as np

from Environment import Environment
from Learner import Learner
from RoundsHistory import RoundsHistory


class ContextNode:

    def __init__(self, env: Environment, learner_class, feature_1=None, feature_2=None, delta=0.1):
        self.delta = delta
        self.left = None
        self.right = None
        self.env = env
        self.learner: Learner = learner_class(env, feature_1, feature_2)
        self.feature_1 = feature_1
        self.feature_2 = feature_2
        self.feature_probabilities = env.feature_probabilities

    def get_splittable_features(self):
        splittable_features = []
        if self.feature_1 is None:
            splittable_features.append(1)
        if self.feature_2 is None:
            splittable_features.append(2)
        return splittable_features

    def split(self):
        splittable_features = self.get_splittable_features()
        if not splittable_features:
            return False
        parent_x_ = np.sum(np.mean(self.learner.get_means(), axis=1))
        t = RoundsHistory.get_number_rounds(type(self.learner))
        best_lower_bound = parent_x_ - (- np.log(self.delta) / (2 * t)) ** 0.5
        fp = self.feature_probabilities
        for feature in splittable_features:
            if feature == 1:
                left = ContextNode(self.env, type(self.learner), feature_1=False, feature_2=self.feature_2,
                                   delta=self.delta)
                right = ContextNode(self.env, type(self.learner), feature_1=True, feature_2=self.feature_2,
                                    delta=self.delta)
            elif feature == 2:
                left = ContextNode(self.env, type(self.learner), feature_1=self.feature_1, feature_2=False,
                                   delta=self.delta)
                right = ContextNode(self.env, type(self.learner), feature_1=self.feature_1, feature_2=True,
                                    delta=self.delta)
            else:
                raise NotImplementedError('Non-existing feature passed for the splitting.')
            left_x_ = np.sum(np.mean(left.learner.get_means(), axis=1))
            left_lower_bound = left_x_ - (- np.log(self.delta) / (2 * t)) ** 0.5
            right_x_ = np.sum(np.mean(right.learner.get_means(), axis=1))
            right_lower_bound = right_x_ - (- np.log(self.delta) / (2 * t)) ** 0.5
            children_lower_bound = (1 - fp[feature - 1]) * left_lower_bound + fp[feature - 1] * right_lower_bound
            if children_lower_bound >= best_lower_bound:
                best_lower_bound = children_lower_bound
                self.left = left
                self.right = right
        return self.left is not None and self.right is not None

    def is_leaf(self):
        return self.left is None and self.right is None

    def get_leaves(self):
        if self.is_leaf():
            return [self]
        return self.left.get_leaves() + self.right.get_leaves()
