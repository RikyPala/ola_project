from Environment import Environment


class ContextNode:

    def __init__(self, env: Environment, learner_class, feature_1=None, feature_2=None):
        self.left = None
        self.right = None
        self.env = env
        self.learner = learner_class(env, feature_1, feature_2)
        self.feature_1 = feature_1
        self.feature_2 = feature_2

    def get_splittable_features(self):
        splittable_features = []
        if self.feature_1 is not None:
            splittable_features.append(1)
        if self.feature_2 is not None:
            splittable_features.append(2)
        return splittable_features

    def split(self, feature_to_split):
        if feature_to_split == 1:
            self.left = ContextNode(self.env, type(self.learner), feature_1=False, feature_2=self.feature_2)
            self.right = ContextNode(self.env, type(self.learner), feature_1=True, feature_2=self.feature_2)
        if feature_to_split == 2:
            self.left = ContextNode(self.env, type(self.learner), feature_1=self.feature_1, feature_2=False)
            self.right = ContextNode(self.env, type(self.learner), feature_1=self.feature_1, feature_2=True)
