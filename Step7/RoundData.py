import numpy as np


class RoundData:
    def __init__(self, n_products, n_features):
        self.ctx_configs = []
        self.users = np.zeros(2**n_features)
        self.first_clicks = np.zeros((2**n_features, n_products), dtype=int)
        self.visits = np.zeros((2**n_features, n_products), dtype=int)
        self.conversions = np.zeros((2**n_features, n_products), dtype=int)
        self.rewards = np.zeros(2**n_features)
        self.sales = np.zeros((2**n_features, n_products), dtype=int)
        self.prod_rewards = np.zeros((2**n_features, n_products))