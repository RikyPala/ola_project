from abc import abstractmethod

import numpy as np


class Learner:

    def __init__(self, arms):
        self.t = 0
        self.arms = arms

    '''
    self.arms : array of Arm objects
    initialization in the specific class (UCS, TS)
    '''

    def pull(self):
        best_configuration = (0,)
        best_sample = 0
        for idx, arm in np.ndenumerate(self.arms):
            sample = arm.sample()
            if sample > best_sample:
                best_sample = sample
                best_configuration = idx
        return best_configuration

    @abstractmethod
    def update(self, configuration, reward):
        pass
