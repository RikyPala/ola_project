from RoundData import RoundData
from UCB import UCB
from TS import TS


class RoundsHistory(object):
    history = [[], []]

    @staticmethod
    def append(round_data: RoundData, learner_class):
        if learner_class == UCB:
            RoundsHistory.history[0].append(round_data)
        elif learner_class == TS:
            RoundsHistory.history[1].append(round_data)
        else:
            raise NotImplementedError("Only UCB and TS are valid learner classes")
