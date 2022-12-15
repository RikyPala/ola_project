from RoundData import RoundData


class RoundsHistory(object):

    history = [[], []]
    UCB_index = 0
    TS_index = 1

    @staticmethod
    def append(round_data: RoundData, learner_class):
        if learner_class.__name__ == 'UCB':
            RoundsHistory.history[RoundsHistory.UCB_index].append(round_data)
        elif learner_class.__name__ == 'TS':
            RoundsHistory.history[RoundsHistory.TS_index].append(round_data)
        else:
            raise NotImplementedError("Only UCB and TS are valid learner classes")
