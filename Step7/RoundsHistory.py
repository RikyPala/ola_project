from Environment import RoundData


class RoundsHistory(object):

    history = []

    @staticmethod
    def append(round_data: RoundData):
        RoundsHistory.history.append(round_data)
