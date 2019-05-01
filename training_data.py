"""Training Data."""


from collections import namedtuple, defaultdict
from util import *


Thread = namedtuple('Thread', ['state', 'time', 'user', 'message'])


class TrainingData:
    """
    Represent training data object.

    The idea is to group data by state and to extract transitions.

    Example:
    observations = {
        "4": [[ "back", "Belgrade"], ...etc],
        "0": [[ "Mean", "girls", "does", "aaron", "samuels", "have", "gay", "It's"], ...etc]
    }

    transitions = {
        'user720904': ['4'],
        'user589836': ['4', '0', '4', '4', '0'],
        'user294926': ['4', '4'],
        'user393231': ['0'],
        'user524304': ['4', '4', '4', '4', '4', '4', '4', '4', '4']
    }
    """

    def __init__(self, observations: dict, transitions: dict):
        """
        Class constructor.

        :param observations: dict of observations grouped by state
        :param transitions:  dict of transitions grouped by subject
        """
        self._states_dict = observations
        self._transitions_dict = transitions

    @property
    def observations(self):
        return self._states_dict

    @observations.setter
    def observations(self, value):
        self._states_dict = value

    @property
    def transitions(self):
        return self._transitions_dict

    @transitions.setter
    def transitions(self, value):
        self._transitions_dict = value

    @staticmethod
    def parse_message(message: str) -> list:
        """Parse and filter out words in a message."""
        return [parse_word(x) for x in message.split(' ') if not stopword(x)]

    @classmethod
    def from_csv(cls, file) -> 'TrainingData':
        """
        Parse and convert the csv file to training data.

        :param file: csv file to parse
        :return: training data object
        """
        observations_dict = defaultdict(list)
        transitions_dict = defaultdict(list)

        for row in parse_csv(file):
            thread = Thread(*row)
            state, time, user, message = thread

            # group transitions by user
            transitions_dict[user] += [state]
            observations_dict[state].append(cls.parse_message(message))

        return cls(observations_dict, transitions_dict)
