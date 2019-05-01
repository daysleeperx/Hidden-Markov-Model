"""HMM Tests."""

import unittest

from prettytable import PrettyTable

from hidden_markov_model import transition_matrix, sensor_model_from_nb, HiddenMarkovModel
from training_data import TrainingData
from util import *
from random_variable import RandomVariable


POSITIVE_CLASS = '4'
NEGATIVE_CLASS = '0'

LABELS = {
    'positive': POSITIVE_CLASS,
    'negative': NEGATIVE_CLASS
}


def test_user(test_file: str, training_file: str):
    test_data = parse_sample(test_file)
    training_data = TrainingData.from_csv(training_file)
    random_variable = RandomVariable('mood', [POSITIVE_CLASS, NEGATIVE_CLASS])
    transition_model = transition_matrix(training_data.observations.keys(), training_data.transitions.values())
    sensor_model = sensor_model_from_nb(test_data, training_data, LABELS)
    hmm = HiddenMarkovModel(random_variable, sensor_model, transition_model)

    result = HiddenMarkovModel.forward(hmm, {POSITIVE_CLASS: 0.5, NEGATIVE_CLASS: 0.5})

    table = PrettyTable(['good_mood', 'bad_mood'])
    table.add_row(result[-1].values())
    return table


class TestHMM(unittest.TestCase):

    def test_user_a(self):
        print(test_user('test_data/user_a.txt', 'training/users_5000.csv'))

    def test_user_b(self):
        print(test_user('test_data/user_b.txt', 'training/users_5000.csv'))

    def test_user_c(self):
        print(test_user('test_data/user_c.txt', 'training/users_5000.csv'))

    def test_user_d(self):
        print(test_user('test_data/user_d.txt', 'training/users_5000.csv'))


if __name__ == '__main__':
    unittest.main()
