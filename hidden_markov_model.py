"""Hidden Markov Model."""


from nb.naive_bayes import MultinomialNaiveBayes
from random_variable import RandomVariable
from training_data import TrainingData
from util import normalize


def transition_matrix(states: list, data: list) -> dict:
    """
    Convert the data to a transition probability matrix from a 2D list of transitions.

    Example:

    transition_probability = {
        'Rainy' : {'Rainy': 0.7, 'Sunny': 0.3},
        'Sunny' : {'Rainy': 0.4, 'Sunny': 0.6},
    }

    :param states: list of possible states
        ['Rainy', 'Rainy', 'Sunny', 'Rainy']
    :param data: original list
    :return: transition matrix as a dict
    """
    output = {x: {k: 0 for k in states} for x in states}  # initial matrix of zeros

    for row in data:
        for (i, j) in zip(row, row[1:]):
            output[i][j] += 1

    return {x: {k: output[x][k] / sum(output[x].values()) for k in output[x].keys()} for x in output.keys()}


def sensor_model_from_nb(test_data: list, training_data: 'TrainingData', labels: dict) -> list:
    """
    Convert data into a sensor model from a 2D list.
    Data will be presented as index -> observation at time <index>. Values would correspond to
    emission probability based on state.

    Naive Bayes classifier would be used to compute the probability distribution.

    Example:

    sensor_model = [
        {'Rainy': 0.7, 'Sunny': 0.3},
        {'Rainy': 0.4, 'Sunny': 0.6}
    ]

    :param test_data:
    :param labels:
    :param training_data:
    :return: list of probability dicts
    """
    # train the NB classifier
    positive_ = labels['positive']
    negative_ = labels['negative']
    nb = MultinomialNaiveBayes(training_data.observations[positive_], training_data.observations[negative_], labels)

    states = training_data.observations.keys()

    return [{k: nb.predict(item)[k] for k in states} for item in test_data]


class HiddenMarkovModel:
    """
    Represent Hidden Markov Model object.

    Statistical Markov model in which the system being modeled is assumed
    to be a Markov process with unobservable (i.e. hidden) states.
    """
    def __init__(self, state_variable: 'RandomVariable', sensor_model: list, transition_model: dict):
        """
        Class constructor.

        :param state_variable:
        :param sensor_model:
        :param transition_model:
        """
        self._state_variable = state_variable
        self._sensor_model = sensor_model
        self._transition_model = transition_model

    @property
    def state_variable(self):
        return self._state_variable

    @state_variable.setter
    def state_variable(self, value: 'RandomVariable'):
        self._state_variable = value

    @property
    def sensor_model(self):
        return self._sensor_model

    @sensor_model.setter
    def sensor_model(self, value):
        self._sensor_model = value

    @property
    def transition_model(self):
        return self._transition_model

    @transition_model.setter
    def transition_model(self, value):
        self._transition_model = value

    @staticmethod
    def forward(hmm: 'HiddenMarkovModel', start_prob):
        transition_model = hmm.transition_model
        observations = hmm.sensor_model
        states = hmm.state_variable.domain

        history = []
        prev_state = {}

        for i, observation in enumerate(observations):
            current = {}
            for state in states:
                if i == 0:
                    prev_probs_sum = start_prob[state]

                else:
                    prev_probs_sum = sum(prev_state[k] * transition_model[state][k] for k in states)

                state_ = observations[i][state]
                current[state] = state_ * prev_probs_sum

            current = normalize(current)
            history.append(current)
            prev_state = current

        return history
