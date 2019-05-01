"""Naive Bayes."""

import math
from abc import ABC, abstractmethod
from collections import Counter
from collections import defaultdict

import numpy as np


class NaiveBayes(ABC):
    """Abstract Naive Bayes class."""

    @abstractmethod
    def predict(self, sample: list):
        """
        Compute probability and label the sample.

        :param sample: sample to be labeled
        :return: prediction
        """
        pass

    @abstractmethod
    def train(self, data):
        """
        Abstract method for training.

        :param data: training data
        """
        pass


class MultinomialNaiveBayes(NaiveBayes):

    def __init__(self, positive_data: list, negative_data: list, labels: dict, alpha=1):
        """
        Class constructor.

        :param positive_data: positive dataset as 2D list
        :param negative_data: negative dataset as 2D list
        :param labels: labels dict for positive and negative data, should be provided as follows:
            {'positive': <some_value>, 'negative': <some_value>}
        :param alpha: additive smoothing constant
        """
        self._positive_data = positive_data
        self._negative_data = negative_data
        self._labels = labels

        self._positive_bow = self.bag_of_words(positive_data)
        self._negative_bow = self.bag_of_words(negative_data)

        self._alpha = alpha

    @property
    def positive_data(self):
        return self._positive_data

    @property
    def negative_data(self):
        return self._negative_data

    @staticmethod
    def bag_of_words(sentences: list):
        return Counter(np.hstack(sentences))

    def unique_count(self) -> int:
        return len(set(np.hstack(self._positive_data)).union(np.hstack(self._negative_data)))

    def positive_count(self):
        return sum(self._positive_bow.values())

    def negative_count(self):
        return sum(self._negative_bow.values())

    def total_data(self):
        return len(self._positive_data) + len(self._negative_data)

    def predict(self, test: list) -> dict:
        """
        Compute the probability distribution based on the training data
        and return result as a dictionary.

        :param test: list of words
        :return: probabilities dict
        """
        total_mail = self.total_data()
        positive_count = self.positive_count()
        negative_count = self.negative_count()
        unique_count = self.unique_count()
        probs_dict = defaultdict(lambda: 0)

        # initial probabilities
        probs_dict[self._labels['positive']] += math.log(len(self._positive_data) / total_mail)
        probs_dict[self._labels['negative']] += math.log(len(self._negative_data) / total_mail)

        for word in test:
            probs_dict[self._labels['positive']] += math.log((self._positive_bow[word] + self._alpha) /
                                                             (positive_count + unique_count))
            probs_dict[self._labels['negative']] += math.log((self._negative_bow[word] + self._alpha) /
                                                             (negative_count + unique_count))
        # convert back to normal probabilities
        return {k: math.exp(v) for k, v in probs_dict.items()}

    def train(self, data):
        pass
