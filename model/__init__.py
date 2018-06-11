from abc import ABC, abstractmethod
from dataset.dataset import Dataset
import numpy as np
from collections import Counter
from typing import List


class Model(ABC):

    def __init__(self, training_set: Dataset):
        self.features = 0
        self.training_set = training_set

        self.weights_dict = {
            label: np.array([0.0] * len(training_set.data_features))
            for label in training_set.data_labels
        }
        self.weights = np.asmatrix(np.zeros(
            (len(training_set.labels), len(training_set.data_features))
        ))

    @abstractmethod
    def train(self, iterations: int):
        pass

    @abstractmethod
    def predict(self, vector) -> List[str]:
        pass

    def precision(self, testing_set: Dataset):
        num_correct = Counter()
        num_total = Counter()

        predictions = self.predict(testing_set.data)
        for p, a in zip(predictions, testing_set.data_labels):
            if p == a:
                num_correct[a] += 1
                num_total[a] += 1
            else:
                num_total[p] += 1

        return {c: (num_correct[c] / num_total[c])
                for c in testing_set.labels + self.training_set.labels if num_total[c] != 0}

    def recall(self, testing_set: Dataset):
        num_correct = Counter()
        num_total = Counter()

        predictions = self.predict(testing_set.data)
        for p, a in zip(predictions, testing_set.data_labels):
            if p == a:
                num_correct[a] += 1
            num_total[a] += 1

        return {c: (num_correct[c] / num_total[c]) for c in testing_set.labels}

    def accuracy(self, testing_set: Dataset):
        correct, incorrect = 0, 0
        predictions = self.predict(testing_set.data)
        for p, a in zip(predictions, testing_set.data_labels):
            if p == a:
                correct += 1
            else:
                incorrect += 1

        return correct / (correct + incorrect)

    def fbeta_score(self, testing_set: Dataset):
        precision = self.precision(testing_set)
        recall = self.recall(testing_set)
        return {
            c: 2 * ((precision[c] * recall[c]) / (precision[c] + recall[c]))
            for c in precision
        }


from model.multiclass_perceptron import MulticlassPerceptron
from model.lstm import LSTMModel