import numpy as np
from typing import Tuple, Union

from dataset import Dataset
from model import Model


class MulticlassPerceptron(Model):
    """
    This class represents the multi-class perceptron. Initialize it with the training set, then call train() to
    create a multi-class perceptron trained on the given dataset.
    """

    def __init__(self, training_set: Dataset):
        self.training_set = training_set

        # creating weights for the perceptron with size wrt. the training set.
        self.weights = np.asmatrix(np.zeros(
            (len(training_set.labels), len(training_set.data_features))
        ))

    def train(self, iterations: int = 10):
        """
        Train the model on the given training_set. Adjust the weights based on the perceptron algorithm.
        :param iterations: How many iterations to train the model with
        :return: None
        """
        for i in range(iterations):
            print("iteration:", i)
            activations = self.weights * self.training_set.data.T
            arg_max = activations.argmax(0).A1
            predictions = [self.training_set.data_labels[i] for i in arg_max]

            # for each prediction, adjust weights of each corresponding class based on the error
            for i, (p, a) in enumerate(zip(predictions, self.training_set.data_labels)):
                if p != a:
                    self.weights[self.training_set.labels.index(a)] += self.training_set.data[i]
                    self.weights[self.training_set.labels.index(p)] -= self.training_set.data[i]

    def predict(self, vector: Union[Tuple[float, ...], Tuple[Tuple[float, ...]]]):
        """
        Predict a label based on the given vector/vectors. Require model to be trained.
        :param vector: vector/vectors representing the testing data.
        :return: The list of predictions
        """
        vector = np.matrix(vector)

        activations = self.weights * vector.T
        predictions = [self.training_set.labels[i] for i in activations.argmax(0).A1]
        return predictions

    def accuracy(self, testing_set: Dataset):
        """
        Calculate the accuracy of the model based on the given testing_set
        :param testing_set: testing data
        :return: accuracy value
        """
        correct, incorrect = 0, 0
        predictions = self.predict(testing_set.data)
        for p, a in zip(predictions, testing_set.data_labels):
            if p == a:
                correct += 1
            else:
                incorrect += 1

        return correct / (correct + incorrect)
