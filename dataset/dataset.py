from typing import List, Tuple, Union
import numpy as np

from keras.preprocessing import sequence


class Dataset:
    """
    A wrapper class to represent the data set. Currently only used by the perceptron model.
    LSTM uses the the vectors directly.
    """

    @classmethod
    def from_set(cls, data, label, maxlen=None):
        """
        Convenient method for creating a Dataset from the vectors
        :param data: Feature vectors
        :param label: Label vector
        :param maxlen: maximum length of the feature vectors. Feature vectors padded to this size.
        :return: Dataset representation of the data.
        """
        return cls(
            features=tuple(range(len(data[0]) if maxlen is None else maxlen)),
            labels=label,
            matrix=[l for l in sequence.pad_sequences(data, maxlen=maxlen)]
        )

    @classmethod
    def from_dict(cls, features: Tuple[str, ...], data: List[Tuple[str, List[float]]]):
        """
        Convenient method to create Dataset from [{label: features}]
        :param features: label of each features, used only for sizes
        :param data: the list of dict containing the label and feature vectors
        :return: Dataset representation of the data
        """
        labels, vectors = zip(*data)
        return cls(
            features=features,
            labels=labels,
            matrix=vectors
        )

    def __init__(self,
                 features: Tuple[str, ...],
                 labels: Tuple[str, ...],
                 matrix: Union[Tuple[Tuple[float, ...]], np.matrix]
                 ):
        """
        Create dataset based on the data
        :param features: label of each features, used only for sizes
        :param labels: Label vector
        :param matrix: Feature vectors as either Tuple of Tuples or np.matrix
        """

        if not isinstance(matrix, np.matrix):
            matrix = np.matrix(matrix)
        if matrix.shape != (len(labels), len(features)):
            raise ValueError("Matrix shape must match data_features and data_labels shape. ({}), {}, {}".format(
                matrix.shape, len(labels), len(features)
            ))

        self.data_features = np.array(features)
        self.data_labels = np.array(labels)
        self.data = matrix
        self.labels = sorted(list(set(labels)))  # sorted list of all possible labels

    @property
    def vectors_sorted(self):
        """
        Sort the feature vectors based on the labels
        :return:
        """
        for label in self.data_labels:
            indexes = np.where(self.data_labels == label)[0]
            yield label, self.data[indexes, :]

    @property
    def vectors(self):
        """
        Property representing the data as both labels and feature vectors
        :return: zipped list of labels and feature vectors
        """
        return zip(self.data_labels, self.data)
