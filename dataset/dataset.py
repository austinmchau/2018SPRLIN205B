from typing import List, Tuple
import numpy as np


class Dataset:

    @classmethod
    def from_dict(cls, features: Tuple[str, ...], data: List[Tuple[str, List[float]]]):
        labels, vectors = zip(*data)
        return cls(
            features=features,
            labels=labels,
            matrix=vectors
        )

    def __init__(self,
                 features: Tuple[str, ...],
                 labels: Tuple[str, ...],
                 matrix: Tuple[Tuple[float, ...]]
                 ):

        matrix = np.matrix(matrix)
        if matrix.shape != (len(labels), len(features)):
            raise ValueError("Matrix shape must match data_features and data_labels shape.")

        self.data_features = np.array(features)
        self.data_labels = np.array(labels)
        self.data = matrix
        self.labels = sorted(list(set(labels)))

    @property
    def vectors_sorted(self):
        for label in self.data_labels:
            indexes = np.where(self.data_labels == label)[0]
            yield label, self.data[indexes, :]

    @property
    def vectors(self):
        return zip(self.data_labels, self.data)

    @classmethod
    def sample_training_set(cls):
        return cls.from_dict(('a', 'b', 'c', 'd'), [
            ('apples', 	[100.0, 0.0, 0.0, 0.0]),
            ('oranges', [0.0, 1.0, 0.0, 0.0]),
            ('bananas', [0.0, 0.0, 1.0, 0.0]),
            ('peaches', [0.0, 0.0, 0.0, 1.0]),
            ('oranges',	[1.0, 1.0, 0.0, 0.0]),
        ])

    @classmethod
    def sample_testing_set(cls):
        return cls.from_dict(('a', 'b', 'c', 'd'), [
            ('oranges', [1.0, 0.0, 0.0, 0.0]),
            ('oranges', [1.0, 1.0, 0.0, 0.0]),
            ('bananas', [0.0, 0.0, 1.0, 0.0]),
            ('peaches', [0.0, 0.0, 0.0, 1.0])
        ])


if __name__ == '__main__':
    from pprint import pprint
    d = Dataset.sample_training_set()
    v = [m for m in d.vectors]
    pprint(v)
