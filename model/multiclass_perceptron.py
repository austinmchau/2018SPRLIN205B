import numpy as np
from typing import Tuple, Union
from pprint import pprint

from dataset import Dataset, DSTC2
from model import Model


class MulticlassPerceptron(Model):

    def train(self, iterations: int):
        for i in range(iterations):
            print("iteration:", i)
            activations = self.weights * self.training_set.data.T
            arg_max = activations.argmax(0).A1
            predictions = [self.training_set.data_labels[i] for i in arg_max]

            for i, (p, a) in enumerate(zip(predictions, self.training_set.data_labels)):
                if p != a:
                    self.weights[self.training_set.labels.index(a)] += self.training_set.data[i]
                    self.weights[self.training_set.labels.index(p)] -= self.training_set.data[i]

    def predict(self, vector: Union[Tuple[float, ...], Tuple[Tuple[float, ...]]]):
        vector = np.matrix(vector)

        activations = self.weights * vector.T
        predictions = [self.training_set.labels[i] for i in activations.argmax(0).A1]
        return predictions


def data_from(utterances, labels):
    from collections import Counter
    import nltk
    counter = Counter()
    for u in utterances:
        # print(u)
        tokens = nltk.word_tokenize(u)
        counter.update(tokens)

    features = [w for w, _ in counter.most_common()]

    vecs = []
    for u in utterances:
        tokens = nltk.word_tokenize(u)
        c = Counter(tokens)
        vec = [c[t] for t in features]
        vecs.append(vec)

    return Dataset(features, labels, vecs)


if __name__ == '__main__':
    training = DSTC2.trainset(500)
    utterances, labels = training.read_json()

    print("original dataset size:", len(utterances))
    size = int(len(utterances) * 0.9)
    utterances = utterances[:size]
    labels = labels[:size]
    print("current dataset size:", len(utterances))

    training_data = data_from(utterances, labels)

    model = MulticlassPerceptron(training_data)

    # print(model.training_set.labels)

    model.train(iterations=10)

    # pprint(model.weights.shape)

    testing = DSTC2.testset(500)
    utterances_t, labels_t = training.read_json()
    # max_length_t = 500
    # utterances_t = sequence.pad_sequences(utterances_t, maxlen=max_length_t)

    testing_data = data_from(utterances_t, labels_t)

    a = model.accuracy(testing_data)
    pprint(model.weights)
    print(a)


