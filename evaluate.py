from collections import Counter
from pprint import pprint

from dataset import Dataset, DSTC2
from model import Model, MulticlassPerceptron


def test_perceptron(data_size: float):
    training = DSTC2.trainset(500).dataset(data_size)

    model = MulticlassPerceptron(training)

    # print(model.training_set.labels)

    model.train(iterations=10)

    # pprint(model.weights.shape)

    testing = DSTC2.testset(500).dataset(1.0, features=training.data_features)

    print(set(testing.data_features) ^ set(training.data_features))

    return model.accuracy(testing), model.fbeta_score(testing)


if __name__ == '__main__':
    scores = test_perceptron(1.0)
    print(scores)

