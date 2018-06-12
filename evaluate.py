from pprint import pprint

from dataset import Dataset, DSTC2, Reuters
from model import MulticlassPerceptron, LSTMModel


def test_perceptron(data_size: float, iterations: int):
    """
    Evaluate the perceptron model on the DSTC2 data
    :param data_size: size of data sliced
    :param iterations: number of iterations to train the model
    :return:
    """
    training = DSTC2.trainset(500).dataset(data_size)

    model = MulticlassPerceptron(training)
    print("Evaluating perception using DSTC2. Using {:.1f} of data, training with {} iterations"
          .format(data_size, iterations))
    model.train(iterations=iterations)

    testing = DSTC2.testset(500).dataset(features=training.data_features)

    return 'pd', data_size, iterations, model.accuracy(testing)


def test_perceptron_reuters(data_size: float, iterations: int):
    """
    Evaluate the perceptron model on the Reuters data
    :param data_size: size of data sliced
    :param iterations: number of iterations to train the model
    :return:
    """
    reuters = Reuters(num_words=500, maxlen=500)

    # setting up dataset
    training_data, training_labels = reuters.training_set()
    size = int(len(training_labels) * data_size)
    training_data, training_labels = training_data[:size], training_labels[:size]
    training_set = Dataset.from_set(training_data, training_labels, maxlen=500)

    model = MulticlassPerceptron(training_set)
    print("Evaluating perception using Reuters. Using {:.1f} of data, training with {} iterations"
          .format(data_size, iterations))
    model.train(iterations=iterations)

    testing_data, testing_labels = reuters.testing_set()
    testing_set = Dataset.from_set(testing_data, testing_labels, maxlen=500)

    return 'pr', data_size, iterations, model.accuracy(testing_set)


def test_lstm(data_size: float, epoch: int, batch_size: int = 64):
    """
    Evaluate the LSTM model on the DSTC2 data
    :param data_size: size of data sliced
    :param epoch: number of epochs to train the model
    :param batch_size: batch size for each training
    :return:
    """
    training_data, training_labels = DSTC2.trainset(500).word_vecs(raw_label=True)

    model = LSTMModel(training_data, training_labels, max_feature_length=50)
    model.verbose = 1
    model.train(data_size, epoch, batch_size)

    testing_data, testing_labels = DSTC2.testset(500).word_vecs(raw_label=True)
    return 'ld', data_size, epoch, model.predict(testing_data, testing_labels)


def test_lstm_reuters(data_size: float, epoch: int, batch_size: int = 64):
    """
    Evaluate the LSTM model on the Reuters data
    :param data_size: size of data sliced
    :param epoch: number of epochs to train the model
    :param batch_size: batch size for each training
    :return:
    """
    reuters = Reuters(num_words=500, maxlen=500)
    training_data, training_labels = reuters.training_set()
    testing_data, testing_labels = reuters.testing_set()

    model = LSTMModel(training_data, training_labels, max_feature_length=500, top_words=5000)
    model.verbose = 1
    model.train(data_size, epoch, batch_size)

    return 'lr', data_size, epoch, model.predict(testing_data, testing_labels)


if __name__ == '__main__':
    """
    This is where the data is collected.
    Comment out whichever model and trials are not wanted
    Then change the data_size to the required datasize
    Run the model to obatin results
    Run the script with multiple instances and varying datasizes to batch-collect data in parallel
    """

    data_size = 1.0

    scores_perceptron = [test_perceptron(data_size, iteration) for iteration in [10, 50, 100, 500]]
    pprint(scores_perceptron)

    scores_perceptron_reuters = [test_perceptron_reuters(data_size, iteration) for iteration in [10, 50, 100, 500]]
    pprint(scores_perceptron_reuters)

    scores_lstm = [test_lstm(data_size, epoch) for epoch in [3, 6, 9, 12]]
    pprint(scores_lstm)

    scores_lstm_reuters = [test_lstm_reuters(data_size, epoch) for epoch in [3, 6, 9, 12]]
    pprint(scores_lstm_reuters)

