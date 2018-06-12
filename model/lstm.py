import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

from model import Model


class LSTMModel(Model):
    """
    This model represents the LSTM model. It uses the keras package for training, but the class provides
    a wrapper for the setup of the LSTM model.
    """

    def __init__(self, training_vectors, training_labels, max_feature_length: int, top_words=500):
        """
        Initialize the model with the training data.
        :param training_vectors: The feature vectors of the training data
        :param training_labels: The labels for the training data
        :param max_feature_length: Feature vectors must be padded to this length.
        :param top_words: During word embedding, how many words are to be tagged. The rest will be <unk>
        """

        # convenient flag to set keras verbose level
        self.verbose = 1

        self.training_vectors, self.training_labels = training_vectors, training_labels
        self.max_feature_length = max_feature_length

        # create the model
        self.top_words = top_words
        self.embedding_vector_length = 32
        self.lstm = Sequential()

    def train(self, data_size: float = 1.0, epoch: int = 2, batch_size:int = 64):
        """
        Train the LSTM model based on the training data.
        :param data_size: how much to slice from the original dataset.
        :param epoch: How many epochs the model will be trained.
        :param batch_size: batch size of the training
        :return:
        """
        size = int(len(self.training_vectors) * data_size)

        # slicing the dataset
        print("original dataset size:", len(self.training_vectors))
        self.training_vectors, self.training_labels = self.training_vectors[:size], self.training_labels[:size]
        print("current dataset size:", len(self.training_vectors))

        # structuring the data for training
        training_vectors = sequence.pad_sequences(self.training_vectors, maxlen=self.max_feature_length)
        training_labels = self._encode_labels(self.training_labels)

        # build the model
        self._build_model(label_size=training_labels.shape[1])

        # running the model for training
        # print(self.lstm.summary())
        self.lstm.fit(training_vectors, training_labels, epochs=epoch, batch_size=batch_size, verbose=self.verbose)

    def predict(self, testing_vectors, testing_labels):
        """
        Run the model against the testing data for scoring
        :param testing_vectors: The feature vectors of the testing data
        :param testing_labels: The labels for the testing data
        :return: Accuracy score
        """

        # structuring the data for testing
        testing_vectors = sequence.pad_sequences(testing_vectors, maxlen=self.max_feature_length)
        testing_labels = self._encode_labels(testing_labels, fitting_labels=self.training_labels)

        # scoring the model
        scores = self.lstm.evaluate(testing_vectors, testing_labels, verbose=self.verbose)
        print("Accuracy: {}".format(scores[1]))
        return scores[1]

    def _encode_labels(self, labels, fitting_labels=None):
        """
        Transform the labels from a list of labels to a matrix one-hot representation
        :param labels: dataset labels
        :param fitting_labels: provide a list of labels should the dataset labels not containing all the possible
                               labels
        :return: a on-hot matrix of the labels
        """
        encoder = LabelEncoder()
        encoder.fit(labels if fitting_labels is None else fitting_labels)
        encoded_labels = encoder.transform(labels)
        return np_utils.to_categorical(encoded_labels)

    def _build_model(self, label_size: int):
        """
        Create the LSTM model. Word embedding -> LSTM layer with 100 memory units -> softmax output layer of labels
        :param label_size: how many labels are there
        :return:
        """
        self.lstm.add(Embedding(self.top_words, self.embedding_vector_length, input_length=self.max_feature_length))
        self.lstm.add(LSTM(100))
        self.lstm.add(Dense(label_size, activation='softmax'))
        self.lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
