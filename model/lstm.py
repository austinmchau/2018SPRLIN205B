from typing import List
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from model import Model
from dataset import Dataset


class LSTMModel(Model):

    def __init__(self, training_set: Dataset):
        super().__init__(training_set)
        self.features = 0
        self.training_set = training_set

        self.weights_dict = {
            label: np.array([0.0] * len(training_set.data_features))
            for label in training_set.data_labels
        }
        self.weights = np.asmatrix(np.zeros(
            (len(training_set.labels), len(training_set.data_features))
        ))

    def train(self, iterations: int):
        np.random.seed(7)
        top_words = 5000
        from dataset.dstc import DSTC2
        (X_train, y_train) = DSTC2.trainset(500).word_vecs()
        (X_test, y_test) = DSTC2.testset(500).word_vecs()

        size = int(len(X_train) * 1)

        print("original dataset size:", len(X_train))
        (X_train, y_train) = (X_train[:size], y_train[:size])

        # print(X_train[0])
        # print(y_train[0])
        print("current dataset size:", len(X_train))

        # truncate and pad input sequences
        max_review_length = 500
        X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
        X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
        # create the model
        embedding_vector_length = 32
        lstm = Sequential()
        lstm.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
        lstm.add(LSTM(100))
        lstm.add(Dense(1, activation='sigmoid'))
        lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(lstm.summary())
        lstm.fit(X_train, y_train, epochs=2, batch_size=64)
        # Final evaluation of the model
        scores = lstm.evaluate(X_test, y_test, verbose=0)
        print("Accuracy: {}".format(scores[1]))


    def predict(self, vector) -> List[str]:
        pass


def tutorial():
    # LSTM for sequence classification in the IMDB dataset

    # fix random seed for reproducibility

    # load the dataset but only keep the top n words, zero the rest
    top_words = 5000
    # (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
    # print(X_train[0])
    # print(y_train[0])



    (X_train, y_train) = DSTC2.trainset(500).word_vecs()
    (X_test, y_test) = DSTC2.testset(500).word_vecs()

    size = int(len(X_train) * 1)

    print("original dataset size:", len(X_train))
    (X_train, y_train) = (X_train[:size], y_train[:size])

    # print(X_train[0])
    # print(y_train[0])
    print("current dataset size:", len(X_train))

    # truncate and pad input sequences
    max_review_length = 500
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
    # create the model
    embedding_vector_length = 32
    lstm = Sequential()
    lstm.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    lstm.add(LSTM(100))
    lstm.add(Dense(1, activation='sigmoid'))
    lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(lstm.summary())
    lstm.fit(X_train, y_train, epochs=2, batch_size=64)
    # Final evaluation of the model
    scores = lstm.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: {}".format(scores[1]))


if __name__ == '__main__':
    tutorial()
