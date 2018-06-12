from keras.datasets import reuters


class Reuters:
    """
    Class wrapping the keras reuters dataset
    """

    def __init__(self, num_words: int, maxlen: int):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = reuters.load_data(num_words=num_words,
                                                                                     maxlen=maxlen)

    def training_set(self):
        return self.x_train, self.y_train

    def testing_set(self):
        return self.x_test, self.y_test
