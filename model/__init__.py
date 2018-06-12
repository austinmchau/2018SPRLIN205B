from abc import ABC


class Model(ABC):
    """
    Base class for all models
    """
    pass


from model.multiclass_perceptron import MulticlassPerceptron
from model.lstm import LSTMModel
