"""npdl: a small deep-learning library written from scratch with NumPy only."""

from .layers import (BatchNorm1D, Conv2D, Dense, Dropout, Flatten, Layer,
                     MaxPool2D, ReLU, Sigmoid, Tanh)
from .losses import SoftmaxCrossEntropy
from .model import Sequential
from .optimizers import SGD, Adam

__all__ = [
    "Layer", "Dense", "Conv2D", "MaxPool2D", "Flatten", "BatchNorm1D",
    "Dropout", "ReLU", "Tanh", "Sigmoid", "SoftmaxCrossEntropy",
    "Sequential", "SGD", "Adam",
]
__version__ = "0.1.0"
