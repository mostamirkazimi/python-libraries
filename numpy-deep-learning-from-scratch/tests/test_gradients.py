"""Every layer's analytic backward pass is verified against finite differences."""
import numpy as np

from npdl import (BatchNorm1D, Conv2D, Dense, Flatten, MaxPool2D, ReLU,
                  Sequential, SoftmaxCrossEntropy, Tanh)
from npdl.utils import gradient_check, numerical_gradient, relative_error

TOL = 1e-5


def _check(layers, x, y):
    return gradient_check(Sequential(layers), SoftmaxCrossEntropy(), x, y)


def test_dense_tanh_gradients():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(8, 5)), rng.integers(0, 3, 8)
    assert _check([Dense(5, 7, seed=1), Tanh(), Dense(7, 3, seed=2)], x, y) < TOL


def test_batchnorm_gradients():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(10, 4)), rng.integers(0, 3, 10)
    layers = [Dense(4, 6, seed=1), BatchNorm1D(6), ReLU(), Dense(6, 3, seed=2)]
    assert _check(layers, x, y) < TOL


def test_conv_pool_gradients():
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=(3, 2, 8, 8)), rng.integers(0, 4, 3)
    layers = [Conv2D(2, 3, kernel_size=3, stride=1, padding=1, seed=1), ReLU(),
              MaxPool2D(2), Flatten(), Dense(3 * 4 * 4, 4, seed=2)]
    assert _check(layers, x, y) < TOL


def test_conv_stride_input_gradient():
    """Check dL/dx (not only parameters) for a strided convolution."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(2, 2, 7, 7))
    conv = Conv2D(2, 3, kernel_size=3, stride=2, padding=1, seed=4)
    w = rng.normal(size=conv.forward(x).shape)
    f = lambda: float((conv.forward(x) * w).sum())
    conv.forward(x)
    analytic = conv.backward(w)
    assert relative_error(analytic, numerical_gradient(f, x)) < TOL
