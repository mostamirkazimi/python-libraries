"""Layers with hand-written forward and backward passes (pure NumPy).

Every layer follows the same contract:
    forward(x, training)  -> output
    backward(grad_output) -> grad_input  (also fills ``self.grads``)
Parameters live in ``self.params`` and their gradients in ``self.grads``
under identical keys, so optimizers can update them in place.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class Layer:
    def __init__(self):
        self.params: dict[str, np.ndarray] = {}
        self.grads: dict[str, np.ndarray] = {}

    def forward(self, x, training=True):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError


class Dense(Layer):
    """Fully connected layer: y = xW + b  (He initialisation)."""

    def __init__(self, in_features, out_features, seed=None):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.params["W"] = rng.normal(0, np.sqrt(2.0 / in_features),
                                      (in_features, out_features))
        self.params["b"] = np.zeros(out_features)

    def forward(self, x, training=True):
        self.x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, grad):
        self.grads["W"] = self.x.T @ grad
        self.grads["b"] = grad.sum(axis=0)
        return grad @ self.params["W"].T


class Conv2D(Layer):
    """2-D convolution, fully vectorised (no Python loop over pixels).

    Forward uses ``sliding_window_view`` (a zero-copy strided view) + ``einsum``.
    Backward loops only over the kernel size (e.g. 3x3 = 9 iterations).
    Input shape: (N, C, H, W).
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, seed=None):
        super().__init__()
        rng = np.random.default_rng(seed)
        k = kernel_size
        fan_in = in_channels * k * k
        self.params["W"] = rng.normal(0, np.sqrt(2.0 / fan_in),
                                      (out_channels, in_channels, k, k))
        self.params["b"] = np.zeros(out_channels)
        self.k, self.stride, self.padding = k, stride, padding

    def forward(self, x, training=True):
        p, s, k = self.padding, self.stride, self.k
        xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
        self.xp_shape, self.x_shape = xp.shape, x.shape
        windows = sliding_window_view(xp, (k, k), axis=(2, 3))[:, :, ::s, ::s]
        self.windows = windows  # (N, C, Ho, Wo, k, k)
        out = np.einsum("nchwij,fcij->nfhw", windows, self.params["W"],
                        optimize=True)
        return out + self.params["b"][None, :, None, None]

    def backward(self, grad):
        s, k, p = self.stride, self.k, self.padding
        W = self.params["W"]
        self.grads["W"] = np.einsum("nchwij,nfhw->fcij", self.windows, grad,
                                    optimize=True)
        self.grads["b"] = grad.sum(axis=(0, 2, 3))
        dxp = np.zeros(self.xp_shape)
        Ho, Wo = grad.shape[2:]
        for i in range(k):
            for j in range(k):
                dxp[:, :, i:i + s * Ho:s, j:j + s * Wo:s] += np.einsum(
                    "nfhw,fc->nchw", grad, W[:, :, i, j], optimize=True)
        return dxp[:, :, p:dxp.shape[2] - p, p:dxp.shape[3] - p]


class MaxPool2D(Layer):
    """Non-overlapping max pooling implemented with reshape tricks."""

    def __init__(self, size=2):
        super().__init__()
        self.size = size

    def forward(self, x, training=True):
        n, c, h, w = x.shape
        p = self.size
        h2, w2 = h // p, w // p
        xr = x[:, :, :h2 * p, :w2 * p].reshape(n, c, h2, p, w2, p)
        out = xr.max(axis=(3, 5))
        self.xr, self.out, self.x_shape = xr, out, x.shape
        return out

    def backward(self, grad):
        mask = self.xr == self.out[:, :, :, None, :, None]
        mask = mask / mask.sum(axis=(3, 5), keepdims=True)  # split ties evenly
        dxr = mask * grad[:, :, :, None, :, None]
        n, c, h2, p, w2, _ = dxr.shape
        dx = np.zeros(self.x_shape)
        dx[:, :, :h2 * p, :w2 * p] = dxr.reshape(n, c, h2 * p, w2 * p)
        return dx


class Flatten(Layer):
    def forward(self, x, training=True):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self.shape)


class BatchNorm1D(Layer):
    """Batch normalisation with running statistics for inference."""

    def __init__(self, features, momentum=0.9, eps=1e-5):
        super().__init__()
        self.params["gamma"] = np.ones(features)
        self.params["beta"] = np.zeros(features)
        self.running_mean = np.zeros(features)
        self.running_var = np.ones(features)
        self.momentum, self.eps = momentum, eps

    def forward(self, x, training=True):
        if training:
            mu, var = x.mean(axis=0), x.var(axis=0)
            m = self.momentum
            self.running_mean = m * self.running_mean + (1 - m) * mu
            self.running_var = m * self.running_var + (1 - m) * var
        else:
            mu, var = self.running_mean, self.running_var
        self.inv_std = 1.0 / np.sqrt(var + self.eps)
        self.x_hat = (x - mu) * self.inv_std
        return self.params["gamma"] * self.x_hat + self.params["beta"]

    def backward(self, grad):
        n = grad.shape[0]
        self.grads["gamma"] = (grad * self.x_hat).sum(axis=0)
        self.grads["beta"] = grad.sum(axis=0)
        dxh = grad * self.params["gamma"]
        return (self.inv_std / n) * (
            n * dxh - dxh.sum(axis=0) - self.x_hat * (dxh * self.x_hat).sum(axis=0))


class Dropout(Layer):
    """Inverted dropout (no rescaling needed at inference)."""

    def __init__(self, rate=0.5, seed=None):
        super().__init__()
        self.rate = rate
        self.rng = np.random.default_rng(seed)

    def forward(self, x, training=True):
        if not training or self.rate == 0:
            self.mask = None
            return x
        self.mask = (self.rng.random(x.shape) >= self.rate) / (1 - self.rate)
        return x * self.mask

    def backward(self, grad):
        return grad if self.mask is None else grad * self.mask


class ReLU(Layer):
    def forward(self, x, training=True):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask


class Tanh(Layer):
    def forward(self, x, training=True):
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad):
        return grad * (1 - self.out ** 2)


class Sigmoid(Layer):
    def forward(self, x, training=True):
        self.out = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return self.out

    def backward(self, grad):
        return grad * self.out * (1 - self.out)
