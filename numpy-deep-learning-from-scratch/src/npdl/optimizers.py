"""Optimizers that update parameters in place."""
import numpy as np


class Optimizer:
    def __init__(self, lr, weight_decay=0.0):
        self.lr, self.weight_decay = lr, weight_decay

    def step(self, params_and_grads):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9, weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self.momentum = momentum
        self.v = {}

    def step(self, params_and_grads):
        for i, (p, g) in enumerate(params_and_grads):
            g = g + self.weight_decay * p
            v = self.v.get(i, np.zeros_like(p))
            v = self.momentum * v - self.lr * g
            self.v[i] = v
            p += v


class Adam(Optimizer):
    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self.b1, self.b2, self.eps = *betas, eps
        self.m, self.v, self.t = {}, {}, 0

    def step(self, params_and_grads):
        self.t += 1
        for i, (p, g) in enumerate(params_and_grads):
            g = g + self.weight_decay * p
            m = self.b1 * self.m.get(i, 0) + (1 - self.b1) * g
            v = self.b2 * self.v.get(i, 0) + (1 - self.b2) * g * g
            self.m[i], self.v[i] = m, v
            m_hat = m / (1 - self.b1 ** self.t)
            v_hat = v / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
