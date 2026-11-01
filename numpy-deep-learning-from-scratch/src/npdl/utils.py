"""Data helpers and numerical gradient checking."""
import numpy as np


def make_spirals(n_per_class=200, n_classes=3, noise=0.2, seed=0):
    """Classic non-linearly-separable toy dataset."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n_per_class)
    X, y = [], []
    for c in range(n_classes):
        angle = t * 4 + c * (2 * np.pi / n_classes) + rng.normal(0, noise, n_per_class)
        r = t
        X.append(np.c_[r * np.sin(angle), r * np.cos(angle)])
        y.append(np.full(n_per_class, c))
    return np.vstack(X), np.concatenate(y)


def train_test_split(X, y, test_size=0.2, seed=0):
    idx = np.random.default_rng(seed).permutation(len(X))
    cut = int(len(X) * (1 - test_size))
    return X[idx[:cut]], X[idx[cut:]], y[idx[:cut]], y[idx[cut:]]


def numerical_gradient(f, x, eps=1e-6):
    """Central finite differences of scalar function f w.r.t. array x (in place)."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps
        fp = f()
        x[i] = old - eps
        fm = f()
        x[i] = old
        grad[i] = (fp - fm) / (2 * eps)
    return grad


def relative_error(a, b):
    return np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b)))


def gradient_check(model, loss, x, y, eps=1e-6):
    """Return the max relative error between analytic and numeric gradients."""
    logits = model.forward(x, training=True)
    loss.forward(logits, y)
    model.backward(loss.backward())
    worst = 0.0
    for layer in model.layers:
        for name, p in layer.params.items():
            f = lambda: loss.forward(model.forward(x, training=True), y)
            num = numerical_gradient(f, p, eps)
            worst = max(worst, relative_error(layer.grads[name], num))
    return worst
