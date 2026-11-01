import numpy as np

from npdl import Adam, Dense, Dropout, ReLU, SGD, Sequential, SoftmaxCrossEntropy
from npdl.utils import make_spirals, train_test_split


def test_mlp_learns_spirals():
    X, y = make_spirals(150, 3, noise=0.15, seed=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, seed=0)
    model = Sequential([Dense(2, 64, seed=0), ReLU(), Dense(64, 64, seed=1),
                        ReLU(), Dense(64, 3, seed=2)])
    model.fit(Xtr, ytr, SoftmaxCrossEntropy(), Adam(lr=0.01), epochs=80,
              batch_size=32, verbose=False)
    _, acc = model.evaluate(Xte, yte, SoftmaxCrossEntropy())
    assert acc > 0.9


def test_sgd_momentum_decreases_loss():
    X, y = make_spirals(100, 2, seed=1)
    model = Sequential([Dense(2, 16, seed=0), ReLU(), Dense(16, 2, seed=1)])
    hist = model.fit(X, y, SoftmaxCrossEntropy(), SGD(lr=0.05), epochs=20,
                     verbose=False)
    assert hist["loss"][-1] < hist["loss"][0]


def test_dropout_is_identity_at_inference():
    x = np.ones((4, 10))
    d = Dropout(0.5, seed=0)
    assert np.array_equal(d.forward(x, training=False), x)
    assert not np.array_equal(d.forward(x, training=True), x)
