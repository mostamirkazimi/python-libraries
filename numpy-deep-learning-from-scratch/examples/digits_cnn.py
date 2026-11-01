"""Train a small CNN on scikit-learn's 8x8 digits dataset (scikit-learn is used
only to load the data; the whole network is pure NumPy)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

from npdl import (Adam, Conv2D, Dense, Dropout, Flatten, MaxPool2D, ReLU,
                  Sequential, SoftmaxCrossEntropy)
from npdl.utils import train_test_split


def main():
    d = load_digits()
    X = (d.images / 16.0)[:, None, :, :]  # (N, 1, 8, 8)
    Xtr, Xte, ytr, yte = train_test_split(X, d.target, test_size=0.2)
    model = Sequential([
        Conv2D(1, 8, 3, padding=1, seed=0), ReLU(), MaxPool2D(2),
        Conv2D(8, 16, 3, padding=1, seed=1), ReLU(), MaxPool2D(2),
        Flatten(), Dropout(0.3, seed=2), Dense(16 * 2 * 2, 10, seed=3),
    ])
    hist = model.fit(Xtr, ytr, SoftmaxCrossEntropy(), Adam(lr=3e-3, weight_decay=1e-4),
                     epochs=25, batch_size=32, validation=(Xte, yte))
    print(f"final test accuracy: {hist['val_acc'][-1]:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist["loss"], label="train"); ax[0].plot(hist["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(hist["val_acc"]); ax[1].set_title("Validation accuracy")
    fig.tight_layout(); fig.savefig("outputs/digits_cnn.png", dpi=130)


if __name__ == "__main__":
    main()
