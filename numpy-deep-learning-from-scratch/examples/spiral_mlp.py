"""Train an MLP on the 3-class spiral dataset and plot the decision boundary."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from npdl import Adam, Dense, ReLU, Sequential, SoftmaxCrossEntropy
from npdl.utils import make_spirals, train_test_split


def main():
    X, y = make_spirals(200, 3, noise=0.15)
    Xtr, Xte, ytr, yte = train_test_split(X, y)
    model = Sequential([Dense(2, 64, seed=0), ReLU(), Dense(64, 64, seed=1),
                        ReLU(), Dense(64, 3, seed=2)])
    hist = model.fit(Xtr, ytr, SoftmaxCrossEntropy(), Adam(lr=0.01), epochs=100,
                     batch_size=32, validation=(Xte, yte), verbose=False)
    print(f"test accuracy: {hist['val_acc'][-1]:.3f}")

    xx, yy = np.meshgrid(np.linspace(-1.1, 1.1, 300), np.linspace(-1.1, 1.1, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict(grid).argmax(axis=1).reshape(xx.shape)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].contourf(xx, yy, zz, alpha=0.35, cmap="viridis")
    ax[0].scatter(X[:, 0], X[:, 1], c=y, s=12, cmap="viridis", edgecolor="k", lw=0.2)
    ax[0].set_title("Decision boundary")
    ax[1].plot(hist["loss"], label="train loss")
    ax[1].plot(hist["val_loss"], label="val loss")
    ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].set_title("Loss")
    fig.tight_layout()
    fig.savefig("outputs/spiral_mlp.png", dpi=130)


if __name__ == "__main__":
    main()
