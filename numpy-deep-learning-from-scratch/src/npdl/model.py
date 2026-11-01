"""Sequential container with a Keras-like ``fit`` / ``predict`` API."""
import numpy as np


class Sequential:
    def __init__(self, layers):
        self.layers = list(layers)

    def forward(self, x, training=True):
        for layer in self.layers:
            x = layer.forward(x, training)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self):
        """List of (param, grad) pairs; arrays are shared so updates are in place."""
        return [(layer.params[k], layer.grads[k])
                for layer in self.layers for k in layer.params]

    def predict(self, x, batch_size=256):
        out = [self.forward(x[i:i + batch_size], training=False)
               for i in range(0, len(x), batch_size)]
        return np.concatenate(out)

    def evaluate(self, x, y, loss):
        logits = self.predict(x)
        return loss.forward(logits, y), (logits.argmax(axis=1) == y).mean()

    def fit(self, x, y, loss, optimizer, epochs=10, batch_size=64,
            validation=None, seed=0, verbose=True):
        rng = np.random.default_rng(seed)
        history = {"loss": [], "val_loss": [], "val_acc": []}
        for epoch in range(1, epochs + 1):
            idx = rng.permutation(len(x))
            losses = []
            for i in range(0, len(x), batch_size):
                b = idx[i:i + batch_size]
                logits = self.forward(x[b], training=True)
                losses.append(loss.forward(logits, y[b]))
                self.backward(loss.backward())
                optimizer.step(self.parameters())
            history["loss"].append(float(np.mean(losses)))
            msg = f"epoch {epoch:3d}/{epochs}  loss {history['loss'][-1]:.4f}"
            if validation is not None:
                vl, va = self.evaluate(*validation, loss)
                history["val_loss"].append(float(vl))
                history["val_acc"].append(float(va))
                msg += f"  val_loss {vl:.4f}  val_acc {va:.3f}"
            if verbose:
                print(msg)
        return history
