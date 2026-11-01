"""Loss functions."""
import numpy as np


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)  # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SoftmaxCrossEntropy:
    """Fused softmax + cross-entropy (stable log-sum-exp, simple gradient).

    ``y`` holds integer class labels of shape (N,).
    """

    def forward(self, logits, y):
        z = logits - logits.max(axis=1, keepdims=True)
        log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        self.probs = np.exp(log_probs)
        self.y = y
        return -log_probs[np.arange(len(y)), y].mean()

    def backward(self):
        grad = self.probs.copy()
        grad[np.arange(len(self.y)), self.y] -= 1
        return grad / len(self.y)
