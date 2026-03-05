"""Simple logistic regression trained with gradient descent."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z_clip = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z_clip))


@dataclass
class LogisticRegressionGD:
    learning_rate: float = 0.05
    n_iter: int = 400
    l2: float = 1e-3

    weights_: np.ndarray | None = None
    bias_: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogisticRegressionGD":
        n, d = x.shape
        self.weights_ = np.zeros(d, dtype=float)
        self.bias_ = 0.0

        for _ in range(self.n_iter):
            logits = x @ self.weights_ + self.bias_
            probs = _sigmoid(logits)
            err = probs - y

            grad_w = (x.T @ err) / n + self.l2 * self.weights_
            grad_b = float(np.mean(err))

            self.weights_ -= self.learning_rate * grad_w
            self.bias_ -= self.learning_rate * grad_b

        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Model must be fit before predict_proba")
        logits = x @ self.weights_ + self.bias_
        probs = _sigmoid(logits)
        return np.clip(probs, 1e-6, 1.0 - 1e-6)
