"""L1-regularized logistic regression via proximal gradient descent."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z_clip = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z_clip))


def _soft_threshold(w: np.ndarray, lam: float) -> np.ndarray:
    return np.sign(w) * np.maximum(np.abs(w) - lam, 0.0)


@dataclass
class LassoLogisticRegression:
    learning_rate: float = 0.05
    n_iter: int = 500
    l1: float = 5e-4

    weights_: np.ndarray | None = None
    bias_: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LassoLogisticRegression":
        n, d = x.shape
        self.weights_ = np.zeros(d, dtype=float)
        self.bias_ = 0.0

        for _ in range(self.n_iter):
            logits = x @ self.weights_ + self.bias_
            probs = _sigmoid(logits)
            err = probs - y

            grad_w = (x.T @ err) / n
            grad_b = float(np.mean(err))

            self.weights_ -= self.learning_rate * grad_w
            self.weights_ = _soft_threshold(self.weights_, self.learning_rate * self.l1)
            self.bias_ -= self.learning_rate * grad_b

        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Model must be fit before predict_proba")
        logits = x @ self.weights_ + self.bias_
        probs = _sigmoid(logits)
        return np.clip(probs, 1e-6, 1.0 - 1e-6)
