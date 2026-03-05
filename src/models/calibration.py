"""Probability calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z_clip = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z_clip))


@dataclass
class PlattCalibrator:
    learning_rate: float = 0.05
    n_iter: int = 300

    a_: float = 1.0
    b_: float = 0.0

    def fit(self, raw_prob: np.ndarray, y_true: np.ndarray) -> "PlattCalibrator":
        p = np.clip(raw_prob, 1e-6, 1.0 - 1e-6)
        x = np.log(p / (1.0 - p))
        y = y_true.astype(float)

        a = 1.0
        b = 0.0
        n = len(y)
        for _ in range(self.n_iter):
            logits = a * x + b
            pred = _sigmoid(logits)
            err = pred - y

            grad_a = float(np.dot(err, x) / n)
            grad_b = float(np.mean(err))

            a -= self.learning_rate * grad_a
            b -= self.learning_rate * grad_b

        self.a_ = float(a)
        self.b_ = float(b)
        return self

    def predict(self, raw_prob: np.ndarray) -> np.ndarray:
        p = np.clip(raw_prob, 1e-6, 1.0 - 1e-6)
        x = np.log(p / (1.0 - p))
        out = _sigmoid(self.a_ * x + self.b_)
        return np.clip(out, 1e-6, 1.0 - 1e-6)


@dataclass
class IsotonicCalibrator:
    """Isotonic regression probability calibrator."""

    y_min: float = 0.0
    y_max: float = 1.0
    _model: object | None = None

    def fit(self, raw_prob: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        from sklearn.isotonic import IsotonicRegression

        p = np.clip(raw_prob, 1e-6, 1.0 - 1e-6)
        y = y_true.astype(float)
        model = IsotonicRegression(y_min=self.y_min, y_max=self.y_max, out_of_bounds="clip")
        model.fit(p, y)
        self._model = model
        return self

    def predict(self, raw_prob: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Calibrator must be fit before predict")
        p = np.clip(raw_prob, 1e-6, 1.0 - 1e-6)
        out = np.asarray(self._model.predict(p), dtype=float)
        return np.clip(out, 1e-6, 1.0 - 1e-6)
