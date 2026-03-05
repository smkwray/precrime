"""Baseline predictors for NIJ recidivism experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BaseRateModel:
    """Predicts the training-set base rate for every sample."""

    prob_: float = 0.5

    def fit(self, y: np.ndarray) -> "BaseRateModel":
        self.prob_ = float(np.clip(np.mean(y), 1e-6, 1.0 - 1e-6))
        return self

    def predict_proba(self, n: int) -> np.ndarray:
        return np.full(n, self.prob_, dtype=float)


@dataclass
class DemographicNaiveModel:
    """Predicts group-level base rates by demographic slices."""

    group_cols: tuple[str, ...] = ("Gender", "Race", "age_group")
    global_rate_: float = 0.5
    group_rates_: dict[tuple[str, ...], float] | None = None

    def fit(self, frame: pd.DataFrame, y: np.ndarray) -> "DemographicNaiveModel":
        use_cols = [col for col in self.group_cols if col in frame.columns]
        if not use_cols:
            self.global_rate_ = float(np.clip(np.mean(y), 1e-6, 1.0 - 1e-6))
            self.group_rates_ = {}
            return self

        keyed = frame[use_cols].fillna("Unknown").astype(str).copy()
        keyed["target"] = y.astype(float)

        rates: dict[tuple[str, ...], float] = {}
        for keys, sub in keyed.groupby(use_cols, dropna=False):
            keys_tuple = (keys,) if not isinstance(keys, tuple) else tuple(str(k) for k in keys)
            rates[keys_tuple] = float(np.clip(sub["target"].mean(), 1e-6, 1.0 - 1e-6))

        self.global_rate_ = float(np.clip(np.mean(y), 1e-6, 1.0 - 1e-6))
        self.group_rates_ = rates
        self.group_cols = tuple(use_cols)
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.group_rates_ is None:
            raise RuntimeError("Model must be fit before predict_proba")

        if not self.group_cols:
            return np.full(len(frame), self.global_rate_, dtype=float)

        keyed = frame[list(self.group_cols)].fillna("Unknown").astype(str)
        out = np.full(len(frame), self.global_rate_, dtype=float)
        for i, (_, row) in enumerate(keyed.iterrows()):
            key = tuple(row[col] for col in self.group_cols)
            out[i] = self.group_rates_.get(key, self.global_rate_)
        return np.clip(out, 1e-6, 1.0 - 1e-6)
