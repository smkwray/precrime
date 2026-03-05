"""XGBoost training, tuning, and attribution helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np
import optuna
import shap
from xgboost import XGBClassifier

from src.eval.metrics import brier_score


@dataclass
class XGBTuneResult:
    best_params: dict[str, float | int]
    best_score: float
    n_trials: int


def _n_jobs() -> int:
    cap = int(os.getenv("PRECRIME_N_JOBS", "16"))
    detected = os.cpu_count() or 4
    return max(1, min(cap, int(detected)))


def _default_xgb_params(seed: int) -> dict[str, float | int | str]:
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 1.0,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": _n_jobs(),
    }


def train_xgb(
    x_train: np.ndarray,
    y_train: np.ndarray,
    params: dict[str, float | int] | None = None,
    seed: int = 42,
) -> XGBClassifier:
    cfg = _default_xgb_params(seed)
    if params:
        cfg.update(params)

    model = XGBClassifier(**cfg)
    model.fit(x_train, y_train)
    return model


def _suggest_params(trial: optuna.Trial, seed: int) -> dict[str, float | int | str]:
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_estimators": trial.suggest_int("n_estimators", 200, 900),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 8.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 2.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": seed,
        "n_jobs": _n_jobs(),
    }


def tune_xgb(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    seed: int = 42,
    n_trials: int = 16,
) -> XGBTuneResult:
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, seed=seed)
        model = XGBClassifier(**params)
        model.fit(x_train, y_train)

        p_valid = model.predict_proba(x_valid)[:, 1]
        return float(brier_score(y_valid.tolist(), p_valid.tolist()))

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = dict(study.best_params)
    return XGBTuneResult(best_params=best, best_score=float(study.best_value), n_trials=n_trials)


def feature_importance_table(
    model: XGBClassifier,
    feature_names: list[str],
    top_k: int = 30,
) -> list[dict[str, float | str]]:
    imp = model.feature_importances_
    rows = [
        {"feature": str(name), "importance": float(score)}
        for name, score in zip(feature_names, imp)
    ]
    rows.sort(key=lambda r: float(r["importance"]), reverse=True)
    return rows[:top_k]


def shap_summary_table(
    model: XGBClassifier,
    x_matrix: np.ndarray,
    feature_names: list[str],
    max_samples: int = 2000,
    top_k: int = 30,
) -> list[dict[str, float | str]]:
    if x_matrix.shape[0] == 0:
        return []

    if x_matrix.shape[0] > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(x_matrix.shape[0], size=max_samples, replace=False)
        x_sample = x_matrix[idx]
    else:
        x_sample = x_matrix

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(x_sample)

    if isinstance(values, list):
        values_arr = np.asarray(values[0])
    else:
        values_arr = np.asarray(values)

    mean_abs = np.mean(np.abs(values_arr), axis=0)
    rows = [
        {"feature": str(name), "mean_abs_shap": float(score)}
        for name, score in zip(feature_names, mean_abs)
    ]
    rows.sort(key=lambda r: float(r["mean_abs_shap"]), reverse=True)
    return rows[:top_k]
