"""Shared splitting, encoding, and evaluation utilities for pipeline scripts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.metrics import auprc, auroc, brier_score, expected_calibration_error, log_loss


RANDOM_SEED = 42
N_FOLDS = 5


# ---------------------------------------------------------------------------
# Target detection
# ---------------------------------------------------------------------------

def extract_target_column(ds: pd.DataFrame) -> str:
    for candidate in ("target", "y"):
        if candidate in ds.columns:
            return candidate
    for col in ds.columns:
        if "recidivism_arrest_year" in str(col).lower():
            return str(col)
    raise ValueError("Unable to detect target column in dataset")


# ---------------------------------------------------------------------------
# Splitting helpers
# ---------------------------------------------------------------------------

def split_train_cal_test(
    y: np.ndarray,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3-way stratified split: fit (60%), cal (20%), test (20%)."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=seed, stratify=y,
    )
    fit_idx, cal_idx = train_test_split(
        train_idx, test_size=0.25, random_state=seed, stratify=y[train_idx],
    )
    return fit_idx, cal_idx, test_idx


def split_train_cal_select_test(
    y: np.ndarray,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """4-way stratified split: fit (48%), cal (16%), select (16%), test (20%).

    - fit_idx:    model training (and internal tune sub-split)
    - cal_idx:    calibrator fitting (Platt / Isotonic)
    - select_idx: calibration method selection + threshold derivation
    - test_idx:   final evaluation only (touched once)
    """
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=seed, stratify=y,
    )
    fit_idx, holdout_idx = train_test_split(
        train_idx, test_size=0.4, random_state=seed, stratify=y[train_idx],
    )
    cal_idx, select_idx = train_test_split(
        holdout_idx, test_size=0.5, random_state=seed, stratify=y[holdout_idx],
    )
    return fit_idx, cal_idx, select_idx, test_idx


def build_cv_splits(
    n_rows: int,
    n_folds: int = N_FOLDS,
    seed: int = RANDOM_SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n_rows)
    rng.shuffle(idx)
    fold_sizes = np.full(n_folds, n_rows // n_folds, dtype=int)
    fold_sizes[: n_rows % n_folds] += 1

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    cursor = 0
    for fold_size in fold_sizes:
        val_idx = idx[cursor : cursor + fold_size]
        train_idx = np.concatenate([idx[:cursor], idx[cursor + fold_size :]])
        splits.append((train_idx, val_idx))
        cursor += fold_size
    return splits


def fit_calibration_split(
    train_idx: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = train_idx.copy()
    rng.shuffle(shuffled)

    cut = max(1, int(0.8 * len(shuffled)))
    fit_idx = shuffled[:cut]
    cal_idx = shuffled[cut:]
    if len(cal_idx) == 0:
        cal_idx = fit_idx[-1:]
        fit_idx = fit_idx[:-1]
    if len(fit_idx) == 0:
        fit_idx = cal_idx
    return fit_idx, cal_idx


# ---------------------------------------------------------------------------
# Feature encoding (fit / transform)
# ---------------------------------------------------------------------------

def prepare_feature_matrix(
    frame: pd.DataFrame,
    *,
    id_columns: tuple[str, ...] = ("ID",),
    fit_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode a feature frame.

    If *fit_columns* is ``None`` (fit step), discover columns from *frame*.
    If *fit_columns* is provided (transform step), align to those columns:
    add missing as 0, drop extras.
    """
    model_frame = frame.drop(columns=list(id_columns), errors="ignore").copy()
    x_df = pd.get_dummies(model_frame, dummy_na=True)
    x_df = x_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if fit_columns is not None:
        for col in fit_columns:
            if col not in x_df.columns:
                x_df[col] = 0.0
        x_df = x_df[fit_columns]

    return x_df, x_df.columns.astype(str).tolist()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_list = y_true.astype(int).tolist()
    p_list = y_prob.astype(float).tolist()
    return {
        "brier": brier_score(y_list, p_list),
        "auroc": auroc(y_list, p_list),
        "auprc": auprc(y_list, p_list),
        "log_loss": log_loss(y_list, p_list),
        "ece": expected_calibration_error(y_list, p_list, n_bins=10),
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_float(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value:.5f}"
