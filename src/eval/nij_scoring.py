"""NIJ Recidivism Forecasting Challenge-style scoring helpers.

This module implements a small subset of the NIJ challenge scoring logic so we can
report "NIJ-style" metrics on our own held-out split. These are *not* official
leaderboard scores because NIJ used a separate held-out test set.
"""

from __future__ import annotations

import math
from typing import Callable

from src.eval.metrics import ArrayLike, brier_score


def _is_nan(value: float) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _prepare_group_inputs(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    group: ArrayLike,
) -> tuple[list[int], list[float], list[str]]:
    y_t = [int(v) for v in y_true]
    y_p = [float(v) for v in y_prob]
    g = [str(v) for v in group]

    if len(y_t) != len(y_p):
        raise ValueError("y_true and y_prob must have the same length")
    if len(y_t) != len(g):
        raise ValueError("group must match y_true length")
    if not y_t:
        raise ValueError("inputs cannot be empty")

    for yt in y_t:
        if yt not in (0, 1):
            raise ValueError("y_true must contain only 0/1 labels")
    for yp in y_p:
        if yp < 0.0 or yp > 1.0:
            raise ValueError("y_prob must be in [0, 1]")

    return y_t, y_p, g


def _metric_by_group(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    group: ArrayLike,
    metric_fn: Callable[[ArrayLike, ArrayLike], float],
) -> dict[str, float]:
    y_t, y_p, g = _prepare_group_inputs(y_true, y_prob, group)
    out: dict[str, float] = {}
    for label in sorted(set(g)):
        idx = [i for i, value in enumerate(g) if value == label]
        out[label] = float(metric_fn([y_t[i] for i in idx], [y_p[i] for i in idx]))
    return out


def brier_by_sex(y_true: ArrayLike, y_prob: ArrayLike, sex: ArrayLike) -> dict[str, float]:
    """Brier score (error) by sex category.

    NIJ’s official scoring computed Brier separately for males and females.
    """

    return _metric_by_group(y_true, y_prob, sex, metric_fn=brier_score)


def fpr_by_race_at_threshold(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    race: ArrayLike,
    threshold: float = 0.5,
) -> dict[str, float]:
    """False positive rate by race at a fixed threshold.

    FPR is undefined when a subgroup has zero negatives (TN+FP == 0); in that
    case this function returns NaN for that subgroup.
    """

    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must be in [0, 1]")

    y_t, y_p, g = _prepare_group_inputs(y_true, y_prob, race)
    out: dict[str, float] = {}
    for label in sorted(set(g)):
        idx = [i for i, value in enumerate(g) if value == label]
        fp = tn = 0
        for i in idx:
            pred = 1 if y_p[i] >= threshold else 0
            if y_t[i] == 0 and pred == 1:
                fp += 1
            elif y_t[i] == 0 and pred == 0:
                tn += 1
        denom = fp + tn
        out[label] = (fp / denom) if denom > 0 else float("nan")
    return out


def nij_fp_term(fpr_black: float, fpr_white: float) -> float:
    """NIJ-style fairness penalty term (FP).

    FP = 1 - |FPR_black - FPR_white|.
    """

    if _is_nan(fpr_black) or _is_nan(fpr_white):
        return float("nan")
    value = 1.0 - abs(float(fpr_black) - float(fpr_white))
    return max(0.0, min(1.0, value))


def nij_fair_and_accurate(brier: float, fp_term: float) -> float:
    """NIJ-style "fair-and-accurate" index for a single subgroup.

    FairAcc = (1 - BS) * FP, where BS is the Brier *error*.
    """

    if _is_nan(brier) or _is_nan(fp_term):
        return float("nan")
    acc = 1.0 - float(brier)
    return max(0.0, min(1.0, acc)) * max(0.0, min(1.0, float(fp_term)))
