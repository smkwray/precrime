"""Core probabilistic classification metrics used across model pipelines."""

from __future__ import annotations

import math
import random
from typing import Callable


ArrayLike = list[float] | list[int] | tuple[float, ...] | tuple[int, ...]


def _is_nan(value: float) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _prepare_binary_inputs(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    sample_weight: ArrayLike | None = None,
) -> tuple[list[int], list[float], list[float] | None]:
    y_t = list(y_true)
    y_p = list(y_prob)

    if len(y_t) != len(y_p):
        raise ValueError("y_true and y_prob must have the same length")
    if not y_t:
        raise ValueError("y_true and y_prob cannot be empty")

    out_true: list[int] = []
    out_prob: list[float] = []
    for yt, yp in zip(y_t, y_p):
        if yt not in (0, 1):
            raise ValueError("y_true must contain only 0/1 labels")
        yp_f = float(yp)
        if yp_f < 0.0 or yp_f > 1.0:
            raise ValueError("y_prob must be in [0, 1]")
        out_true.append(int(yt))
        out_prob.append(yp_f)

    if sample_weight is None:
        out_weight = None
    else:
        out_weight = [float(w) for w in sample_weight]
        if len(out_weight) != len(out_true):
            raise ValueError("sample_weight must match y_true length")
        if any(w < 0.0 for w in out_weight):
            raise ValueError("sample_weight must be non-negative")

    return out_true, out_prob, out_weight


def _weighted_mean(values: list[float], weights: list[float] | None) -> float:
    if not values:
        return float("nan")
    if weights is None:
        return sum(values) / len(values)

    w_sum = sum(weights)
    if w_sum <= 0.0:
        return float("nan")
    return sum(v * w for v, w in zip(values, weights)) / w_sum


def brier_score(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    group: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
) -> float:
    del group
    y_t, y_p, w = _prepare_binary_inputs(y_true, y_prob, sample_weight=sample_weight)
    sq = [(p - y) ** 2 for y, p in zip(y_t, y_p)]
    return _weighted_mean(sq, w)


def log_loss(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    group: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
    eps: float = 1e-15,
) -> float:
    del group
    y_t, y_p, w = _prepare_binary_inputs(y_true, y_prob, sample_weight=sample_weight)
    losses: list[float] = []
    for y, p in zip(y_t, y_p):
        p_clip = min(max(p, eps), 1.0 - eps)
        losses.append(-(y * math.log(p_clip) + (1 - y) * math.log(1.0 - p_clip)))
    return _weighted_mean(losses, w)


def auroc(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    group: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
) -> float:
    del group
    y_t, y_p, w = _prepare_binary_inputs(y_true, y_prob, sample_weight=sample_weight)
    weights = w if w is not None else [1.0] * len(y_t)

    pos = [(p, wt) for y, p, wt in zip(y_t, y_p, weights) if y == 1]
    neg = [(p, wt) for y, p, wt in zip(y_t, y_p, weights) if y == 0]
    if not pos or not neg:
        return float("nan")

    wins = 0.0
    total = 0.0
    for p_pos, w_pos in pos:
        for p_neg, w_neg in neg:
            pair_w = w_pos * w_neg
            total += pair_w
            if p_pos > p_neg:
                wins += pair_w
            elif p_pos == p_neg:
                wins += 0.5 * pair_w

    if total <= 0.0:
        return float("nan")
    return wins / total


def roc_curve_points(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    group: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
) -> tuple[list[float], list[float], list[float]]:
    del group
    y_t, y_p, w = _prepare_binary_inputs(y_true, y_prob, sample_weight=sample_weight)
    weights = w if w is not None else [1.0] * len(y_t)

    thresholds = sorted(set(y_p), reverse=True)
    thresholds = [float("inf")] + thresholds + [float("-inf")]

    pos_total = sum(wt for y, wt in zip(y_t, weights) if y == 1)
    neg_total = sum(wt for y, wt in zip(y_t, weights) if y == 0)

    fpr: list[float] = []
    tpr: list[float] = []
    for thr in thresholds:
        tp = 0.0
        fp = 0.0
        for y, p, wt in zip(y_t, y_p, weights):
            pred = 1 if p >= thr else 0
            if pred == 1 and y == 1:
                tp += wt
            elif pred == 1 and y == 0:
                fp += wt
        tpr.append(tp / pos_total if pos_total > 0 else float("nan"))
        fpr.append(fp / neg_total if neg_total > 0 else float("nan"))

    return fpr, tpr, thresholds


def precision_recall_curve_points(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    group: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
) -> tuple[list[float], list[float], list[float]]:
    del group
    y_t, y_p, w = _prepare_binary_inputs(y_true, y_prob, sample_weight=sample_weight)
    weights = w if w is not None else [1.0] * len(y_t)

    thresholds = sorted(set(y_p), reverse=True)
    precision = [1.0]
    recall = [0.0]

    for thr in thresholds:
        tp = 0.0
        fp = 0.0
        fn = 0.0
        for y, p, wt in zip(y_t, y_p, weights):
            pred = 1 if p >= thr else 0
            if pred == 1 and y == 1:
                tp += wt
            elif pred == 1 and y == 0:
                fp += wt
            elif pred == 0 and y == 1:
                fn += wt
        precision.append(tp / (tp + fp) if (tp + fp) > 0 else 1.0)
        recall.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)

    return precision, recall, thresholds


def _trapz(x: list[float], y: list[float]) -> float:
    if len(x) != len(y):
        raise ValueError("x and y must be same length")
    if len(x) < 2:
        return 0.0
    area = 0.0
    for i in range(1, len(x)):
        area += 0.5 * (x[i] - x[i - 1]) * (y[i] + y[i - 1])
    return area


def auprc(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    group: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
) -> float:
    precision, recall, _ = precision_recall_curve_points(
        y_true,
        y_prob,
        group=group,
        sample_weight=sample_weight,
    )
    points = sorted(zip(recall, precision), key=lambda item: item[0])
    rec_sorted = [p[0] for p in points]
    pre_sorted = [p[1] for p in points]
    return _trapz(rec_sorted, pre_sorted)


def calibration_curve(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    group: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
    n_bins: int = 10,
) -> dict[str, list[float] | list[int] | list[bool]]:
    del group
    y_t, y_p, w = _prepare_binary_inputs(y_true, y_prob, sample_weight=sample_weight)
    weights = w if w is not None else [1.0] * len(y_t)

    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    edges = [i / n_bins for i in range(n_bins + 1)]
    pred_mean = [float("nan")] * n_bins
    true_rate = [float("nan")] * n_bins
    counts = [0] * n_bins
    empty = [True] * n_bins

    for b in range(n_bins):
        lo = edges[b]
        hi = edges[b + 1]
        bucket = [
            i
            for i, p in enumerate(y_p)
            if (p >= lo and (p < hi or (b == n_bins - 1 and p <= hi)))
        ]
        counts[b] = len(bucket)
        if not bucket:
            continue

        bucket_w = [weights[i] for i in bucket]
        bucket_y = [float(y_t[i]) for i in bucket]
        bucket_p = [y_p[i] for i in bucket]
        pred_mean[b] = _weighted_mean(bucket_p, bucket_w)
        true_rate[b] = _weighted_mean(bucket_y, bucket_w)
        empty[b] = False

    return {
        "bin_edges": edges,
        "pred_mean": pred_mean,
        "true_rate": true_rate,
        "counts": counts,
        "empty_bins": empty,
    }


def expected_calibration_error(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    group: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
    n_bins: int = 10,
) -> float:
    cal = calibration_curve(
        y_true,
        y_prob,
        group=group,
        sample_weight=sample_weight,
        n_bins=n_bins,
    )
    counts = cal["counts"]
    total = sum(counts)
    if total == 0:
        return float("nan")

    weighted_gap = 0.0
    for c, p_m, t_r in zip(counts, cal["pred_mean"], cal["true_rate"]):
        if c == 0 or _is_nan(float(p_m)) or _is_nan(float(t_r)):
            continue
        weighted_gap += (c / total) * abs(float(p_m) - float(t_r))
    return weighted_gap


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)

    vals = sorted(values)
    pos = q * (len(vals) - 1)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return vals[low]
    frac = pos - low
    return vals[low] * (1.0 - frac) + vals[high] * frac


def _sample_with_replacement(indices: list[int], size: int, rng: random.Random) -> list[int]:
    return [indices[rng.randrange(len(indices))] for _ in range(size)]


def bootstrap_ci(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    metric_fn: Callable[..., float],
    group: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
    n_bootstrap: int = 1000,
    alpha: float = 0.95,
    random_state: int | None = None,
    stratify_by_label: bool = True,
) -> dict[str, float | int | bool]:
    y_t, y_p, w = _prepare_binary_inputs(y_true, y_prob, sample_weight=sample_weight)

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    if group is not None and len(group) != len(y_t):
        raise ValueError("group must match y_true length")

    group_list = None if group is None else list(group)
    weight_list = None if w is None else list(w)

    rng = random.Random(random_state)
    n = len(y_t)
    boot_vals: list[float] = []

    pos_idx = [i for i, y in enumerate(y_t) if y == 1]
    neg_idx = [i for i, y in enumerate(y_t) if y == 0]
    can_stratify = stratify_by_label and bool(pos_idx) and bool(neg_idx)

    for _ in range(n_bootstrap):
        if can_stratify:
            idx = _sample_with_replacement(pos_idx, len(pos_idx), rng) + _sample_with_replacement(
                neg_idx,
                len(neg_idx),
                rng,
            )
        else:
            idx = [rng.randrange(n) for _ in range(n)]

        y_t_b = [y_t[i] for i in idx]
        y_p_b = [y_p[i] for i in idx]
        g_b = None if group_list is None else [group_list[i] for i in idx]
        w_b = None if weight_list is None else [weight_list[i] for i in idx]

        val = metric_fn(y_t_b, y_p_b, group=g_b, sample_weight=w_b)
        if not _is_nan(float(val)):
            boot_vals.append(float(val))

    estimate = float(metric_fn(y_t, y_p, group=group_list, sample_weight=weight_list))
    if not boot_vals:
        return {
            "estimate": estimate,
            "lower": float("nan"),
            "upper": float("nan"),
            "alpha": alpha,
            "n_bootstrap": n_bootstrap,
            "n_valid": 0,
            "stratified": can_stratify,
        }

    q = (1.0 - alpha) / 2.0
    return {
        "estimate": estimate,
        "lower": _quantile(boot_vals, q),
        "upper": _quantile(boot_vals, 1.0 - q),
        "alpha": alpha,
        "n_bootstrap": n_bootstrap,
        "n_valid": len(boot_vals),
        "stratified": can_stratify,
    }
