"""Fairness and subgroup evaluation helpers for binary risk models."""

from __future__ import annotations

import math

from .metrics import ArrayLike, auprc, auroc, brier_score, calibration_curve


def _is_nan(value: float) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _prepare_inputs(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    group: ArrayLike | None,
    sample_weight: ArrayLike | None = None,
) -> tuple[list[int], list[float], list[str], list[float] | None]:
    y_t = [int(v) for v in y_true]
    y_p = [float(v) for v in y_prob]
    if group is None:
        raise ValueError("group is required for subgroup fairness metrics")
    g = [str(x) for x in group]

    if len(y_t) != len(y_p):
        raise ValueError("y_true and y_prob must have the same length")
    if len(y_t) != len(g):
        raise ValueError("group must match y_true length")

    if sample_weight is None:
        w = None
    else:
        w = [float(x) for x in sample_weight]
        if len(w) != len(y_t):
            raise ValueError("sample_weight must match y_true length")
        if any(x < 0.0 for x in w):
            raise ValueError("sample_weight must be non-negative")

    return y_t, y_p, g, w


def _subset(values: list, idx: list[int]) -> list:
    return [values[i] for i in idx]


def subgroup_metric(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    group: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
    metric_fn=brier_score,
) -> dict[str, float]:
    y_t, y_p, g, w = _prepare_inputs(y_true, y_prob, group, sample_weight=sample_weight)

    out: dict[str, float] = {}
    for label in sorted(set(g)):
        idx = [i for i, value in enumerate(g) if value == label]
        y_t_sub = _subset(y_t, idx)
        y_p_sub = _subset(y_p, idx)
        w_sub = None if w is None else _subset(w, idx)
        out[label] = float(metric_fn(y_t_sub, y_p_sub, group=None, sample_weight=w_sub))
    return out


def subgroup_brier(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    group: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
) -> dict[str, float]:
    return subgroup_metric(
        y_true,
        y_prob,
        group=group,
        sample_weight=sample_weight,
        metric_fn=brier_score,
    )


def subgroup_auroc(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    group: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
) -> dict[str, float]:
    return subgroup_metric(
        y_true,
        y_prob,
        group=group,
        sample_weight=sample_weight,
        metric_fn=auroc,
    )


def subgroup_auprc(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    group: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
) -> dict[str, float]:
    return subgroup_metric(
        y_true,
        y_prob,
        group=group,
        sample_weight=sample_weight,
        metric_fn=auprc,
    )


def calibration_by_group(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    group: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
    n_bins: int = 10,
) -> dict[str, dict[str, list[float] | list[int] | list[bool]]]:
    y_t, y_p, g, w = _prepare_inputs(y_true, y_prob, group, sample_weight=sample_weight)

    out: dict[str, dict[str, list[float] | list[int] | list[bool]]] = {}
    for label in sorted(set(g)):
        idx = [i for i, value in enumerate(g) if value == label]
        y_t_sub = _subset(y_t, idx)
        y_p_sub = _subset(y_p, idx)
        w_sub = None if w is None else _subset(w, idx)
        out[label] = calibration_curve(
            y_t_sub,
            y_p_sub,
            group=None,
            sample_weight=w_sub,
            n_bins=n_bins,
        )

    return out


def _binary_rates(
    y_true: list[int],
    y_prob: list[float],
    threshold: float,
    sample_weight: list[float] | None = None,
) -> dict[str, float]:
    weights = sample_weight if sample_weight is not None else [1.0] * len(y_true)

    tp = fp = tn = fn = 0.0
    selected = 0.0
    total = sum(weights)

    for y, p, wt in zip(y_true, y_prob, weights):
        pred = 1 if p >= threshold else 0
        if pred == 1:
            selected += wt

        if pred == 1 and y == 1:
            tp += wt
        elif pred == 1 and y == 0:
            fp += wt
        elif pred == 0 and y == 0:
            tn += wt
        elif pred == 0 and y == 1:
            fn += wt

    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    fnr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
    tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "fpr": fpr,
        "fnr": fnr,
        "tpr": tpr,
        "precision": precision,
        "selection_rate": (selected / total) if total > 0 else float("nan"),
    }


def threshold_sweep(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    group: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
    thresholds: ArrayLike | None = None,
) -> dict[str, list[dict[str, float]]]:
    y_t = [int(v) for v in y_true]
    y_p = [float(v) for v in y_prob]
    if len(y_t) != len(y_p):
        raise ValueError("y_true and y_prob must have the same length")

    if sample_weight is None:
        w = None
    else:
        w = [float(x) for x in sample_weight]
        if len(w) != len(y_t):
            raise ValueError("sample_weight must match y_true length")

    thr = [i / 20 for i in range(21)] if thresholds is None else [float(t) for t in thresholds]

    out: dict[str, list[dict[str, float]]] = {"overall": []}
    for t in thr:
        row = _binary_rates(y_t, y_p, threshold=t, sample_weight=w)
        row["threshold"] = t
        out["overall"].append(row)

    if group is not None:
        g = [str(x) for x in group]
        if len(g) != len(y_t):
            raise ValueError("group must match y_true length")
        for label in sorted(set(g)):
            idx = [i for i, value in enumerate(g) if value == label]
            y_t_sub = _subset(y_t, idx)
            y_p_sub = _subset(y_p, idx)
            w_sub = None if w is None else _subset(w, idx)
            out[label] = []
            for t in thr:
                row = _binary_rates(y_t_sub, y_p_sub, threshold=t, sample_weight=w_sub)
                row["threshold"] = t
                out[label].append(row)

    return out


def threshold_gap_summary(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    group: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
    thresholds: ArrayLike | None = None,
) -> dict[str, float | list[float]]:
    if group is None:
        raise ValueError("group is required for threshold gap summary")

    sweep = threshold_sweep(
        y_true,
        y_prob,
        group=group,
        sample_weight=sample_weight,
        thresholds=thresholds,
    )

    group_labels = [key for key in sweep.keys() if key != "overall"]
    if not group_labels:
        return {
            "thresholds": [],
            "fpr_gap_per_threshold": [],
            "fnr_gap_per_threshold": [],
            "fpr_gap_max": float("nan"),
            "fpr_gap_mean": float("nan"),
            "fnr_gap_max": float("nan"),
            "fnr_gap_mean": float("nan"),
        }

    rows = len(sweep["overall"])
    fpr_gaps: list[float] = []
    fnr_gaps: list[float] = []
    grid: list[float] = []

    for i in range(rows):
        fprs = [sweep[label][i]["fpr"] for label in group_labels if not _is_nan(sweep[label][i]["fpr"])]
        fnrs = [sweep[label][i]["fnr"] for label in group_labels if not _is_nan(sweep[label][i]["fnr"])]
        grid.append(float(sweep["overall"][i]["threshold"]))
        fpr_gaps.append((max(fprs) - min(fprs)) if fprs else float("nan"))
        fnr_gaps.append((max(fnrs) - min(fnrs)) if fnrs else float("nan"))

    fpr_finite = [x for x in fpr_gaps if not _is_nan(x)]
    fnr_finite = [x for x in fnr_gaps if not _is_nan(x)]

    return {
        "thresholds": grid,
        "fpr_gap_per_threshold": fpr_gaps,
        "fnr_gap_per_threshold": fnr_gaps,
        "fpr_gap_max": max(fpr_finite) if fpr_finite else float("nan"),
        "fpr_gap_mean": (sum(fpr_finite) / len(fpr_finite)) if fpr_finite else float("nan"),
        "fnr_gap_max": max(fnr_finite) if fnr_finite else float("nan"),
        "fnr_gap_mean": (sum(fnr_finite) / len(fnr_finite)) if fnr_finite else float("nan"),
    }


def fpr_gap(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    group: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
    threshold: float = 0.5,
) -> float:
    y_t, y_p, g, w = _prepare_inputs(y_true, y_prob, group, sample_weight=sample_weight)

    fprs: list[float] = []
    for label in sorted(set(g)):
        idx = [i for i, value in enumerate(g) if value == label]
        y_t_sub = _subset(y_t, idx)
        y_p_sub = _subset(y_p, idx)
        w_sub = None if w is None else _subset(w, idx)
        fprs.append(_binary_rates(y_t_sub, y_p_sub, threshold, sample_weight=w_sub)["fpr"])

    finite = [v for v in fprs if not _is_nan(v)]
    if not finite:
        return float("nan")
    return max(finite) - min(finite)


def fnr_gap(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    group: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
    threshold: float = 0.5,
) -> float:
    y_t, y_p, g, w = _prepare_inputs(y_true, y_prob, group, sample_weight=sample_weight)

    fnrs: list[float] = []
    for label in sorted(set(g)):
        idx = [i for i, value in enumerate(g) if value == label]
        y_t_sub = _subset(y_t, idx)
        y_p_sub = _subset(y_p, idx)
        w_sub = None if w is None else _subset(w, idx)
        fnrs.append(_binary_rates(y_t_sub, y_p_sub, threshold, sample_weight=w_sub)["fnr"])

    finite = [v for v in fnrs if not _is_nan(v)]
    if not finite:
        return float("nan")
    return max(finite) - min(finite)


def equalized_odds_gaps(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    group: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    gap_fpr = fpr_gap(
        y_true,
        y_prob,
        group=group,
        sample_weight=sample_weight,
        threshold=threshold,
    )
    gap_fnr = fnr_gap(
        y_true,
        y_prob,
        group=group,
        sample_weight=sample_weight,
        threshold=threshold,
    )

    finite = [v for v in (gap_fpr, gap_fnr) if not _is_nan(v)]
    eo_max = max(finite) if finite else float("nan")
    return {
        "fpr_gap": gap_fpr,
        "fnr_gap": gap_fnr,
        "eo_gap_max": eo_max,
    }


def predictive_parity_proxy(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    group: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
    threshold: float = 0.5,
) -> dict[str, float | dict[str, float]]:
    y_t, y_p, g, w = _prepare_inputs(y_true, y_prob, group, sample_weight=sample_weight)

    ppv_by_group: dict[str, float] = {}
    for label in sorted(set(g)):
        idx = [i for i, value in enumerate(g) if value == label]
        y_t_sub = _subset(y_t, idx)
        y_p_sub = _subset(y_p, idx)
        w_sub = None if w is None else _subset(w, idx)
        ppv_by_group[label] = _binary_rates(
            y_t_sub,
            y_p_sub,
            threshold,
            sample_weight=w_sub,
        )["precision"]

    finite = [v for v in ppv_by_group.values() if not _is_nan(v)]
    gap = (max(finite) - min(finite)) if finite else float("nan")
    return {
        "ppv_by_group": ppv_by_group,
        "ppv_gap": gap,
    }
