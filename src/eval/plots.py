"""Plot builders for model evaluation and fairness reporting.

The module returns serializable specs that are HTML-friendly and easy to render
with Plotly/Altair in downstream reporting.
"""

from __future__ import annotations

from .fairness import equalized_odds_gaps, predictive_parity_proxy
from .metrics import calibration_curve, precision_recall_curve_points, roc_curve_points


def plot_calibration(
    y_true,
    y_prob,
    group=None,
    sample_weight=None,
    n_bins: int = 10,
    title: str = "Calibration Curve",
):
    overall = calibration_curve(
        y_true,
        y_prob,
        group=None,
        sample_weight=sample_weight,
        n_bins=n_bins,
    )

    traces = [
        {
            "type": "line",
            "name": "Overall",
            "x": overall["pred_mean"],
            "y": overall["true_rate"],
        },
        {
            "type": "line",
            "name": "Perfect",
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
            "style": "dashed",
        },
    ]

    if group is not None:
        y_t = list(y_true)
        y_p = list(y_prob)
        g = [str(x) for x in group]
        w = None if sample_weight is None else [float(x) for x in sample_weight]
        for label in sorted(set(g)):
            idx = [i for i, value in enumerate(g) if value == label]
            cal = calibration_curve(
                [y_t[i] for i in idx],
                [y_p[i] for i in idx],
                sample_weight=None if w is None else [w[i] for i in idx],
                n_bins=n_bins,
            )
            traces.append(
                {
                    "type": "line",
                    "name": f"Group {label}",
                    "x": cal["pred_mean"],
                    "y": cal["true_rate"],
                }
            )

    return {
        "kind": "figure",
        "title": title,
        "xaxis": "Mean Predicted Probability",
        "yaxis": "Observed Positive Rate",
        "traces": traces,
    }


def plot_roc_curve(y_true, y_prob, group=None, sample_weight=None, title: str = "ROC Curve"):
    fpr, tpr, _ = roc_curve_points(
        y_true,
        y_prob,
        group=group,
        sample_weight=sample_weight,
    )
    return {
        "kind": "figure",
        "title": title,
        "xaxis": "False Positive Rate",
        "yaxis": "True Positive Rate",
        "traces": [
            {"type": "line", "name": "ROC", "x": fpr, "y": tpr},
            {
                "type": "line",
                "name": "Chance",
                "x": [0.0, 1.0],
                "y": [0.0, 1.0],
                "style": "dashed",
            },
        ],
    }


def plot_pr_curve(y_true, y_prob, group=None, sample_weight=None, title: str = "Precision-Recall Curve"):
    precision, recall, _ = precision_recall_curve_points(
        y_true,
        y_prob,
        group=group,
        sample_weight=sample_weight,
    )
    return {
        "kind": "figure",
        "title": title,
        "xaxis": "Recall",
        "yaxis": "Precision",
        "traces": [{"type": "line", "name": "PR", "x": recall, "y": precision}],
    }


def plot_fairness_bars(
    y_true,
    y_prob,
    group,
    sample_weight=None,
    threshold: float = 0.5,
    title: str = "Fairness Gap Comparison",
):
    eo = equalized_odds_gaps(
        y_true,
        y_prob,
        group=group,
        sample_weight=sample_weight,
        threshold=threshold,
    )
    pp = predictive_parity_proxy(
        y_true,
        y_prob,
        group=group,
        sample_weight=sample_weight,
        threshold=threshold,
    )

    labels = ["FPR gap", "FNR gap", "EO max gap", "PPV gap"]
    values = [eo["fpr_gap"], eo["fnr_gap"], eo["eo_gap_max"], pp["ppv_gap"]]
    return {
        "kind": "bar",
        "title": title,
        "xaxis": "Metric",
        "yaxis": "Gap",
        "x": labels,
        "y": values,
    }
