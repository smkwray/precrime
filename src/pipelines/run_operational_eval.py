"""Generate an operational-style evaluation report for NIJ best models.

This complements the fairness report by showing error rates (FP/FN, FPR/FNR/PPV)
under a few thresholding policies that are easier to interpret than a single
fixed cutoff.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.metrics import auprc, auroc, brier_score, expected_calibration_error, log_loss
from src.features.build_nij_dynamic import build_dynamic_datasets
from src.features.build_nij_static import build_static_datasets
from src.models.calibration import IsotonicCalibrator, PlattCalibrator
from src.models.xgb import train_xgb
from src.pipelines._split_utils import (
    extract_target_column,
    format_float,
    prepare_feature_matrix,
    split_train_cal_select_test,
)


RANDOM_SEED_DEFAULT = 42


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _reports_dir() -> Path:
    path = _repo_root() / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_best_models(path: Path | None = None) -> dict[str, dict[str, object]]:
    resolved = path or (_reports_dir() / "xgb_best_models.json")
    if not resolved.exists():
        raise FileNotFoundError(f"Missing {resolved}. Run `python -m src.pipelines.run_nij --task xgb` first.")
    return json.loads(resolved.read_text())


def _drop_race_feature(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "Race" in out.columns:
        out = out.drop(columns=["Race"])
    return out


def _train_predict_xgb(
    x: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    cal_idx: np.ndarray,
    eval_idx: np.ndarray,
    params: dict[str, object],
    calibration: str,
    seed: int,
) -> np.ndarray:
    """Train XGBoost, calibrate, and return predictions on eval_idx."""
    model = train_xgb(x_train=x[fit_idx], y_train=y[fit_idx], params=params, seed=seed)
    p_cal_raw = model.predict_proba(x[cal_idx])[:, 1]
    p_eval_raw = model.predict_proba(x[eval_idx])[:, 1]

    if calibration == "platt":
        return PlattCalibrator().fit(p_cal_raw, y[cal_idx]).predict(p_eval_raw)
    if calibration == "isotonic":
        return IsotonicCalibrator().fit(p_cal_raw, y[cal_idx]).predict(p_eval_raw)
    return np.clip(p_eval_raw, 1e-6, 1.0 - 1e-6)


def _binary_rates(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y = y_true.astype(int)
    pred = (y_prob >= threshold).astype(int)
    tp = float(np.sum((pred == 1) & (y == 1)))
    fp = float(np.sum((pred == 1) & (y == 0)))
    tn = float(np.sum((pred == 0) & (y == 0)))
    fn = float(np.sum((pred == 0) & (y == 1)))

    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    fnr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
    tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    sel = float(np.mean(pred))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "fpr": fpr, "fnr": fnr, "tpr": tpr, "ppv": ppv, "selection": sel}


def _threshold_top_k(y_prob: np.ndarray, k: float) -> float:
    if not (0.0 < k <= 1.0):
        raise ValueError("k must be in (0, 1]")
    n = len(y_prob)
    m = int(np.ceil(k * n))
    if m <= 0:
        return 1.0
    if m >= n:
        return 0.0
    # kth largest
    return float(np.partition(y_prob, n - m)[n - m])


def _threshold_target_fpr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_fpr: float,
    grid: int = 1001,
    min_selection: float = 1e-6,
) -> float:
    if not (0.0 <= target_fpr <= 1.0):
        raise ValueError("target_fpr must be in [0, 1]")
    thresholds = np.linspace(0.0, 1.0, grid)
    best_t: float | None = None
    best_sel: float = -1.0
    for t in thresholds:
        rates = _binary_rates(y_true, y_prob, float(t))
        if (
            not np.isnan(rates["fpr"])
            and rates["fpr"] <= target_fpr
            and not np.isnan(rates["selection"])
            and rates["selection"] > min_selection
        ):
            # Prefer the threshold that keeps FPR under the cap while maximizing selection.
            # (Operationally: maximize capture subject to an FP-rate constraint.)
            if float(rates["selection"]) > best_sel:
                best_sel = float(rates["selection"])
                best_t = float(t)
    return 1.0 if best_t is None else float(best_t)


def _load_top_features_from_plot(path: Path, top_n: int = 5) -> list[tuple[str, float]]:
    obj = json.loads(path.read_text())
    # Plot spec format: {x:[feature], y:[value]}
    if isinstance(obj, dict) and "x" in obj and "y" in obj:
        pairs = list(zip(obj["x"], obj["y"]))
        pairs_sorted = sorted(pairs, key=lambda t: float(t[1]), reverse=True)
        return [(str(f), float(v)) for f, v in pairs_sorted[:top_n]]
    # List-of-dicts format: [{"feature":..., "mean_abs_shap":...}, ...]
    if isinstance(obj, list) and obj and isinstance(obj[0], dict) and "feature" in obj[0]:
        value_key = "mean_abs_shap" if "mean_abs_shap" in obj[0] else "importance"
        rows_sorted = sorted(obj, key=lambda r: float(r.get(value_key, 0.0)), reverse=True)
        return [(str(r["feature"]), float(r.get(value_key, 0.0))) for r in rows_sorted[:top_n]]
    return []


def write_operational_report(best_models_path: Path | None = None, out_path: Path | None = None, seed: int = RANDOM_SEED_DEFAULT) -> Path:
    best = _load_best_models(best_models_path)
    out = out_path or (_reports_dir() / "operational_eval.md")

    all_sets = {
        "static": build_static_datasets(),
        "dynamic": build_dynamic_datasets(),
    }

    lines: list[str] = [
        "# Operational-Style Evaluation (NIJ Best Models)",
        "",
        f"Generated: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}",
        "",
        "This report summarizes prediction quality and subgroup error rates under several thresholding policies.",
        "It is meant to be easier to interpret than a single fixed cutoff; it does **not** choose an operational threshold for you.",
        "",
        '## What "prediction success" means (plain language)',
        "- **AUROC**: how well the model ranks higher-risk people above lower-risk people (0.5 = random; 1.0 = perfect ranking).",
        "- **Brier score**: average squared error of predicted probabilities (lower = better probability accuracy).",
        "- **FPR/FNR/PPV** below depend on a thresholding policy; they are not intrinsic model properties.",
        "",
        "## Thresholding policies shown",
        "- `t=0.5`: fixed threshold at 0.5 (mainly illustrative).",
        "- `top10%`: flag the top 10% highest predicted risks (threshold derived from the held-out selection split).",
        "- `top20%`: flag the top 20% highest predicted risks (threshold derived from the held-out selection split).",
        "- `FPR<=0.06`: choose the **highest** threshold that achieves overall FPR ≤ 6% on the held-out selection split (grid search).",
        "",
        f"Random seed for split/train/calibration: `{seed}`.",
        "",
    ]

    for horizon in sorted(best.keys()):
        cfg = best[horizon]
        dataset_key = str(cfg["dataset"])
        calibration = str(cfg["calibration"])
        params = cfg.get("tuning", {}).get("best_params", {})

        ds = all_sets[dataset_key][horizon].copy()
        target_col = extract_target_column(ds)
        y = pd.to_numeric(ds[target_col], errors="coerce").fillna(0).astype(int).to_numpy()

        # 4-way split: thresholds derived from select split, evaluated on test split
        fit_idx, cal_idx, select_idx, test_idx = split_train_cal_select_test(y, seed=seed)
        test_frame = ds.iloc[test_idx].copy()
        y_test = y[test_idx].astype(int)
        y_select = y[select_idx].astype(int)

        feature_frame = ds.drop(columns=[target_col]).copy()

        variants = [
            ("with race", feature_frame),
            ("without race", _drop_race_feature(feature_frame)),
        ]

        for variant_name, variant_features in variants:
            x_df, _ = prepare_feature_matrix(variant_features)
            x = x_df.to_numpy(dtype=float)

            # Get predictions on both select and test splits
            p_select = _train_predict_xgb(
                x=x, y=y,
                fit_idx=fit_idx, cal_idx=cal_idx, eval_idx=select_idx,
                params=dict(params), calibration=calibration, seed=seed,
            )
            p_test = _train_predict_xgb(
                x=x, y=y,
                fit_idx=fit_idx, cal_idx=cal_idx, eval_idx=test_idx,
                params=dict(params), calibration=calibration, seed=seed,
            )

            metrics = {
                "brier": brier_score(y_test.tolist(), p_test.astype(float).tolist()),
                "auroc": auroc(y_test.tolist(), p_test.astype(float).tolist()),
                "auprc": auprc(y_test.tolist(), p_test.astype(float).tolist()),
                "log_loss": log_loss(y_test.tolist(), p_test.astype(float).tolist()),
                "ece": expected_calibration_error(y_test.tolist(), p_test.astype(float).tolist(), n_bins=10),
            }

            lines.extend(
                [
                    f"## {dataset_key.upper()} {horizon.upper()} — {variant_name} (calibration: {calibration})",
                    "",
                    "| Metric | Value |",
                    "|---|---:|",
                    f"| Brier | {format_float(float(metrics['brier']))} |",
                    f"| AUROC | {format_float(float(metrics['auroc']))} |",
                    f"| AUPRC | {format_float(float(metrics['auprc']))} |",
                    f"| Log loss | {format_float(float(metrics['log_loss']))} |",
                    f"| ECE | {format_float(float(metrics['ece']))} |",
                    "",
                ]
            )

            # Threshold policies — derived from select split, applied to test split
            thresholds: list[tuple[str, float]] = [
                ("t=0.5", 0.5),
                ("top10%", _threshold_top_k(p_select, 0.10)),
                ("top20%", _threshold_top_k(p_select, 0.20)),
                ("FPR<=0.06", _threshold_target_fpr(y_select, p_select, 0.06)),
            ]

            # Race breakdown if present
            if "Race" in test_frame.columns:
                groups = test_frame["Race"].fillna("Unknown").astype(str).tolist()
                labels = sorted(set(groups), key=str)
                lines.append("### Race (error rates by policy)")
                lines.append("")
                for policy_name, thr in thresholds:
                    overall = _binary_rates(y_test, p_test, thr)
                    lines.extend(
                        [
                            f"#### {policy_name} (threshold={thr:.5f}; overall selection={overall['selection']:.5f})",
                            "",
                            "| Group | N | FP | FN | FPR | FNR | PPV | TPR | Selection |",
                            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                        ]
                    )
                    for label in labels:
                        idx = np.array([i for i, g in enumerate(groups) if g == label], dtype=int)
                        y_sub = y_test[idx]
                        p_sub = p_test[idx]
                        r = _binary_rates(y_sub, p_sub, thr)
                        lines.append(
                            "| "
                            + f"{label} | {len(idx)} | {int(r['fp'])} | {int(r['fn'])} | {format_float(r['fpr'])} | {format_float(r['fnr'])} | {format_float(r['ppv'])} | {format_float(r['tpr'])} | {format_float(r['selection'])} |"
                        )
                    lines.append("")

            # Top features
            plot_dir = _reports_dir() / "plots"
            shap_path = plot_dir / f"xgb_shap_{dataset_key}_{horizon}.json"
            imp_path = plot_dir / f"xgb_importance_{dataset_key}_{horizon}.json"
            lines.append("### Top factors (model explanation snapshots)")
            lines.append("")
            if shap_path.exists():
                top = _load_top_features_from_plot(shap_path, top_n=5)
                if top:
                    lines.append("- SHAP (top 5): " + ", ".join([str(f) for f, _ in top[:5]]))
                else:
                    lines.append("- SHAP: (missing/empty)")
            else:
                lines.append("- SHAP: (missing)")
            if imp_path.exists():
                top = _load_top_features_from_plot(imp_path, top_n=5)
                if top:
                    lines.append("- XGB importance (top 5): " + ", ".join([str(f) for f, _ in top[:5]]))
                else:
                    lines.append("- XGB importance: (missing/empty)")
            else:
                lines.append("- XGB importance: (missing)")
            lines.append("")

    out.write_text("\n".join(lines))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-models", type=Path, default=None, help="Path to xgb_best_models.json")
    parser.add_argument("--out", type=Path, default=None, help="Output path for markdown report")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED_DEFAULT, help="Random seed for split/train/calibration")
    args = parser.parse_args()

    report = write_operational_report(best_models_path=args.best_models, out_path=args.out, seed=int(args.seed))
    print(f"wrote operational report: {report}")


if __name__ == "__main__":
    main()
