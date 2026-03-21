"""Generate fairness / subgroup audit reports for NIJ best models."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.fairness import (
    equalized_odds_gaps,
    fnr_gap,
    fpr_gap,
    predictive_parity_proxy,
    subgroup_auprc,
    subgroup_auroc,
    subgroup_brier,
    threshold_sweep,
    threshold_gap_summary,
)
from src.eval.metrics import auprc, auroc, brier_score, bootstrap_ci, expected_calibration_error, log_loss
from src.features.build_nij_dynamic import build_dynamic_datasets
from src.features.build_nij_static import build_static_datasets
from src.models.calibration import IsotonicCalibrator, PlattCalibrator
from src.models.xgb import train_xgb
from src.pipelines._split_utils import (
    RANDOM_SEED,
    evaluate_metrics,
    extract_target_column,
    format_float,
    prepare_feature_matrix,
    split_train_cal_test,
)


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


def _add_age_group(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "Age_at_Release" in out:
        out["age_group"] = out["Age_at_Release"].astype(str)
    else:
        out["age_group"] = "Unknown"
    out["age_group"] = out["age_group"].replace({"nan": "Unknown"}).fillna("Unknown")
    return out


def _drop_race_feature(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "Race" in out.columns:
        out = out.drop(columns=["Race"])
    return out


def _subgroup_table_lines(
    name: str,
    y_true: list[int],
    y_prob: list[float],
    groups: list[str],
    bootstrap_n: int = 0,
    bootstrap_alpha: float = 0.95,
) -> list[str]:
    counts: dict[str, int] = {}
    for g in groups:
        counts[g] = counts.get(g, 0) + 1

    sb = subgroup_brier(y_true, y_prob, group=groups)
    sa = subgroup_auroc(y_true, y_prob, group=groups)
    sp = subgroup_auprc(y_true, y_prob, group=groups)
    eo = equalized_odds_gaps(y_true, y_prob, group=groups, threshold=0.5)
    pp = predictive_parity_proxy(y_true, y_prob, group=groups, threshold=0.5)
    gaps = threshold_gap_summary(y_true, y_prob, group=groups)
    sweep = threshold_sweep(y_true, y_prob, group=groups)
    thresholds = [float(row["threshold"]) for row in sweep.get("overall", [])]
    try:
        thr_idx = thresholds.index(0.5)
    except ValueError:
        thr_idx = None

    lines = [
        f"### {name}",
        "",
        "| Group | N | Brier | Brier (CI) | AUROC | AUPRC |",
        "|---|---:|---:|---|---:|---:|",
    ]

    for label in sorted(sb.keys(), key=lambda k: (-counts.get(k, 0), str(k))):
        if bootstrap_n > 0 and counts.get(label, 0) >= 50:
            idx = [i for i, value in enumerate(groups) if value == label]
            y_t_sub = [y_true[i] for i in idx]
            y_p_sub = [y_prob[i] for i in idx]
            ci_brier = bootstrap_ci(
                y_t_sub,
                y_p_sub,
                metric_fn=brier_score,
                n_bootstrap=bootstrap_n,
                alpha=bootstrap_alpha,
                random_state=RANDOM_SEED,
            )
            ci_text = f"{format_float(float(ci_brier['lower']))}–{format_float(float(ci_brier['upper']))}"
        else:
            ci_text = ""

        lines.append(
            "| "
            + f"{label} | {counts.get(label, 0)} | {format_float(sb[label])} | {ci_text} | {format_float(sa[label])} | {format_float(sp[label])} |"
        )

    lines.extend(
        [
            "",
            f"- Equalized-odds gaps @0.5: `fpr_gap={format_float(float(eo['fpr_gap']))}`, `fnr_gap={format_float(float(eo['fnr_gap']))}`, `max={format_float(float(eo['eo_gap_max']))}`",
            f"- Predictive parity proxy @0.5: `ppv_gap={format_float(float(pp['ppv_gap']))}`",
            f"- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max={format_float(float(gaps['fpr_gap_max']))}`, `fpr_gap_mean={format_float(float(gaps['fpr_gap_mean']))}`, `fnr_gap_max={format_float(float(gaps['fnr_gap_max']))}`, `fnr_gap_mean={format_float(float(gaps['fnr_gap_mean']))}`",
            "",
        ]
    )

    if thr_idx is not None:
        lines.extend(
            [
                "#### Threshold 0.5 (error rates by group)",
                "",
                "| Group | N | FP | FN | FPR | FNR | PPV | Selection rate |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for label in sorted(set(groups), key=lambda k: (-counts.get(k, 0), str(k))):
            row = sweep.get(label, [])[thr_idx] if sweep.get(label) else {}
            fp = row.get("fp", float("nan"))
            fn = row.get("fn", float("nan"))
            lines.append(
                "| "
                + f"{label} | {counts.get(label, 0)} | {int(round(fp)) if not np.isnan(fp) else 'nan'} | {int(round(fn)) if not np.isnan(fn) else 'nan'} | "
                + f"{format_float(float(row.get('fpr', float('nan'))))} | {format_float(float(row.get('fnr', float('nan'))))} | {format_float(float(row.get('precision', float('nan'))))} | {format_float(float(row.get('selection_rate', float('nan'))))} |"
            )
        lines.append("")

    return lines


def _train_predict_xgb(
    x: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    params: dict[str, object],
    calibration: str,
) -> np.ndarray:
    model = train_xgb(x_train=x[fit_idx], y_train=y[fit_idx], params=params, seed=RANDOM_SEED)
    p_cal_raw = model.predict_proba(x[cal_idx])[:, 1]
    p_test_raw = model.predict_proba(x[test_idx])[:, 1]

    if calibration == "platt":
        return PlattCalibrator().fit(p_cal_raw, y[cal_idx]).predict(p_test_raw)
    if calibration == "isotonic":
        return IsotonicCalibrator().fit(p_cal_raw, y[cal_idx]).predict(p_test_raw)
    return np.clip(p_test_raw, 1e-6, 1.0 - 1e-6)


def write_fairness_report(best_models_path: Path | None = None, out_path: Path | None = None) -> Path:
    best = _load_best_models(best_models_path)
    out = out_path or (_reports_dir() / "fairness_report.md")

    all_sets = {
        "static": build_static_datasets(),
        "dynamic": build_dynamic_datasets(),
    }

    lines: list[str] = [
        "# NIJ Fairness / Subgroup Audit (Best XGBoost Models)",
        "",
        f"Generated: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}",
        "",
        "This report computes subgroup metrics on a fixed train/calibration/test split (seeded) for the best-per-horizon XGBoost configuration in `reports/xgb_best_models.json`.",
        "",
        "Two variants are reported:",
        "- **With race**: training features include `Race` (if present in the dataset).",
        "- **Without race**: training features exclude `Race`, but evaluation still reports subgroup metrics by `Race`.",
        "",
    ]

    for horizon in sorted(best.keys()):
        cfg = best[horizon]
        dataset_key = str(cfg["dataset"])
        calibration = str(cfg["calibration"])
        params = cfg.get("tuning", {}).get("best_params", {})

        ds = all_sets[dataset_key][horizon]
        ds = _add_age_group(ds)
        target_col = extract_target_column(ds)

        y = pd.to_numeric(ds[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
        feature_frame = ds.drop(columns=[target_col]).copy()
        fit_idx, cal_idx, test_idx = split_train_cal_test(y)
        test_frame = ds.iloc[test_idx].copy()
        y_list = y[test_idx].astype(int).tolist()

        eval_groups: dict[str, list[str]] = {}
        if "Race" in test_frame:
            eval_groups["Race"] = test_frame["Race"].fillna("Unknown").astype(str).tolist()
        if "Gender" in test_frame:
            eval_groups["Gender"] = test_frame["Gender"].fillna("Unknown").astype(str).tolist()
        eval_groups["Age Group"] = test_frame["age_group"].fillna("Unknown").astype(str).tolist()

        variants = [
            ("with race", feature_frame),
            ("without race", _drop_race_feature(feature_frame)),
        ]

        for variant_name, variant_features in variants:
            train_indices = np.concatenate([fit_idx, cal_idx])
            _, fit_cols = prepare_feature_matrix(variant_features.iloc[train_indices])
            x_df, _ = prepare_feature_matrix(variant_features, fit_columns=fit_cols)
            x = x_df.to_numpy(dtype=float)
            p_test = _train_predict_xgb(
                x=x,
                y=y,
                fit_idx=fit_idx,
                cal_idx=cal_idx,
                test_idx=test_idx,
                params=dict(params),
                calibration=calibration,
            )
            metrics = evaluate_metrics(y[test_idx], p_test)

            ci_brier = bootstrap_ci(
                y[test_idx].astype(int).tolist(),
                p_test.astype(float).tolist(),
                metric_fn=brier_score,
                n_bootstrap=int(os.getenv("PRECRIME_FAIRNESS_BOOTSTRAP", "500")),
                alpha=float(os.getenv("PRECRIME_FAIRNESS_ALPHA", "0.95")),
                random_state=RANDOM_SEED,
            )

            def _metric_fpr_gap(y_true, y_prob, group=None, sample_weight=None) -> float:
                return float(fpr_gap(y_true, y_prob, group=group, sample_weight=sample_weight, threshold=0.5))

            def _metric_fnr_gap(y_true, y_prob, group=None, sample_weight=None) -> float:
                return float(fnr_gap(y_true, y_prob, group=group, sample_weight=sample_weight, threshold=0.5))

            lines.extend(
                [
                    f"## {dataset_key.upper()} {horizon.upper()} — {variant_name} (calibration: {calibration})",
                    "",
                    "| Metric | Value |",
                    "|---|---:|",
                    f"| Brier | {format_float(metrics['brier'])} |",
                    f"| Brier (CI) | {format_float(float(ci_brier['lower']))}–{format_float(float(ci_brier['upper']))} |",
                    f"| AUROC | {format_float(metrics['auroc'])} |",
                    f"| AUPRC | {format_float(metrics['auprc'])} |",
                    f"| Log loss | {format_float(metrics['log_loss'])} |",
                    f"| ECE | {format_float(metrics['ece'])} |",
                    "",
                ]
            )

            p_list = p_test.astype(float).tolist()
            for group_name, group_values in eval_groups.items():
                gaps = threshold_gap_summary(y_list, p_list, group=group_values)
                ci_fpr_gap = bootstrap_ci(
                    y_list,
                    p_list,
                    metric_fn=_metric_fpr_gap,
                    group=group_values,
                    n_bootstrap=int(os.getenv("PRECRIME_FAIRNESS_BOOTSTRAP", "500")),
                    alpha=float(os.getenv("PRECRIME_FAIRNESS_ALPHA", "0.95")),
                    random_state=RANDOM_SEED,
                )
                ci_fnr_gap = bootstrap_ci(
                    y_list,
                    p_list,
                    metric_fn=_metric_fnr_gap,
                    group=group_values,
                    n_bootstrap=int(os.getenv("PRECRIME_FAIRNESS_BOOTSTRAP", "500")),
                    alpha=float(os.getenv("PRECRIME_FAIRNESS_ALPHA", "0.95")),
                    random_state=RANDOM_SEED,
                )
                lines.extend(
                    [
                        f"### {group_name} (gap summary)",
                        "",
                        f"- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max={format_float(float(gaps['fpr_gap_max']))}`, `fpr_gap_mean={format_float(float(gaps['fpr_gap_mean']))}`, `fnr_gap_max={format_float(float(gaps['fnr_gap_max']))}`, `fnr_gap_mean={format_float(float(gaps['fnr_gap_mean']))}`",
                        f"- Bootstrap gaps @0.5 (CI): `fpr_gap={format_float(float(ci_fpr_gap['estimate']))}` ({format_float(float(ci_fpr_gap['lower']))}–{format_float(float(ci_fpr_gap['upper']))}), `fnr_gap={format_float(float(ci_fnr_gap['estimate']))}` ({format_float(float(ci_fnr_gap['lower']))}–{format_float(float(ci_fnr_gap['upper']))})",
                        "",
                    ]
                )
                lines.extend(
                    _subgroup_table_lines(
                        group_name,
                        y_list,
                        p_list,
                        group_values,
                        bootstrap_n=int(os.getenv("PRECRIME_FAIRNESS_BOOTSTRAP_SUBGROUP", "0")),
                        bootstrap_alpha=float(os.getenv("PRECRIME_FAIRNESS_ALPHA", "0.95")),
                    )
                )

    out.write_text("\n".join(lines))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-models", type=Path, default=None, help="Path to xgb_best_models.json")
    parser.add_argument("--out", type=Path, default=None, help="Output path for markdown report")
    args = parser.parse_args()

    report = write_fairness_report(best_models_path=args.best_models, out_path=args.out)
    print(f"wrote fairness report: {report}")


if __name__ == "__main__":
    main()
