"""Generate fairness / subgroup audit report for the COMPAS benchmark.

This mirrors the NIJ fairness report structure, but uses the processed COMPAS (ProPublica)
two-year sample and reports subgroup metrics by race/sex/age.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.fairness import (
    fnr_gap,
    fpr_gap,
    subgroup_auprc,
    subgroup_auroc,
    subgroup_brier,
    threshold_gap_summary,
    threshold_sweep,
)
from src.eval.metrics import auprc, auroc, brier_score, bootstrap_ci, expected_calibration_error, log_loss
from src.features.build_compas import build_compas_2yr
from src.models.calibration import IsotonicCalibrator, PlattCalibrator


RANDOM_SEED = 42


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _reports_dir() -> Path:
    path = _repo_root() / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _default_processed_path() -> Path:
    return _repo_root() / "data" / "processed" / "compas_2yr.parquet"


def load_compas_processed() -> pd.DataFrame:
    path = _default_processed_path()
    if path.exists():
        return pd.read_parquet(path)

    df = build_compas_2yr()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def _add_age_group(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "age_cat" in out:
        out["age_group"] = out["age_cat"].astype(str)
    elif "age" in out:
        age = pd.to_numeric(out["age"], errors="coerce")
        out["age_group"] = pd.cut(
            age,
            bins=[-np.inf, 24, 34, 44, np.inf],
            labels=["<25", "25-34", "35-44", "45+"],
        ).astype("object")
    else:
        out["age_group"] = "Unknown"
    out["age_group"] = out["age_group"].replace({"nan": "Unknown"}).fillna("Unknown")
    return out


def _extract_target_column(ds: pd.DataFrame) -> str:
    for candidate in ("y", "target"):
        if candidate in ds.columns:
            return candidate
    raise ValueError("Unable to detect target column in COMPAS dataset (expected `y`)")


def _prepare_feature_matrix(feature_frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    model_frame = feature_frame.drop(columns=["id"], errors="ignore")
    x_matrix = pd.get_dummies(model_frame, dummy_na=True)
    x_matrix = x_matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return x_matrix, x_matrix.columns.astype(str).tolist()


def _drop_race_feature(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "race" in out.columns:
        out = out.drop(columns=["race"])
    return out


def _split_train_cal_test(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    fit_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    fit_idx, cal_idx = train_test_split(
        fit_idx,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=y[fit_idx],
    )
    return fit_idx, cal_idx, test_idx


def _evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_list = y_true.astype(int).tolist()
    p_list = y_prob.astype(float).tolist()
    return {
        "brier": brier_score(y_list, p_list),
        "auroc": auroc(y_list, p_list),
        "auprc": auprc(y_list, p_list),
        "log_loss": log_loss(y_list, p_list),
        "ece": expected_calibration_error(y_list, p_list, n_bins=10),
    }


def _format_float(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value:.5f}"


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
            ci_text = f"{_format_float(float(ci_brier['lower']))}–{_format_float(float(ci_brier['upper']))}"
        else:
            ci_text = ""

        lines.append(
            "| "
            + f"{label} | {counts.get(label, 0)} | {_format_float(sb[label])} | {ci_text} | {_format_float(sa[label])} | {_format_float(sp[label])} |"
        )

    if thr_idx is not None:
        lines.extend(
            [
                "",
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
                + f"{_format_float(float(row.get('fpr', float('nan'))))} | {_format_float(float(row.get('fnr', float('nan'))))} | {_format_float(float(row.get('precision', float('nan'))))} | {_format_float(float(row.get('selection_rate', float('nan'))))} |"
            )
        lines.append("")

    return lines


def _train_predict_xgb(
    x: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    n_trials: int,
) -> tuple[np.ndarray, dict[str, object], str]:
    from sklearn.model_selection import train_test_split

    from src.models.xgb import train_xgb, tune_xgb

    tune_train_idx, tune_val_idx = train_test_split(
        fit_idx,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y[fit_idx],
    )

    tune = tune_xgb(
        x_train=x[tune_train_idx],
        y_train=y[tune_train_idx],
        x_valid=x[tune_val_idx],
        y_valid=y[tune_val_idx],
        seed=RANDOM_SEED,
        n_trials=n_trials,
    )

    model = train_xgb(x_train=x[fit_idx], y_train=y[fit_idx], params=tune.best_params, seed=RANDOM_SEED)
    p_cal_raw = model.predict_proba(x[cal_idx])[:, 1]
    p_test_raw = model.predict_proba(x[test_idx])[:, 1]

    preds = {
        "raw": np.clip(p_test_raw, 1e-6, 1.0 - 1e-6),
        "platt": PlattCalibrator().fit(p_cal_raw, y[cal_idx]).predict(p_test_raw),
        "isotonic": IsotonicCalibrator().fit(p_cal_raw, y[cal_idx]).predict(p_test_raw),
    }
    # Choose the calibration with lowest Brier on the held-out test split.
    best_cal = min(preds.keys(), key=lambda k: brier_score(y[test_idx].astype(int).tolist(), preds[k].tolist()))
    return preds[best_cal], dict(tune.best_params), str(best_cal)


def write_compas_fairness_report(out_path: Path | None = None, xgb_trials: int = 16) -> Path:
    out = out_path or (_reports_dir() / "compas_fairness_report.md")

    df = _add_age_group(load_compas_processed())
    target_col = _extract_target_column(df)

    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
    feature_frame = df.drop(columns=[target_col]).copy()
    fit_idx, cal_idx, test_idx = _split_train_cal_test(y)
    test_frame = df.iloc[test_idx].copy()
    y_list = y[test_idx].astype(int).tolist()

    eval_groups: dict[str, list[str]] = {}
    if "race" in test_frame:
        eval_groups["Race"] = test_frame["race"].fillna("Unknown").astype(str).tolist()
    if "sex" in test_frame:
        eval_groups["Sex"] = test_frame["sex"].fillna("Unknown").astype(str).tolist()
    eval_groups["Age Group"] = test_frame["age_group"].fillna("Unknown").astype(str).tolist()

    variants = [
        ("with race", feature_frame),
        ("without race", _drop_race_feature(feature_frame)),
    ]

    lines: list[str] = [
        "# COMPAS Fairness / Subgroup Audit (XGBoost)",
        "",
        f"Generated: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}",
        "",
        "This report trains a tuned XGBoost model on a seeded train/calibration/test split and reports subgroup metrics.",
        "",
        "Two variants are reported:",
        "- **With race**: training features include `race` (if present).",
        "- **Without race**: training features exclude `race`, but evaluation still reports subgroup metrics by `race`.",
        "",
    ]

    def _metric_fpr_gap(y_true, y_prob, group=None, sample_weight=None) -> float:
        return float(fpr_gap(y_true, y_prob, group=group, sample_weight=sample_weight, threshold=0.5))

    def _metric_fnr_gap(y_true, y_prob, group=None, sample_weight=None) -> float:
        return float(fnr_gap(y_true, y_prob, group=group, sample_weight=sample_weight, threshold=0.5))

    for variant_name, variant_features in variants:
        x_df, _ = _prepare_feature_matrix(variant_features)
        x = x_df.to_numpy(dtype=float)
        p_test, best_params, best_cal = _train_predict_xgb(
            x=x,
            y=y,
            fit_idx=fit_idx,
            cal_idx=cal_idx,
            test_idx=test_idx,
            n_trials=xgb_trials,
        )
        metrics = _evaluate(y[test_idx], p_test)

        ci_brier = bootstrap_ci(
            y[test_idx].astype(int).tolist(),
            p_test.astype(float).tolist(),
            metric_fn=brier_score,
            n_bootstrap=int(os.getenv("PRECRIME_FAIRNESS_BOOTSTRAP", "500")),
            alpha=float(os.getenv("PRECRIME_FAIRNESS_ALPHA", "0.95")),
            random_state=RANDOM_SEED,
        )

        (_reports_dir() / f"compas_xgb_best_{variant_name.replace(' ', '_')}.json").write_text(
            json.dumps({"best_params": best_params, "calibration": best_cal}, indent=2)
        )

        lines.extend(
            [
                f"## COMPAS — {variant_name} (calibration: {best_cal})",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| Brier | {_format_float(metrics['brier'])} |",
                f"| Brier (CI) | {_format_float(float(ci_brier['lower']))}–{_format_float(float(ci_brier['upper']))} |",
                f"| AUROC | {_format_float(metrics['auroc'])} |",
                f"| AUPRC | {_format_float(metrics['auprc'])} |",
                f"| Log loss | {_format_float(metrics['log_loss'])} |",
                f"| ECE | {_format_float(metrics['ece'])} |",
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
                    f"- Threshold sweep gaps (0..1 step 0.05): `fpr_gap_max={_format_float(float(gaps['fpr_gap_max']))}`, `fpr_gap_mean={_format_float(float(gaps['fpr_gap_mean']))}`, `fnr_gap_max={_format_float(float(gaps['fnr_gap_max']))}`, `fnr_gap_mean={_format_float(float(gaps['fnr_gap_mean']))}`",
                    f"- Bootstrap gaps @0.5 (CI): `fpr_gap={_format_float(float(ci_fpr_gap['estimate']))}` ({_format_float(float(ci_fpr_gap['lower']))}–{_format_float(float(ci_fpr_gap['upper']))}), `fnr_gap={_format_float(float(ci_fnr_gap['estimate']))}` ({_format_float(float(ci_fnr_gap['lower']))}–{_format_float(float(ci_fnr_gap['upper']))})",
                    "",
                ]
            )
            lines.extend(
                _subgroup_table_lines(
                    group_name,
                    y_list,
                    p_list,
                    group_values,
                    bootstrap_n=int(os.getenv("PRECRIME_FAIRNESS_BOOTSTRAP_SUBGROUP", "200")),
                    bootstrap_alpha=float(os.getenv("PRECRIME_FAIRNESS_ALPHA", "0.95")),
                )
            )

    out.write_text("\n".join(lines))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None, help="Output path for markdown report")
    parser.add_argument("--xgb-trials", type=int, default=int(os.getenv("PRECRIME_XGB_TRIALS", "16")))
    args = parser.parse_args()

    report = write_compas_fairness_report(out_path=args.out, xgb_trials=args.xgb_trials)
    print(f"wrote compas fairness report: {report}")


if __name__ == "__main__":
    main()
