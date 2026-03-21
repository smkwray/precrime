"""Generate NIJ-style scoring report for the current best NIJ models.

This reports sex-specific Brier and the NIJ "fair-and-accurate" index component:

  FP = 1 - |FPR_black@0.5 - FPR_white@0.5|
  FairAcc = (1 - BS) * FP

These numbers are computed on this project's seeded held-out split and are not
directly comparable to NIJ's official leaderboard (which used a separate test set).
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.metrics import auprc, auroc, brier_score, expected_calibration_error, log_loss
from src.eval.nij_scoring import fpr_by_race_at_threshold, nij_fair_and_accurate, nij_fp_term
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


def _no_race_columns(frame: pd.DataFrame) -> pd.DataFrame:
    cols = [
        c
        for c in frame.columns
        if c != "Race" and not str(c).startswith("Race_") and "race" not in str(c).lower()
    ]
    return frame[cols].copy()


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


def write_nij_scoring_report(best_models_path: Path | None = None, out_path: Path | None = None) -> Path:
    best = _load_best_models(best_models_path)
    out = out_path or (_reports_dir() / "nij_scoring.md")

    all_sets = {
        "static": build_static_datasets(),
        "dynamic": build_dynamic_datasets(),
    }

    lines: list[str] = [
        "# NIJ-Style Scoring (Held-out Split; Not Official Leaderboard Scores)",
        "",
        f"Generated: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}",
        "",
        "This report computes NIJ-style scoring terms on this project's seeded train/calibration/test split (seed 42).",
        "NIJ's official Challenge scoring used a separate held-out test set, so these values are **not directly comparable** to the NIJ leaderboard.",
        "",
        "Definitions (NIJ-style):",
        "- `BS`: Brier score **error** (lower is better).",
        "- `FPR_black@0.5` / `FPR_white@0.5`: false positive rates at threshold 0.5 within a sex subgroup.",
        "- `FP = 1 - |FPR_black@0.5 - FPR_white@0.5|` (fairness penalty term).",
        "- `FairAcc = (1 - BS) * FP`.",
        "",
    ]

    # Summary table first (avg across sexes).
    summary_rows: list[list[str]] = []

    for horizon in sorted(best.keys()):
        cfg = best[horizon]
        dataset_key = str(cfg["dataset"])
        calibration = str(cfg["calibration"])
        params = dict(cfg.get("tuning", {}).get("best_params", {}))

        ds = _add_age_group(all_sets[dataset_key][horizon])
        target_col = extract_target_column(ds)
        y = pd.to_numeric(ds[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
        feature_frame = ds.drop(columns=[target_col]).copy()
        fit_idx, cal_idx, test_idx = split_train_cal_test(y)

        test_frame = ds.iloc[test_idx].copy()
        y_test = y[test_idx].astype(int).tolist()
        sex_vals = test_frame["Gender"].fillna("Unknown").astype(str).tolist() if "Gender" in test_frame else ["Unknown"] * len(test_frame)
        race_vals = test_frame["Race"].fillna("Unknown").astype(str).tolist() if "Race" in test_frame else ["Unknown"] * len(test_frame)

        variants = [
            ("with_race", feature_frame),
            ("without_race", _no_race_columns(feature_frame)),
        ]

        lines.extend(
            [
                f"## {horizon.upper()} ({dataset_key}; calibration={calibration})",
                "",
            ]
        )

        for variant_name, variant_features in variants:
            x_df, _ = prepare_feature_matrix(variant_features)
            x = x_df.to_numpy(dtype=float)
            p_test = _train_predict_xgb(
                x=x,
                y=y,
                fit_idx=fit_idx,
                cal_idx=cal_idx,
                test_idx=test_idx,
                params=params,
                calibration=calibration,
            )
            metrics = evaluate_metrics(y[test_idx], p_test)

            # Per-sex scoring.
            per_sex_rows: list[list[str]] = []
            sex_labels = sorted(set(sex_vals))
            fairacc_values: list[float] = []
            brier_values: list[float] = []

            for sex_label in sex_labels:
                idx = [i for i, s in enumerate(sex_vals) if s == sex_label]
                if not idx:
                    continue
                y_sub = [y_test[i] for i in idx]
                p_sub = [float(p_test[i]) for i in idx]
                race_sub = [race_vals[i] for i in idx]
                brier_sub = float(brier_score(y_sub, p_sub))
                fpr_by_race = fpr_by_race_at_threshold(y_true=y_sub, y_prob=p_sub, race=race_sub, threshold=0.5)
                fpr_black = float(fpr_by_race.get("BLACK", float("nan")))
                fpr_white = float(fpr_by_race.get("WHITE", float("nan")))
                fp_term = float(nij_fp_term(fpr_black, fpr_white))
                fair_acc = float(nij_fair_and_accurate(brier_sub, fp_term))

                brier_values.append(brier_sub)
                fairacc_values.append(fair_acc)

                per_sex_rows.append(
                    [
                        str(sex_label),
                        str(len(idx)),
                        format_float(brier_sub),
                        format_float(fpr_black),
                        format_float(fpr_white),
                        format_float(fp_term),
                        format_float(fair_acc),
                    ]
                )

            brier_avg = float(np.mean(brier_values)) if brier_values else float("nan")
            fairacc_avg = float(np.mean([v for v in fairacc_values if not np.isnan(v)])) if fairacc_values else float("nan")

            summary_rows.append(
                [
                    horizon.upper(),
                    variant_name,
                    dataset_key,
                    calibration,
                    format_float(float(metrics["brier"])),
                    format_float(brier_avg),
                    format_float(fairacc_avg),
                ]
            )

            lines.extend(
                [
                    f"### Variant: {variant_name}",
                    "",
                    "| Metric | Value |",
                    "|---|---:|",
                    f"| Overall Brier | {format_float(float(metrics['brier']))} |",
                    f"| AUROC | {format_float(float(metrics['auroc']))} |",
                    f"| AUPRC | {format_float(float(metrics['auprc']))} |",
                    f"| Log loss | {format_float(float(metrics['log_loss']))} |",
                    f"| ECE | {format_float(float(metrics['ece']))} |",
                    "",
                    "#### NIJ-style terms by sex (threshold=0.5)",
                    "",
                    "| Sex | N | BS (Brier) | FPR_black@0.5 | FPR_white@0.5 | FP | FairAcc |",
                    "|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in per_sex_rows:
                lines.append("| " + " | ".join(row) + " |")

            lines.extend(
                [
                    "",
                    f"- Sex-average BS: `{format_float(brier_avg)}`",
                    f"- Sex-average FairAcc: `{format_float(fairacc_avg)}`",
                    "",
                ]
            )

    lines.extend(
        [
            "# Summary (sex-averaged)",
            "",
            "| Horizon | Variant | Dataset | Calibration | Overall Brier | Sex-avg BS | Sex-avg FairAcc |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append("| " + " | ".join(row) + " |")

    out.write_text("\n".join(lines))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-models", type=Path, default=None, help="Path to xgb_best_models.json")
    parser.add_argument("--out", type=Path, default=None, help="Output path for markdown report")
    args = parser.parse_args()

    report = write_nij_scoring_report(best_models_path=args.best_models, out_path=args.out)
    print(f"wrote nij scoring report: {report}")


if __name__ == "__main__":
    main()

