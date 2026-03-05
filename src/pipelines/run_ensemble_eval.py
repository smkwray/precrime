"""Evaluate simple seed-ensembles for NIJ best configs (no new deps).

This trains the best-per-horizon XGBoost configs across multiple random seeds,
averages predicted probabilities, then (optionally) recalibrates once.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.metrics import auprc, auroc, brier_score, expected_calibration_error, log_loss
from src.eval.nij_scoring import brier_by_sex, fpr_by_race_at_threshold, nij_fair_and_accurate, nij_fp_term
from src.features.build_nij_dynamic import build_dynamic_datasets
from src.features.build_nij_static import build_static_datasets
from src.models.calibration import IsotonicCalibrator, PlattCalibrator
from src.models.xgb import train_xgb


SPLIT_SEED = 42


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


def _extract_target_column(ds: pd.DataFrame) -> str:
    for candidate in ("y", "target"):
        if candidate in ds.columns:
            return candidate
    raise ValueError("Unable to detect target column in dataset")


def _prepare_feature_matrix(feature_frame: pd.DataFrame) -> pd.DataFrame:
    model_frame = feature_frame.drop(columns=["ID"], errors="ignore")
    x_matrix = pd.get_dummies(model_frame, dummy_na=True)
    x_matrix = x_matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return x_matrix


def _split_train_cal_test(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=SPLIT_SEED,
        stratify=y,
    )
    fit_idx, cal_idx = train_test_split(
        train_idx,
        test_size=0.25,
        random_state=SPLIT_SEED,
        stratify=y[train_idx],
    )
    return fit_idx, cal_idx, test_idx


def _evaluate_overall(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_list = y_true.astype(int).tolist()
    p_list = y_prob.astype(float).tolist()
    return {
        "brier": brier_score(y_list, p_list),
        "auroc": auroc(y_list, p_list),
        "auprc": auprc(y_list, p_list),
        "log_loss": log_loss(y_list, p_list),
        "ece": expected_calibration_error(y_list, p_list, n_bins=10),
    }


def _nij_terms(y_true: list[int], y_prob: list[float], sex: list[str], race: list[str]) -> dict[str, float]:
    brier_sex = brier_by_sex(y_true, y_prob, sex)
    bs_m = float(brier_sex.get("M", float("nan")))
    bs_f = float(brier_sex.get("F", float("nan")))
    sex_avg_bs = float(np.nanmean([bs_m, bs_f]))

    fairacc: dict[str, float] = {}
    for sex_label in ("M", "F"):
        idx = [i for i, s in enumerate(sex) if s == sex_label]
        if not idx:
            fairacc[sex_label] = float("nan")
            continue
        y_sub = [y_true[i] for i in idx]
        p_sub = [y_prob[i] for i in idx]
        race_sub = [race[i] for i in idx]
        bs = float(brier_score(y_sub, p_sub))
        fpr_by = fpr_by_race_at_threshold(y_sub, p_sub, race_sub, threshold=0.5)
        fpr_black = float(fpr_by.get("BLACK", float("nan")))
        fpr_white = float(fpr_by.get("WHITE", float("nan")))
        fp = float(nij_fp_term(fpr_black, fpr_white))
        fairacc[sex_label] = float(nij_fair_and_accurate(bs, fp))

    fairacc_m = float(fairacc.get("M", float("nan")))
    fairacc_f = float(fairacc.get("F", float("nan")))
    sex_avg_fairacc = float(np.nanmean([fairacc_m, fairacc_f]))

    return {
        "bs_m": bs_m,
        "bs_f": bs_f,
        "bs_sex_avg": sex_avg_bs,
        "fairacc_m": fairacc_m,
        "fairacc_f": fairacc_f,
        "fairacc_sex_avg": sex_avg_fairacc,
    }


def _format_float(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value:.5f}"


def _calibrate(name: str, p_cal_raw: np.ndarray, y_cal: np.ndarray, p_test_raw: np.ndarray) -> np.ndarray:
    if name == "platt":
        return PlattCalibrator().fit(p_cal_raw, y_cal).predict(p_test_raw)
    if name == "isotonic":
        return IsotonicCalibrator().fit(p_cal_raw, y_cal).predict(p_test_raw)
    return np.clip(p_test_raw, 1e-6, 1.0 - 1e-6)


def write_ensemble_report(
    best_models_path: Path | None = None,
    out_path: Path | None = None,
    seeds: list[int] | None = None,
) -> Path:
    best = _load_best_models(best_models_path)
    out_path = out_path or (_reports_dir() / "ensemble_eval.md")
    seeds = seeds or [42, 43, 44, 45, 46]

    all_sets = {
        "static": build_static_datasets(),
        "dynamic": build_dynamic_datasets(),
    }

    lines: list[str] = [
        "# Seed-Ensemble Evaluation (NIJ Best XGBoost Configs)",
        "",
        f"Generated: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}",
        "",
        f"Split seed: `{SPLIT_SEED}` (fixed). Model seeds ensembled: `{','.join(str(s) for s in seeds)}`.",
        "",
        "Ensembling method: average predicted probabilities across model seeds, then optionally recalibrate once (Platt / isotonic).",
        "",
        "| Horizon | Dataset | Variant | Calibration | bs_sex_avg | fairacc_sex_avg | Overall Brier | AUROC | AUPRC | ECE |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for horizon in sorted(best.keys()):
        cfg = best[horizon]
        dataset_key = str(cfg["dataset"])
        best_params = dict(cfg.get("tuning", {}).get("best_params", {}))
        baseline_cal = str(cfg.get("calibration", "raw"))

        ds = _add_age_group(all_sets[dataset_key][horizon])
        target_col = _extract_target_column(ds)
        y = pd.to_numeric(ds[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
        feature_frame = ds.drop(columns=[target_col]).copy()
        fit_idx, cal_idx, test_idx = _split_train_cal_test(y)

        x_df = _prepare_feature_matrix(feature_frame)
        x = x_df.to_numpy(dtype=float)

        test_frame = ds.iloc[test_idx].copy()
        y_test = y[test_idx].astype(int).tolist()
        sex_vals = test_frame["Gender"].fillna("Unknown").astype(str).tolist() if "Gender" in test_frame else ["Unknown"] * len(test_frame)
        race_vals = test_frame["Race"].fillna("Unknown").astype(str).tolist() if "Race" in test_frame else ["Unknown"] * len(test_frame)

        # Baseline single model at seed=42 with its chosen calibration.
        model = train_xgb(x_train=x[fit_idx], y_train=y[fit_idx], params=best_params, seed=42)
        p_cal_raw = model.predict_proba(x[cal_idx])[:, 1]
        p_test_raw = model.predict_proba(x[test_idx])[:, 1]
        p_base = _calibrate(baseline_cal, p_cal_raw, y[cal_idx], p_test_raw)
        overall = _evaluate_overall(y[test_idx], p_base)
        nij = _nij_terms(y_test, p_base.astype(float).tolist(), sex_vals, race_vals)
        lines.append(
            "| "
            + f"{horizon.upper()} | {dataset_key} | single_seed42 | {baseline_cal} | "
            + f"{_format_float(nij['bs_sex_avg'])} | {_format_float(nij['fairacc_sex_avg'])} | {_format_float(overall['brier'])} | "
            + f"{_format_float(overall['auroc'])} | {_format_float(overall['auprc'])} | {_format_float(overall['ece'])} |"
        )

        # Seed ensemble: average raw probs.
        p_cal_list: list[np.ndarray] = []
        p_test_list: list[np.ndarray] = []
        for seed in seeds:
            m = train_xgb(x_train=x[fit_idx], y_train=y[fit_idx], params=best_params, seed=int(seed))
            p_cal_list.append(m.predict_proba(x[cal_idx])[:, 1])
            p_test_list.append(m.predict_proba(x[test_idx])[:, 1])

        p_cal_mean = np.mean(np.stack(p_cal_list, axis=0), axis=0)
        p_test_mean = np.mean(np.stack(p_test_list, axis=0), axis=0)

        for cal_name in ("raw", "platt", "isotonic"):
            p_ens = _calibrate(cal_name, p_cal_mean, y[cal_idx], p_test_mean)
            overall = _evaluate_overall(y[test_idx], p_ens)
            nij = _nij_terms(y_test, p_ens.astype(float).tolist(), sex_vals, race_vals)
            lines.append(
                "| "
                + f"{horizon.upper()} | {dataset_key} | seed_ensemble | {cal_name} | "
                + f"{_format_float(nij['bs_sex_avg'])} | {_format_float(nij['fairacc_sex_avg'])} | {_format_float(overall['brier'])} | "
                + f"{_format_float(overall['auroc'])} | {_format_float(overall['auprc'])} | {_format_float(overall['ece'])} |"
            )

    lines.extend(
        [
            "",
            "Notes:",
            "- This keeps the data split fixed and varies only model random seeds.",
            "- If you want a more pessimistic robustness check, vary the split seed and ensemble across splits (but then test sets differ).",
        ]
    )

    out_path.write_text("\n".join(lines))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-models", type=Path, default=None, help="Path to xgb_best_models.json")
    parser.add_argument("--out", type=Path, default=None, help="Output markdown path")
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44,45,46",
        help="Comma-separated model seeds to ensemble (default: 42-46).",
    )
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    report = write_ensemble_report(best_models_path=args.best_models, out_path=args.out, seeds=seeds)
    print(f"wrote ensemble report: {report}")


if __name__ == "__main__":
    main()

