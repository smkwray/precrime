"""Run NIJ model pipelines (baselines and XGBoost)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.fairness import (
    calibration_by_group,
    equalized_odds_gaps,
    fpr_gap,
    fnr_gap,
    predictive_parity_proxy,
    subgroup_auprc,
    subgroup_auroc,
    subgroup_brier,
    threshold_gap_summary,
)
from src.eval.metrics import auprc, auroc, bootstrap_ci, brier_score, expected_calibration_error, log_loss
from src.eval.plots import plot_calibration
from src.features.build_nij_dynamic import build_dynamic_datasets
from src.features.build_nij_static import build_static_datasets
from src.models.baselines import BaseRateModel, DemographicNaiveModel
from src.models.calibration import IsotonicCalibrator, PlattCalibrator
from src.models.lasso import LassoLogisticRegression
from src.models.logistic import LogisticRegressionGD
from src.pipelines._split_utils import (
    RANDOM_SEED,
    N_FOLDS,
    build_cv_splits,
    evaluate_metrics,
    extract_target_column,
    fit_calibration_split,
    prepare_feature_matrix,
    split_train_cal_select_test,
    split_train_cal_test,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _reports_dir() -> Path:
    path = _repo_root() / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _plots_dir() -> Path:
    path = _reports_dir() / "plots"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _add_age_group(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    age = pd.to_numeric(out.get("Age_at_Release"), errors="coerce")
    out["age_group"] = pd.cut(
        age,
        bins=[-np.inf, 24, 34, 44, np.inf],
        labels=["<25", "25-34", "35-44", "45+"],
    ).astype("object")
    out["age_group"] = out["age_group"].fillna("Unknown")
    return out




def _standardize(x_train: np.ndarray, x_other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.mean(x_train, axis=0)
    sigma = np.std(x_train, axis=0)
    sigma[sigma == 0.0] = 1.0
    return (x_train - mu) / sigma, (x_other - mu) / sigma




def _save_calibration_plot_specs(
    dataset_key: str,
    horizon: str,
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    frame: pd.DataFrame,
) -> None:
    groups = {
        "race": frame["Race"].fillna("Unknown").astype(str).tolist() if "Race" in frame else None,
        "gender": frame["Gender"].fillna("Unknown").astype(str).tolist() if "Gender" in frame else None,
        "age_group": frame["age_group"].fillna("Unknown").astype(str).tolist(),
    }

    for group_name, group_vals in groups.items():
        if group_vals is None:
            continue
        fig_spec = plot_calibration(
            y_true=y_true.astype(int).tolist(),
            y_prob=y_prob.astype(float).tolist(),
            group=group_vals,
            n_bins=10,
            title=f"Calibration {model_name} {dataset_key} {horizon} by {group_name}",
        )
        out_path = _plots_dir() / f"calibration_{model_name}_{dataset_key}_{horizon}_{group_name}.json"
        out_path.write_text(json.dumps(fig_spec, indent=2))




def _model_probabilities(
    model_name: str,
    frame: pd.DataFrame,
    y: np.ndarray,
    splits,
) -> np.ndarray:
    pred = np.zeros(len(frame), dtype=float)

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        fit_idx, cal_idx = fit_calibration_split(train_idx, seed=RANDOM_SEED + fold_i)

        # Fit encoding on training fold only (fix preprocessing leakage)
        _, fit_cols = prepare_feature_matrix(frame.iloc[train_idx])
        x_fold, _ = prepare_feature_matrix(frame, fit_columns=fit_cols)
        x_matrix = x_fold.to_numpy(dtype=float)

        y_fit = y[fit_idx]
        y_cal = y[cal_idx]

        if model_name == "base_rate":
            model = BaseRateModel().fit(y_fit)
            p_cal_raw = model.predict_proba(len(cal_idx))
            p_val_raw = model.predict_proba(len(val_idx))

        elif model_name == "naive_demographic":
            model = DemographicNaiveModel().fit(frame.iloc[fit_idx], y_fit)
            p_cal_raw = model.predict_proba(frame.iloc[cal_idx])
            p_val_raw = model.predict_proba(frame.iloc[val_idx])

        elif model_name == "logistic":
            x_fit = x_matrix[fit_idx]
            x_cal = x_matrix[cal_idx]
            x_val = x_matrix[val_idx]
            x_fit_s, x_cal_s = _standardize(x_fit, x_cal)
            _, x_val_s = _standardize(x_fit, x_val)

            model = LogisticRegressionGD(learning_rate=0.05, n_iter=350, l2=1e-3).fit(x_fit_s, y_fit)
            p_cal_raw = model.predict_proba(x_cal_s)
            p_val_raw = model.predict_proba(x_val_s)

        elif model_name == "lasso_logistic":
            x_fit = x_matrix[fit_idx]
            x_cal = x_matrix[cal_idx]
            x_val = x_matrix[val_idx]
            x_fit_s, x_cal_s = _standardize(x_fit, x_cal)
            _, x_val_s = _standardize(x_fit, x_val)

            model = LassoLogisticRegression(learning_rate=0.05, n_iter=450, l1=8e-4).fit(x_fit_s, y_fit)
            p_cal_raw = model.predict_proba(x_cal_s)
            p_val_raw = model.predict_proba(x_val_s)

        else:
            raise ValueError(f"Unknown model: {model_name}")

        calibrator = PlattCalibrator(learning_rate=0.05, n_iter=250).fit(p_cal_raw, y_cal)
        pred[val_idx] = calibrator.predict(p_val_raw)

    return np.clip(pred, 1e-6, 1.0 - 1e-6)


def run_static_baselines() -> pd.DataFrame:
    datasets = build_static_datasets()
    rows: list[dict[str, float | str]] = []
    model_order = ["base_rate", "naive_demographic", "logistic", "lasso_logistic"]

    for horizon, ds in sorted(datasets.items()):
        ds = _add_age_group(ds)
        target_col = extract_target_column(ds)
        y = pd.to_numeric(ds[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
        feature_frame = ds.drop(columns=[target_col]).copy()

        splits = build_cv_splits(len(ds), n_folds=N_FOLDS, seed=RANDOM_SEED)

        for model_name in model_order:
            y_prob = _model_probabilities(
                model_name=model_name,
                frame=feature_frame,
                y=y,
                splits=splits,
            )
            metrics = evaluate_metrics(y, y_prob)
            rows.append(
                {
                    "dataset": "static",
                    "horizon": horizon,
                    "model": model_name,
                    "calibration": "platt",
                    **metrics,
                }
            )
            _save_calibration_plot_specs("static", horizon, model_name, y, y_prob, ds)

    return pd.DataFrame(rows)


def _write_markdown_table(
    title: str,
    rows_df: pd.DataFrame,
    path: Path,
    extra_lines: list[str] | None = None,
) -> Path:
    ordered = rows_df.sort_values(["dataset", "horizon", "brier", "auroc"], ascending=[True, True, True, False]).copy()
    lines = [
        f"# {title}",
        "",
        "Primary sort: Brier score (lower is better).",
        "",
        "| Dataset | Horizon | Model | Calibration | Brier | AUROC | AUPRC | Log Loss | ECE |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in ordered.iterrows():
        lines.append(
            "| "
            + f"{row['dataset']} | {row['horizon']} | {row['model']} | {row['calibration']} | {row['brier']:.5f} | {row['auroc']:.5f} | {row['auprc']:.5f} | {row['log_loss']:.5f} | {row['ece']:.5f} |"
        )

    if extra_lines:
        lines.extend(["", *extra_lines])

    path.write_text("\n".join(lines))
    return path


def write_baseline_leaderboard(results: pd.DataFrame, path: Path | None = None) -> Path:
    out_path = path or (_reports_dir() / "baseline_leaderboard.md")
    return _write_markdown_table(
        title="NIJ Static Baseline Leaderboard",
        rows_df=results,
        path=out_path,
        extra_lines=["Calibration specs are saved under `reports/plots/` as JSON (HTML-friendly)."],
    )


def _save_bar_spec(path: Path, title: str, key_name: str, value_name: str, rows: list[dict[str, float | str]]) -> None:
    spec = {
        "kind": "bar",
        "title": title,
        "xaxis": key_name,
        "yaxis": value_name,
        "x": [str(r[key_name]) for r in rows],
        "y": [float(r[value_name]) for r in rows],
    }
    path.write_text(json.dumps(spec, indent=2))


def run_xgb_models(n_trials: int = 16, seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    from sklearn.model_selection import train_test_split

    from src.models.xgb import feature_importance_table, shap_summary_table, train_xgb, tune_xgb

    all_sets = {
        "static": build_static_datasets(),
        "dynamic": build_dynamic_datasets(),
    }

    records: list[dict[str, float | str]] = []
    best_by_horizon: dict[str, dict[str, object]] = {}

    for dataset_key, horizon_map in all_sets.items():
        for horizon, ds in sorted(horizon_map.items()):
            ds = _add_age_group(ds)
            target_col = extract_target_column(ds)

            y = pd.to_numeric(ds[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
            feature_frame = ds.drop(columns=[target_col]).copy()

            # 4-way split: fit/cal/select/test
            fit_idx, cal_idx, select_idx, test_idx = split_train_cal_select_test(y, seed=seed)

            # Fit encoding on training data only (fix preprocessing leakage)
            train_indices = np.concatenate([fit_idx, cal_idx, select_idx])
            _, fit_columns = prepare_feature_matrix(feature_frame.iloc[train_indices])
            x_df, feature_names = prepare_feature_matrix(feature_frame, fit_columns=fit_columns)
            x = x_df.to_numpy(dtype=float)

            tune_train_idx, tune_val_idx = train_test_split(
                fit_idx,
                test_size=0.2,
                random_state=seed,
                stratify=y[fit_idx],
            )

            tune = tune_xgb(
                x_train=x[tune_train_idx],
                y_train=y[tune_train_idx],
                x_valid=x[tune_val_idx],
                y_valid=y[tune_val_idx],
                seed=seed,
                n_trials=n_trials,
            )

            model = train_xgb(
                x_train=x[fit_idx],
                y_train=y[fit_idx],
                params=tune.best_params,
                seed=seed,
            )

            p_cal_raw = model.predict_proba(x[cal_idx])[:, 1]
            p_select_raw = model.predict_proba(x[select_idx])[:, 1]
            p_test_raw = model.predict_proba(x[test_idx])[:, 1]

            platt = PlattCalibrator().fit(p_cal_raw, y[cal_idx])
            isotonic = IsotonicCalibrator().fit(p_cal_raw, y[cal_idx])

            # Evaluate calibration methods on select split for model selection
            y_select = y[select_idx]
            select_preds = {
                "platt": platt.predict(p_select_raw),
                "isotonic": isotonic.predict(p_select_raw),
            }
            select_scores: dict[str, dict[str, float]] = {}
            for cal_name, p_sel in select_preds.items():
                select_scores[cal_name] = evaluate_metrics(y_select, p_sel)

            # Evaluate all variants on test split for reporting
            preds = {
                "raw": p_test_raw,
                "platt": platt.predict(p_test_raw),
                "isotonic": isotonic.predict(p_test_raw),
            }

            y_test = y[test_idx]
            for cal_name, p_test in preds.items():
                metrics = evaluate_metrics(y_test, p_test)
                records.append(
                    {
                        "dataset": dataset_key,
                        "horizon": horizon,
                        "model": "xgboost",
                        "calibration": cal_name,
                        **metrics,
                    }
                )
                _save_calibration_plot_specs(dataset_key, horizon, f"xgboost_{cal_name}", y_test, p_test, ds.iloc[test_idx])

            imp_rows = feature_importance_table(model, feature_names=feature_names, top_k=30)
            shap_rows = shap_summary_table(model, x_matrix=x[test_idx], feature_names=feature_names, top_k=30)

            imp_path = _plots_dir() / f"xgb_importance_{dataset_key}_{horizon}.json"
            shap_path = _plots_dir() / f"xgb_shap_{dataset_key}_{horizon}.json"
            _save_bar_spec(
                imp_path,
                title=f"XGBoost Feature Importance {dataset_key} {horizon}",
                key_name="feature",
                value_name="importance",
                rows=imp_rows,
            )
            _save_bar_spec(
                shap_path,
                title=f"XGBoost SHAP Summary {dataset_key} {horizon}",
                key_name="feature",
                value_name="mean_abs_shap",
                rows=shap_rows,
            )

            # Select best calibration using select-split Brier (not test-split)
            best_cal_name = min(select_scores, key=lambda k: select_scores[k]["brier"])
            best_test_row = next(
                r for r in records
                if r["dataset"] == dataset_key and r["horizon"] == horizon and r["calibration"] == best_cal_name
            )
            best_key = str(horizon)
            prev = best_by_horizon.get(best_key)
            select_brier = select_scores[best_cal_name]["brier"]
            if prev is None or select_brier < float(prev["select_metrics"]["brier"]):
                best_by_horizon[best_key] = {
                    "dataset": dataset_key,
                    "horizon": horizon,
                    "model": "xgboost",
                    "calibration": best_cal_name,
                    "seed": seed,
                    "tuning": {
                        "n_trials": tune.n_trials,
                        "best_validation_brier": tune.best_score,
                        "best_params": tune.best_params,
                    },
                    "select_metrics": {
                        "brier": float(select_scores[best_cal_name]["brier"]),
                        "auroc": float(select_scores[best_cal_name]["auroc"]),
                        "auprc": float(select_scores[best_cal_name]["auprc"]),
                        "log_loss": float(select_scores[best_cal_name]["log_loss"]),
                        "ece": float(select_scores[best_cal_name]["ece"]),
                    },
                    "metrics": {
                        "brier": float(best_test_row["brier"]),
                        "auroc": float(best_test_row["auroc"]),
                        "auprc": float(best_test_row["auprc"]),
                        "log_loss": float(best_test_row["log_loss"]),
                        "ece": float(best_test_row["ece"]),
                    },
                    "artifacts": {
                        "importance_plot": str(imp_path),
                        "shap_plot": str(shap_path),
                    },
                }

    return pd.DataFrame(records), best_by_horizon


def run_xgb_models_cv(
    n_trials: int = 16,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    """Run XGBoost across multiple seed splits for comparison with CV baselines."""
    seeds = seeds or [42, 43, 44, 45, 46]
    all_records = []
    for seed_val in seeds:
        results, _ = run_xgb_models(n_trials=n_trials, seed=seed_val)
        results["cv_seed"] = seed_val
        all_records.append(results)
    combined = pd.concat(all_records, ignore_index=True)

    # Aggregate mean/std per (dataset, horizon, calibration)
    group_cols = ["dataset", "horizon", "calibration"]
    metric_cols = ["brier", "auroc", "auprc", "log_loss", "ece"]
    agg_dict = {}
    for m in metric_cols:
        agg_dict[f"{m}_mean"] = (m, "mean")
        agg_dict[f"{m}_std"] = (m, "std")
    summary = combined.groupby(group_cols).agg(**agg_dict).reset_index()
    return summary


def write_xgb_outputs(results: pd.DataFrame, best_by_horizon: dict[str, dict[str, object]]) -> tuple[Path, Path]:
    leaderboard = _write_markdown_table(
        title="NIJ XGBoost Leaderboard",
        rows_df=results,
        path=_reports_dir() / "xgb_leaderboard.md",
        extra_lines=[
            "Includes raw, Platt, and isotonic calibration variants.",
            "Feature-importance and SHAP summary specs are under `reports/plots/` as JSON.",
        ],
    )
    best_path = _reports_dir() / "xgb_best_models.json"
    best_path.write_text(json.dumps(best_by_horizon, indent=2))
    return leaderboard, best_path


def write_xgb_cv_leaderboard(summary: pd.DataFrame, path: Path | None = None) -> Path:
    out_path = path or (_reports_dir() / "xgb_cv_leaderboard.md")
    lines = [
        "# NIJ XGBoost CV Leaderboard (Multi-Seed)",
        "",
        "Mean ± std across seeds. Primary sort: mean Brier score.",
        "",
        "| Dataset | Horizon | Calibration | Brier (mean±std) | AUROC (mean±std) | AUPRC (mean±std) |",
        "|---|---|---|---|---|---|",
    ]
    ordered = summary.sort_values(["dataset", "horizon", "brier_mean"], ascending=[True, True, True])
    for _, row in ordered.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['horizon']} | {row['calibration']} "
            f"| {row['brier_mean']:.5f}±{row['brier_std']:.5f} "
            f"| {row['auroc_mean']:.5f}±{row['auroc_std']:.5f} "
            f"| {row['auprc_mean']:.5f}±{row['auprc_std']:.5f} |"
        )
    out_path.write_text("\n".join(lines))
    return out_path


def _load_best_xgb_models(path: Path | None = None) -> dict[str, dict[str, object]]:
    best_path = path or (_reports_dir() / "xgb_best_models.json")
    if not best_path.exists():
        raise FileNotFoundError(
            f"Missing {best_path}. Run `--task xgb` first to generate best model configs."
        )
    return json.loads(best_path.read_text())


def _apply_calibrator(name: str, p_cal: np.ndarray, y_cal: np.ndarray, p_eval: np.ndarray) -> np.ndarray:
    if name == "platt":
        return PlattCalibrator().fit(p_cal, y_cal).predict(p_eval)
    if name == "isotonic":
        return IsotonicCalibrator().fit(p_cal, y_cal).predict(p_eval)
    return np.clip(p_eval, 1e-6, 1.0 - 1e-6)




def _train_predict_xgb(
    x_df: pd.DataFrame,
    y: np.ndarray,
    fit_idx: np.ndarray,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    best_params: dict[str, float | int],
    calibration_name: str,
) -> np.ndarray:
    from src.models.xgb import train_xgb

    x = x_df.to_numpy(dtype=float)
    model = train_xgb(
        x_train=x[fit_idx],
        y_train=y[fit_idx],
        params=best_params,
        seed=RANDOM_SEED,
    )
    p_cal = model.predict_proba(x[cal_idx])[:, 1]
    p_test_raw = model.predict_proba(x[test_idx])[:, 1]
    return _apply_calibrator(calibration_name, p_cal, y[cal_idx], p_test_raw)


def _subgroup_ci_rows(y_true: list[int], y_prob: list[float], group_vals: list[str]) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    labels = sorted(set(group_vals))
    metric_defs = {
        "brier": brier_score,
        "auroc": auroc,
        "auprc": auprc,
    }

    for label in labels:
        idx = [i for i, g in enumerate(group_vals) if g == label]
        y_sub = [y_true[i] for i in idx]
        p_sub = [y_prob[i] for i in idx]
        for metric_name, metric_fn in metric_defs.items():
            ci = bootstrap_ci(
                y_true=y_sub,
                y_prob=p_sub,
                metric_fn=metric_fn,
                n_bootstrap=300,
                random_state=RANDOM_SEED,
                stratify_by_label=True,
            )
            rows.append(
                {
                    "group": label,
                    "metric": metric_name,
                    "estimate": float(ci["estimate"]),
                    "lower": float(ci["lower"]),
                    "upper": float(ci["upper"]),
                }
            )
    return rows


def _format_simple_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def run_fairness_audit() -> Path:
    best_cfg = _load_best_xgb_models()
    all_sets = {
        "static": build_static_datasets(),
        "dynamic": build_dynamic_datasets(),
    }

    report_lines = [
        "# NIJ Fairness Report",
        "",
        "Primary metric: Brier score. Fairness diagnostics include subgroup error, calibration, FPR/FNR gaps, equalized-odds gaps, predictive parity proxy, threshold-gap summaries, and bootstrap CIs.",
        "",
    ]

    for horizon in sorted(best_cfg.keys()):
        cfg = best_cfg[horizon]
        dataset_key = str(cfg["dataset"])
        calibration_name = str(cfg["calibration"])
        best_params = dict(cfg["tuning"]["best_params"])

        ds = _add_age_group(all_sets[dataset_key][horizon])
        target_col = extract_target_column(ds)
        y = pd.to_numeric(ds[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
        feature_frame = ds.drop(columns=[target_col]).copy()
        fit_idx, cal_idx, test_idx = split_train_cal_test(y)

        # Fit encoding on training data only
        train_frame = feature_frame.iloc[np.concatenate([fit_idx, cal_idx])]
        _, fit_cols_with = prepare_feature_matrix(train_frame)
        x_with_race, _ = prepare_feature_matrix(feature_frame, fit_columns=fit_cols_with)
        no_race_cols = [
            c for c in feature_frame.columns
            if c != "Race" and not str(c).startswith("Race_") and "race" not in str(c).lower()
        ]
        no_race_frame = feature_frame[no_race_cols].copy()
        _, fit_cols_no_race = prepare_feature_matrix(train_frame[no_race_cols].copy())
        x_without_race, _ = prepare_feature_matrix(no_race_frame, fit_columns=fit_cols_no_race)

        p_with = _train_predict_xgb(
            x_df=x_with_race,
            y=y,
            fit_idx=fit_idx,
            cal_idx=cal_idx,
            test_idx=test_idx,
            best_params=best_params,
            calibration_name=calibration_name,
        )
        p_without = _train_predict_xgb(
            x_df=x_without_race,
            y=y,
            fit_idx=fit_idx,
            cal_idx=cal_idx,
            test_idx=test_idx,
            best_params=best_params,
            calibration_name=calibration_name,
        )

        y_test = y[test_idx].astype(int).tolist()
        test_frame = ds.iloc[test_idx].copy()
        groups = {
            "race": test_frame["Race"].fillna("Unknown").astype(str).tolist() if "Race" in test_frame else ["Unknown"] * len(test_frame),
            "gender": test_frame["Gender"].fillna("Unknown").astype(str).tolist() if "Gender" in test_frame else ["Unknown"] * len(test_frame),
            "age_group": test_frame["age_group"].fillna("Unknown").astype(str).tolist(),
        }

        overall_with = evaluate_metrics(np.asarray(y_test), p_with)
        overall_without = evaluate_metrics(np.asarray(y_test), p_without)
        better_variant = "with_race" if overall_with["brier"] <= overall_without["brier"] else "without_race"
        p_main = p_with.tolist() if better_variant == "with_race" else p_without.tolist()

        report_lines.extend([
            f"## Horizon {horizon.upper()} ({dataset_key})",
            "",
            f"Selected XGBoost calibration from FELI-0003: `{calibration_name}`",
            "",
            "### With-race vs Without-race",
            "",
        ])
        report_lines.extend(
            _format_simple_table(
                ["Variant", "Brier", "AUROC", "AUPRC", "Log Loss", "ECE"],
                [
                    ["with_race", f"{overall_with['brier']:.5f}", f"{overall_with['auroc']:.5f}", f"{overall_with['auprc']:.5f}", f"{overall_with['log_loss']:.5f}", f"{overall_with['ece']:.5f}"],
                    ["without_race", f"{overall_without['brier']:.5f}", f"{overall_without['auroc']:.5f}", f"{overall_without['auprc']:.5f}", f"{overall_without['log_loss']:.5f}", f"{overall_without['ece']:.5f}"],
                ],
            )
        )
        report_lines.extend(["", f"Chosen for subgroup fairness diagnostics: `{better_variant}`", ""])

        # Pooled vs subgroup-specific (race)
        pooled_by_race = subgroup_brier(y_test, p_main, group=groups["race"])
        subgroup_specific_rows: list[list[str]] = []
        for race_label in sorted(set(groups["race"])):
            race_train_mask = feature_frame["Race"].fillna("Unknown").astype(str) == race_label
            race_test_idx = [i for i, g in enumerate(groups["race"]) if g == race_label]
            if not race_test_idx or int(race_train_mask.sum()) < 50:
                continue

            race_frame = feature_frame.loc[race_train_mask].drop(columns=[c for c in ["Race"] if c in feature_frame.columns]).copy()
            race_y = y[race_train_mask.to_numpy()]
            rf_idx, rc_idx, rt_idx = split_train_cal_test(race_y)
            _, race_fit_cols = prepare_feature_matrix(race_frame.iloc[np.concatenate([rf_idx, rc_idx])])
            race_x, _ = prepare_feature_matrix(race_frame, fit_columns=race_fit_cols)
            p_race = _train_predict_xgb(
                x_df=race_x,
                y=race_y,
                fit_idx=rf_idx,
                cal_idx=rc_idx,
                test_idx=rt_idx,
                best_params=best_params,
                calibration_name=calibration_name,
            )
            subgroup_specific_rows.append(
                [
                    race_label,
                    f"{float(pooled_by_race.get(race_label, float('nan'))):.5f}",
                    f"{brier_score(race_y[rt_idx].astype(int).tolist(), p_race.tolist()):.5f}",
                ]
            )

        report_lines.append("### Pooled vs Subgroup-Specific (Race)")
        report_lines.append("")
        if subgroup_specific_rows:
            report_lines.extend(_format_simple_table(["Race", "Pooled Brier", "Subgroup-Specific Brier"], subgroup_specific_rows))
        else:
            report_lines.append("Insufficient per-race sample size for subgroup-specific model training.")
        report_lines.extend(["", "### Subgroup Fairness Metrics", ""])

        for group_name, group_vals in groups.items():
            sb = subgroup_brier(y_test, p_main, group=group_vals)
            sa = subgroup_auroc(y_test, p_main, group=group_vals)
            sp = subgroup_auprc(y_test, p_main, group=group_vals)
            eo = equalized_odds_gaps(y_test, p_main, group=group_vals, threshold=0.5)
            pp = predictive_parity_proxy(y_test, p_main, group=group_vals, threshold=0.5)
            gap = threshold_gap_summary(y_test, p_main, group=group_vals)
            cal = calibration_by_group(y_test, p_main, group=group_vals, n_bins=10)
            ci_rows = _subgroup_ci_rows(y_test, p_main, group_vals)

            report_lines.append(f"#### {group_name.title()}")
            report_lines.append("")
            metric_rows = []
            for label in sorted(set(group_vals)):
                metric_rows.append([
                    label,
                    f"{float(sb.get(label, float('nan'))):.5f}",
                    f"{float(sa.get(label, float('nan'))):.5f}",
                    f"{float(sp.get(label, float('nan'))):.5f}",
                ])
            report_lines.extend(_format_simple_table(["Group", "Brier", "AUROC", "AUPRC"], metric_rows))
            report_lines.extend([
                "",
                f"- FPR gap: `{eo['fpr_gap']:.5f}`",
                f"- FNR gap: `{eo['fnr_gap']:.5f}`",
                f"- Equalized-odds max gap: `{eo['eo_gap_max']:.5f}`",
                f"- Predictive-parity proxy (PPV gap): `{float(pp['ppv_gap']):.5f}`",
                f"- Threshold sweep FPR gap max/mean: `{float(gap['fpr_gap_max']):.5f}` / `{float(gap['fpr_gap_mean']):.5f}`",
                f"- Threshold sweep FNR gap max/mean: `{float(gap['fnr_gap_max']):.5f}` / `{float(gap['fnr_gap_mean']):.5f}`",
                "",
                "Bootstrap CI (300 resamples, stratified by label):",
            ])
            ci_table_rows = [
                [
                    str(r["group"]),
                    str(r["metric"]),
                    f"{float(r['estimate']):.5f}",
                    f"{float(r['lower']):.5f}",
                    f"{float(r['upper']):.5f}",
                ]
                for r in ci_rows
            ]
            report_lines.extend(
                _format_simple_table(["Group", "Metric", "Estimate", "CI Lower", "CI Upper"], ci_table_rows)
            )
            report_lines.append("")

            # Save per-group calibration curves and threshold summaries as JSON artifacts.
            artifact_prefix = f"fairness_{horizon}_{dataset_key}_{group_name}"
            (_plots_dir() / f"{artifact_prefix}_calibration.json").write_text(json.dumps(cal, indent=2))
            (_plots_dir() / f"{artifact_prefix}_threshold_gap.json").write_text(json.dumps(gap, indent=2))

    out_path = _reports_dir() / "fairness_report.md"
    out_path.write_text("\n".join(report_lines))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NIJ model pipelines")
    parser.add_argument("--task", choices=["baselines", "xgb", "fairness", "all"], default="baselines")
    parser.add_argument(
        "--xgb-trials",
        type=int,
        default=int(os.getenv("PRECRIME_XGB_TRIALS", "16")),
        help="Optuna trial count per feature-set/horizon",
    )
    parser.add_argument("--cv", action="store_true", help="Run XGBoost with multi-seed splits")
    parser.add_argument("--cv-seeds", type=str, default="42,43,44,45,46", help="Comma-separated seeds for CV mode")
    args = parser.parse_args()

    if args.task in {"baselines", "all"}:
        baseline_results = run_static_baselines()
        out = write_baseline_leaderboard(baseline_results)
        print(f"wrote baseline leaderboard: {out}")
        print(baseline_results.sort_values(["dataset", "horizon", "brier"]).to_string(index=False))

    if args.task in {"xgb", "all"}:
        if args.cv:
            cv_seeds = [int(s) for s in args.cv_seeds.split(",")]
            cv_summary = run_xgb_models_cv(n_trials=args.xgb_trials, seeds=cv_seeds)
            cv_path = write_xgb_cv_leaderboard(cv_summary)
            print(f"wrote xgb CV leaderboard: {cv_path}")
            print(cv_summary.to_string(index=False))
        else:
            xgb_results, best = run_xgb_models(n_trials=args.xgb_trials)
            lb_path, best_path = write_xgb_outputs(xgb_results, best)
            print(f"wrote xgb leaderboard: {lb_path}")
            print(f"wrote xgb best-model config: {best_path}")
            print(xgb_results.sort_values(["dataset", "horizon", "brier"]).to_string(index=False))

    if args.task in {"fairness", "all"}:
        report_path = run_fairness_audit()
        print(f"wrote fairness report: {report_path}")


if __name__ == "__main__":
    main()
