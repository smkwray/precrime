"""Run COMPAS (ProPublica) benchmark pipelines (baselines and XGBoost)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.metrics import auprc, auroc, brier_score, expected_calibration_error, log_loss
from src.eval.plots import plot_calibration
from src.features.build_compas import build_compas_2yr
from src.models.baselines import BaseRateModel, DemographicNaiveModel
from src.models.calibration import IsotonicCalibrator, PlattCalibrator
from src.models.lasso import LassoLogisticRegression
from src.models.logistic import LogisticRegressionGD


RANDOM_SEED = 42


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


def _extract_target_column(ds: pd.DataFrame) -> str:
    for candidate in ("y", "target"):
        if candidate in ds.columns:
            return candidate
    raise ValueError("Unable to detect target column in COMPAS dataset (expected `y`)")


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


def _prepare_feature_matrix(feature_frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    model_frame = feature_frame.drop(columns=["id"], errors="ignore")
    x_matrix = pd.get_dummies(model_frame, dummy_na=True)
    x_matrix = x_matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return x_matrix, x_matrix.columns.astype(str).tolist()


def _build_cv_splits(n_rows: int, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n_rows)
    rng.shuffle(idx)

    fold_sizes = np.full(n_folds, n_rows // n_folds, dtype=int)
    fold_sizes[: n_rows % n_folds] += 1

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    cursor = 0
    for fold_size in fold_sizes:
        val_idx = idx[cursor : cursor + fold_size]
        train_idx = np.concatenate([idx[:cursor], idx[cursor + fold_size :]])
        splits.append((train_idx, val_idx))
        cursor += fold_size
    return splits


def _fit_calibration_split(train_idx: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = train_idx.copy()
    rng.shuffle(shuffled)

    cut = max(1, int(0.8 * len(shuffled)))
    fit_idx = shuffled[:cut]
    cal_idx = shuffled[cut:]
    if len(cal_idx) == 0:
        cal_idx = fit_idx[-1:]
        fit_idx = fit_idx[:-1]
    if len(fit_idx) == 0:
        fit_idx = cal_idx
    return fit_idx, cal_idx


def _standardize(x_train: np.ndarray, x_other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.mean(x_train, axis=0)
    sigma = np.std(x_train, axis=0)
    sigma[sigma == 0.0] = 1.0
    return (x_train - mu) / sigma, (x_other - mu) / sigma


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


def _save_calibration_plot_specs(
    model_name: str,
    calibration_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    frame: pd.DataFrame,
) -> None:
    plot_dir = _plots_dir()
    groups = {
        "race": frame["race"].fillna("Unknown").astype(str).tolist() if "race" in frame else None,
        "sex": frame["sex"].fillna("Unknown").astype(str).tolist() if "sex" in frame else None,
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
            title=f"COMPAS Calibration {model_name} ({calibration_name}) by {group_name}",
        )
        out_path = plot_dir / f"calibration_compas_{model_name}_{calibration_name}_{group_name}.json"
        out_path.write_text(json.dumps(fig_spec, indent=2))


def run_baselines(n_folds: int = 5) -> pd.DataFrame:
    df = _add_age_group(load_compas_processed())
    target_col = _extract_target_column(df)

    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
    feature_frame = df.drop(columns=[target_col]).copy()
    x_df, _ = _prepare_feature_matrix(feature_frame)
    x = x_df.to_numpy(dtype=float)

    splits = _build_cv_splits(len(df), n_folds=n_folds, seed=RANDOM_SEED)
    records: list[dict[str, float | str]] = []

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        fit_idx, cal_idx = _fit_calibration_split(train_idx, seed=RANDOM_SEED + fold_i)
        y_fit = y[fit_idx]
        y_cal = y[cal_idx]

        model_frame = feature_frame.iloc[val_idx].copy()

        # base rate
        base = BaseRateModel().fit(y_fit)
        p_cal_raw = base.predict_proba(len(cal_idx))
        p_val_raw = base.predict_proba(len(val_idx))
        p_val = PlattCalibrator().fit(p_cal_raw, y_cal).predict(p_val_raw)
        records.append({"model": "base_rate", "calibration": "platt", "fold": fold_i, **_evaluate(y[val_idx], p_val)})

        # naive demographic base rate
        demo = DemographicNaiveModel(group_cols=("sex", "race", "age_group")).fit(feature_frame.iloc[fit_idx], y_fit)
        p_cal_raw = demo.predict_proba(feature_frame.iloc[cal_idx])
        p_val_raw = demo.predict_proba(model_frame)
        p_val = PlattCalibrator().fit(p_cal_raw, y_cal).predict(p_val_raw)
        records.append(
            {"model": "naive_demographic", "calibration": "platt", "fold": fold_i, **_evaluate(y[val_idx], p_val)}
        )

        # logistic
        x_fit = x[fit_idx]
        x_cal = x[cal_idx]
        x_val = x[val_idx]
        x_fit_s, x_cal_s = _standardize(x_fit, x_cal)
        _, x_val_s = _standardize(x_fit, x_val)
        logit = LogisticRegressionGD(learning_rate=0.05, n_iter=400, l2=1e-3).fit(x_fit_s, y_fit)
        p_cal_raw = logit.predict_proba(x_cal_s)
        p_val_raw = logit.predict_proba(x_val_s)
        p_val = PlattCalibrator().fit(p_cal_raw, y_cal).predict(p_val_raw)
        records.append({"model": "logistic", "calibration": "platt", "fold": fold_i, **_evaluate(y[val_idx], p_val)})

        # lasso logistic
        lasso = LassoLogisticRegression(learning_rate=0.05, n_iter=500, l1=8e-4).fit(x_fit_s, y_fit)
        p_cal_raw = lasso.predict_proba(x_cal_s)
        p_val_raw = lasso.predict_proba(x_val_s)
        p_val = PlattCalibrator().fit(p_cal_raw, y_cal).predict(p_val_raw)
        records.append(
            {"model": "lasso_logistic", "calibration": "platt", "fold": fold_i, **_evaluate(y[val_idx], p_val)}
        )

    # aggregate across folds
    out = pd.DataFrame(records).groupby(["model", "calibration"], as_index=False).mean(numeric_only=True)

    # plot specs with a single fitted run (not CV-averaged) for quick visuals
    for model_name in out["model"].tolist():
        p_all = predict_full(model=model_name, calibration="platt")
        _save_calibration_plot_specs(model_name, "platt", y, p_all, df)

    return out.drop(columns=["fold"], errors="ignore")


def predict_full(model: str, calibration: str = "platt") -> np.ndarray:
    df = _add_age_group(load_compas_processed())
    target_col = _extract_target_column(df)
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
    feature_frame = df.drop(columns=[target_col]).copy()
    x_df, _ = _prepare_feature_matrix(feature_frame)
    x = x_df.to_numpy(dtype=float)

    # simple split for calibration visualization
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(df))
    fit_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    fit_idx, cal_idx = train_test_split(fit_idx, test_size=0.25, random_state=RANDOM_SEED, stratify=y[fit_idx])

    y_fit = y[fit_idx]
    y_cal = y[cal_idx]

    if model == "base_rate":
        base = BaseRateModel().fit(y_fit)
        p_cal_raw = base.predict_proba(len(cal_idx))
        p_test_raw = base.predict_proba(len(test_idx))
    elif model == "naive_demographic":
        demo = DemographicNaiveModel(group_cols=("sex", "race", "age_group")).fit(feature_frame.iloc[fit_idx], y_fit)
        p_cal_raw = demo.predict_proba(feature_frame.iloc[cal_idx])
        p_test_raw = demo.predict_proba(feature_frame.iloc[test_idx])
    elif model == "logistic":
        x_fit_s, x_cal_s = _standardize(x[fit_idx], x[cal_idx])
        _, x_test_s = _standardize(x[fit_idx], x[test_idx])
        logit = LogisticRegressionGD(learning_rate=0.05, n_iter=400, l2=1e-3).fit(x_fit_s, y_fit)
        p_cal_raw = logit.predict_proba(x_cal_s)
        p_test_raw = logit.predict_proba(x_test_s)
    elif model == "lasso_logistic":
        x_fit_s, x_cal_s = _standardize(x[fit_idx], x[cal_idx])
        _, x_test_s = _standardize(x[fit_idx], x[test_idx])
        lasso = LassoLogisticRegression(learning_rate=0.05, n_iter=500, l1=8e-4).fit(x_fit_s, y_fit)
        p_cal_raw = lasso.predict_proba(x_cal_s)
        p_test_raw = lasso.predict_proba(x_test_s)
    else:
        raise ValueError(f"Unknown model: {model}")

    if calibration == "platt":
        p_test = PlattCalibrator().fit(p_cal_raw, y_cal).predict(p_test_raw)
    elif calibration == "isotonic":
        p_test = IsotonicCalibrator().fit(p_cal_raw, y_cal).predict(p_test_raw)
    else:
        p_test = np.clip(p_test_raw, 1e-6, 1.0 - 1e-6)

    # return predictions aligned to test_idx, full-length with NaNs elsewhere for plotting not needed
    out = np.full(len(df), np.nan, dtype=float)
    out[test_idx] = p_test
    return np.nan_to_num(out, nan=float(np.mean(y)))


def run_xgb(n_trials: int = 16) -> pd.DataFrame:
    from sklearn.model_selection import train_test_split

    from src.models.xgb import feature_importance_table, shap_summary_table, train_xgb, tune_xgb

    df = _add_age_group(load_compas_processed())
    target_col = _extract_target_column(df)

    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
    feature_frame = df.drop(columns=[target_col]).copy()
    x_df, feature_names = _prepare_feature_matrix(feature_frame)
    x = x_df.to_numpy(dtype=float)

    idx = np.arange(len(df))
    fit_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    fit_idx, cal_idx = train_test_split(fit_idx, test_size=0.25, random_state=RANDOM_SEED, stratify=y[fit_idx])
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
    model = train_xgb(
        x_train=x[fit_idx],
        y_train=y[fit_idx],
        params=tune.best_params,
        seed=RANDOM_SEED,
    )

    y_test = y[test_idx]
    p_cal_raw = model.predict_proba(x[cal_idx])[:, 1]
    p_test_raw = model.predict_proba(x[test_idx])[:, 1]

    platt = PlattCalibrator().fit(p_cal_raw, y[cal_idx])
    isotonic = IsotonicCalibrator().fit(p_cal_raw, y[cal_idx])
    preds = {
        "raw": p_test_raw,
        "platt": platt.predict(p_test_raw),
        "isotonic": isotonic.predict(p_test_raw),
    }

    records: list[dict[str, float | str]] = []
    for cal_name, p_test in preds.items():
        records.append({"model": "xgboost", "calibration": cal_name, **_evaluate(y_test, p_test)})
        _save_calibration_plot_specs("xgboost", cal_name, y_test, p_test, df.iloc[test_idx])

    imp_rows = feature_importance_table(model, feature_names=feature_names, top_k=30)
    shap_rows = shap_summary_table(model, x_matrix=x[test_idx], feature_names=feature_names, top_k=30)
    (_plots_dir() / "xgb_importance_compas.json").write_text(json.dumps(imp_rows, indent=2))
    (_plots_dir() / "xgb_shap_compas.json").write_text(json.dumps(shap_rows, indent=2))

    return pd.DataFrame(records)


def write_report(baselines: pd.DataFrame | None, xgb: pd.DataFrame | None) -> Path:
    path = _reports_dir() / "compas_benchmark.md"
    lines = [
        "# COMPAS Benchmark (ProPublica Two-Year Sample)",
        "",
        "This benchmark is included as a fairness/evaluation case study; it is not directly comparable to NIJ.",
        "",
    ]

    if baselines is not None:
        lines.extend(
            [
                "## Baselines",
                "",
                "| Model | Calibration | Brier | AUROC | AUPRC | Log Loss | ECE |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in baselines.sort_values(["brier", "auroc"], ascending=[True, False]).iterrows():
            lines.append(
                "| "
                + f"{row['model']} | {row['calibration']} | {row['brier']:.5f} | {row['auroc']:.5f} | {row['auprc']:.5f} | {row['log_loss']:.5f} | {row['ece']:.5f} |"
            )
        lines.append("")

    if xgb is not None:
        lines.extend(
            [
                "## XGBoost",
                "",
                "| Model | Calibration | Brier | AUROC | AUPRC | Log Loss | ECE |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in xgb.sort_values(["brier", "auroc"], ascending=[True, False]).iterrows():
            lines.append(
                "| "
                + f"{row['model']} | {row['calibration']} | {row['brier']:.5f} | {row['auroc']:.5f} | {row['auprc']:.5f} | {row['log_loss']:.5f} | {row['ece']:.5f} |"
            )
        lines.append("")
        lines.append("XGB importance/SHAP tables are written under `reports/plots/` as JSON.")
        lines.append("")

    lines.append("Calibration plot specs are under `reports/plots/` as JSON (HTML-friendly).")
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["baselines", "xgb", "all"], default="all")
    parser.add_argument("--xgb-trials", type=int, default=int(os.getenv("PRECRIME_XGB_TRIALS", "16")))
    args = parser.parse_args()

    baselines = run_baselines() if args.task in {"baselines", "all"} else None
    xgb = run_xgb(n_trials=args.xgb_trials) if args.task in {"xgb", "all"} else None

    report = write_report(baselines, xgb)
    print(f"wrote compas report: {report}")


if __name__ == "__main__":
    main()

