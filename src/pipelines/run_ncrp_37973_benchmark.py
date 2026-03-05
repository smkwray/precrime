"""Run NCRP (ICPSR 37973) benchmark pipelines on term-record-derived tables.

This benchmark is intentionally separate from the rearrest (NIJ/COMPAS) leaderboards:
the label family here is "reincarceration" (return to prison), and event timing is at
year granularity in the public-use extract.

Prereq: run `make ncrp-37973-terms-process` to generate `data/processed/*_y{1,2,3}.parquet`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.metrics import auprc, auroc, brier_score, expected_calibration_error, log_loss
from src.eval.plots import plot_calibration
from src.models.baselines import BaseRateModel, DemographicNaiveModel
from src.models.calibration import IsotonicCalibrator, PlattCalibrator
from src.models.lasso import LassoLogisticRegression
from src.models.logistic import LogisticRegressionGD


RANDOM_SEED = 42
DATASET_SLUG = "ncrp_icpsr_37973"


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


def _processed_dir() -> Path:
    path = _repo_root() / "data" / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _dataset_path(*, horizon: str, id_mod: int, id_rem: int) -> Path:
    suffix = f"terms_mod{id_mod}_r{id_rem}_{horizon}.parquet"
    return _processed_dir() / f"{DATASET_SLUG}_{suffix}"


def load_horizon(*, horizon: str, id_mod: int, id_rem: int) -> pd.DataFrame:
    path = _dataset_path(horizon=horizon, id_mod=id_mod, id_rem=id_rem)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing processed NCRP table: {path}\n"
            f"Run: make ncrp-37973-terms-process ID_MOD={id_mod} ID_REM={id_rem}"
        )
    return pd.read_parquet(path)


def _add_age_group(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    age = pd.to_numeric(out.get("age_at_release"), errors="coerce")
    out["age_group"] = pd.cut(
        age,
        bins=[-np.inf, 24, 34, 44, np.inf],
        labels=["<25", "25-34", "35-44", "45+"],
    ).astype("object")
    out["age_group"] = out["age_group"].fillna("Unknown").astype(str)
    return out


def _sanitize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out = out.drop(columns=["NEXT_ADMITYR"], errors="ignore")

    for col in ("MAND_PRISREL_YEAR", "PROJ_PRISREL_YEAR", "PARELIG_YEAR"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").mask(lambda s: s == 9999)

    categorical_cols = ("state", "sex", "race", "ADMTYPE", "EDUCATION", "OFFGENERAL", "OFFDETAIL", "RELTYPE")
    for col in categorical_cols:
        if col not in out.columns:
            continue
        series = out[col]
        if pd.api.types.is_numeric_dtype(series):
            series = series.astype("Int64").astype(str)
        else:
            series = series.astype(str)
        out[col] = series.replace({"<NA>": "Unknown", "nan": "Unknown"}).fillna("Unknown")
    return out


def _prepare_feature_matrix(feature_frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    model_frame = feature_frame.drop(columns=["ABT_INMATE_ID"], errors="ignore").copy()
    x_matrix = pd.get_dummies(model_frame, dummy_na=True)
    x_matrix = x_matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return x_matrix, x_matrix.columns.astype(str).tolist()


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


def _group_kfold_indices(groups: pd.Series, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    g = groups.astype(str).fillna("Unknown")
    unique = np.unique(g.to_numpy())
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)

    fold_sizes = np.full(n_folds, len(unique) // n_folds, dtype=int)
    fold_sizes[: len(unique) % n_folds] += 1

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    cursor = 0
    for fold_size in fold_sizes:
        val_groups = set(unique[cursor : cursor + fold_size].tolist())
        cursor += fold_size

        is_val = g.isin(val_groups).to_numpy()
        val_idx = np.where(is_val)[0]
        train_idx = np.where(~is_val)[0]
        splits.append((train_idx, val_idx))
    return splits


def _group_split(
    groups: pd.Series,
    *,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    g = groups.astype(str).fillna("Unknown")
    unique = np.unique(g.to_numpy())
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)

    cut = int(round((1.0 - float(test_size)) * len(unique)))
    cut = max(1, min(len(unique) - 1, cut))
    train_groups = set(unique[:cut].tolist())

    is_train = g.isin(train_groups).to_numpy()
    train_idx = np.where(is_train)[0]
    test_idx = np.where(~is_train)[0]
    return train_idx, test_idx


def _fit_calibration_group_split(train_idx: np.ndarray, groups: pd.Series, seed: int) -> tuple[np.ndarray, np.ndarray]:
    train_groups = groups.iloc[train_idx].astype(str).fillna("Unknown")
    unique = np.unique(train_groups.to_numpy())
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)

    cut = int(round(0.8 * len(unique)))
    cut = max(1, min(len(unique) - 1, cut))
    fit_groups = set(unique[:cut].tolist())

    is_fit = groups.astype(str).fillna("Unknown").isin(fit_groups).to_numpy()
    fit_idx = train_idx[is_fit[train_idx]]
    cal_idx = train_idx[~is_fit[train_idx]]
    return fit_idx, cal_idx


def _save_calibration_plot_specs(
    *,
    horizon: str,
    model_name: str,
    calibration_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    frame: pd.DataFrame,
) -> None:
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
            title=f"NCRP 37973 Calibration {horizon} {model_name} ({calibration_name}) by {group_name}",
        )
        out_path = _plots_dir() / f"calibration_ncrp37973_{horizon}_{model_name}_{calibration_name}_{group_name}.json"
        out_path.write_text(json.dumps(fig_spec, indent=2))


def predict_full(*, horizon: str, model: str, calibration: str, id_mod: int, id_rem: int) -> np.ndarray:
    df = _add_age_group(_sanitize_frame(load_horizon(horizon=horizon, id_mod=id_mod, id_rem=id_rem)))
    if "y" not in df.columns:
        raise ValueError("Expected `y` column in processed horizon table")

    y = pd.to_numeric(df["y"], errors="coerce").fillna(0).astype(int).to_numpy()
    groups = df["ABT_INMATE_ID"].astype(str)
    feature_frame = df.drop(columns=["y"]).copy()

    x_df, _ = _prepare_feature_matrix(feature_frame)
    x = x_df.to_numpy(dtype=float)

    # Grouped fit/cal/test split for a single calibration visualization run.
    fitcal_idx, test_idx = _group_split(groups, test_size=0.2, seed=RANDOM_SEED)
    fit_idx, cal_idx = _group_split(groups.iloc[fitcal_idx], test_size=0.25, seed=RANDOM_SEED + 1)
    fit_idx = fitcal_idx[fit_idx]
    cal_idx = fitcal_idx[cal_idx]

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

    out = np.full(len(df), np.nan, dtype=float)
    out[test_idx] = p_test
    return np.nan_to_num(out, nan=float(np.mean(y)))


def run_baselines(*, horizon: str, id_mod: int, id_rem: int, n_folds: int = 5) -> pd.DataFrame:
    df = _add_age_group(_sanitize_frame(load_horizon(horizon=horizon, id_mod=id_mod, id_rem=id_rem)))
    if "y" not in df.columns:
        raise ValueError("Expected `y` column in processed horizon table")

    y = pd.to_numeric(df["y"], errors="coerce").fillna(0).astype(int).to_numpy()
    groups = df["ABT_INMATE_ID"].astype(str)
    feature_frame = df.drop(columns=["y"]).copy()

    x_df, _ = _prepare_feature_matrix(feature_frame)
    x = x_df.to_numpy(dtype=float)

    splits = _group_kfold_indices(groups, n_folds=n_folds, seed=RANDOM_SEED)
    records: list[dict[str, float | str | int]] = []

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        fit_idx, cal_idx = _fit_calibration_group_split(train_idx, groups, seed=RANDOM_SEED + fold_i)
        y_fit = y[fit_idx]
        y_cal = y[cal_idx]

        # base rate
        base = BaseRateModel().fit(y_fit)
        p_cal_raw = base.predict_proba(len(cal_idx))
        p_val_raw = base.predict_proba(len(val_idx))
        p_val = PlattCalibrator().fit(p_cal_raw, y_cal).predict(p_val_raw)
        records.append(
            {"horizon": horizon, "model": "base_rate", "calibration": "platt", "fold": fold_i, **_evaluate(y[val_idx], p_val)}
        )

        # naive demographic base rate
        demo = DemographicNaiveModel(group_cols=("sex", "race", "age_group")).fit(feature_frame.iloc[fit_idx], y_fit)
        p_cal_raw = demo.predict_proba(feature_frame.iloc[cal_idx])
        p_val_raw = demo.predict_proba(feature_frame.iloc[val_idx])
        p_val = PlattCalibrator().fit(p_cal_raw, y_cal).predict(p_val_raw)
        records.append(
            {
                "horizon": horizon,
                "model": "naive_demographic",
                "calibration": "platt",
                "fold": fold_i,
                **_evaluate(y[val_idx], p_val),
            }
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
        records.append(
            {"horizon": horizon, "model": "logistic", "calibration": "platt", "fold": fold_i, **_evaluate(y[val_idx], p_val)}
        )

        # lasso logistic
        lasso = LassoLogisticRegression(learning_rate=0.05, n_iter=500, l1=8e-4).fit(x_fit_s, y_fit)
        p_cal_raw = lasso.predict_proba(x_cal_s)
        p_val_raw = lasso.predict_proba(x_val_s)
        p_val = PlattCalibrator().fit(p_cal_raw, y_cal).predict(p_val_raw)
        records.append(
            {
                "horizon": horizon,
                "model": "lasso_logistic",
                "calibration": "platt",
                "fold": fold_i,
                **_evaluate(y[val_idx], p_val),
            }
        )

    out = pd.DataFrame(records).groupby(["horizon", "model", "calibration"], as_index=False).mean(numeric_only=True)

    # Calibration plot specs (single run, grouped split).
    for model_name in out["model"].tolist():
        p_all = predict_full(horizon=horizon, model=str(model_name), calibration="platt", id_mod=id_mod, id_rem=id_rem)
        _save_calibration_plot_specs(
            horizon=horizon,
            model_name=str(model_name),
            calibration_name="platt",
            y_true=y,
            y_prob=p_all,
            frame=df,
        )

    return out.drop(columns=["fold"], errors="ignore")


def run_xgb(*, horizon: str, id_mod: int, id_rem: int, n_trials: int = 16) -> pd.DataFrame | None:
    try:
        from src.models.xgb import feature_importance_table, shap_summary_table, train_xgb, tune_xgb
    except Exception as exc:  # pragma: no cover
        print(f"[ncrp] skipping xgboost (install requirements-modeling.txt): {exc}")
        return None

    df = _add_age_group(_sanitize_frame(load_horizon(horizon=horizon, id_mod=id_mod, id_rem=id_rem)))
    if "y" not in df.columns:
        raise ValueError("Expected `y` column in processed horizon table")

    y = pd.to_numeric(df["y"], errors="coerce").fillna(0).astype(int).to_numpy()
    groups = df["ABT_INMATE_ID"].astype(str)
    feature_frame = df.drop(columns=["y"]).copy()
    x_df, feature_names = _prepare_feature_matrix(feature_frame)
    x = x_df.to_numpy(dtype=float)

    # Grouped fit/cal/test split.
    fitcal_idx, test_idx = _group_split(groups, test_size=0.2, seed=RANDOM_SEED)
    fit_idx, cal_idx = _group_split(groups.iloc[fitcal_idx], test_size=0.25, seed=RANDOM_SEED + 1)
    fit_idx = fitcal_idx[fit_idx]
    cal_idx = fitcal_idx[cal_idx]

    tune_train_idx, tune_val_idx = _group_split(groups.iloc[fit_idx], test_size=0.2, seed=RANDOM_SEED + 2)
    tune_train_idx = fit_idx[tune_train_idx]
    tune_val_idx = fit_idx[tune_val_idx]

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
        records.append({"horizon": horizon, "model": "xgboost", "calibration": cal_name, **_evaluate(y_test, p_test)})
        _save_calibration_plot_specs(
            horizon=horizon,
            model_name="xgboost",
            calibration_name=str(cal_name),
            y_true=y_test,
            y_prob=p_test,
            frame=df.iloc[test_idx],
        )

    imp_rows = feature_importance_table(model, feature_names=feature_names, top_k=30)
    shap_rows = shap_summary_table(model, x_matrix=x[test_idx], feature_names=feature_names, top_k=30)
    (_plots_dir() / f"xgb_importance_ncrp37973_{horizon}.json").write_text(json.dumps(imp_rows, indent=2))
    (_plots_dir() / f"xgb_shap_ncrp37973_{horizon}.json").write_text(json.dumps(shap_rows, indent=2))

    return pd.DataFrame(records)


def write_report(*, baselines: pd.DataFrame, xgb: pd.DataFrame | None, id_mod: int, id_rem: int) -> Path:
    path = _reports_dir() / "ncrp_37973_terms_benchmark.md"
    lines = [
        "# NCRP Benchmark (ICPSR 37973 selected variables; term-record-derived reincarceration)",
        "",
        f"Variant: `terms_mod{id_mod}_r{id_rem}`.",
        "",
        "- Label family: **reincarceration** (return to prison)",
        "- Timing: **year granularity** (ADMITYR/RELEASEYR only in public-use extract)",
        "- Splits: **grouped by ABT_INMATE_ID** (no person appears in both train/test)",
        "",
    ]

    def _table(df: pd.DataFrame) -> list[str]:
        out = [
            "| Horizon | Model | Calibration | Brier | AUROC | AUPRC | Log Loss | ECE |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
        for _, row in df.sort_values(["horizon", "brier", "auroc"], ascending=[True, True, False]).iterrows():
            out.append(
                "| "
                + f"{row['horizon']} | {row['model']} | {row['calibration']} | {row['brier']:.5f} | {row['auroc']:.5f} | {row['auprc']:.5f} | {row['log_loss']:.5f} | {row['ece']:.5f} |"
            )
        return out

    lines.extend(["## Baselines", ""])
    lines.extend(_table(baselines))
    lines.append("")

    if xgb is not None and not xgb.empty:
        lines.extend(["## XGBoost", ""])
        lines.extend(_table(xgb))
        lines.append("")
        lines.append("XGB importance/SHAP tables are written under `reports/plots/` as JSON.")
        lines.append("")
    else:
        lines.extend(
            [
                "## XGBoost",
                "",
                "(Skipped — install `requirements-modeling.txt` to run tuned XGBoost.)",
                "",
            ]
        )

    lines.append("Calibration plot specs are under `reports/plots/` as JSON (HTML-friendly).")
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["baselines", "xgb", "all"], default="all")
    parser.add_argument("--id-mod", type=int, default=int(os.getenv("ID_MOD", "200")))
    parser.add_argument("--id-rem", type=int, default=int(os.getenv("ID_REM", "0")))
    parser.add_argument("--xgb-trials", type=int, default=int(os.getenv("PRECRIME_XGB_TRIALS", "16")))
    args = parser.parse_args()

    horizons = ["y1", "y2", "y3"]
    baseline_frames: list[pd.DataFrame] = []
    xgb_frames: list[pd.DataFrame] = []

    if args.task in {"baselines", "all"}:
        for h in horizons:
            baseline_frames.append(run_baselines(horizon=h, id_mod=args.id_mod, id_rem=args.id_rem))

    xgb_out: pd.DataFrame | None = None
    if args.task in {"xgb", "all"}:
        for h in horizons:
            one = run_xgb(horizon=h, id_mod=args.id_mod, id_rem=args.id_rem, n_trials=args.xgb_trials)
            if one is not None:
                xgb_frames.append(one)

    baselines = pd.concat(baseline_frames, ignore_index=True) if baseline_frames else pd.DataFrame()
    if xgb_frames:
        xgb_out = pd.concat(xgb_frames, ignore_index=True)
    else:
        xgb_out = None

    report = write_report(baselines=baselines, xgb=xgb_out, id_mod=args.id_mod, id_rem=args.id_rem)
    print(f"wrote ncrp report: {report}")


if __name__ == "__main__":
    main()
