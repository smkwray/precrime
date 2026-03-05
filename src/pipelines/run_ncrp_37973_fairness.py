"""Generate a fairness / subgroup audit for NCRP 37973 term-record benchmark.

This mirrors the COMPAS fairness report style, but uses the NCRP term-record-derived
reincarceration labels (return to prison) at year granularity.

Prereq: run `make ncrp-37973-terms-process` to generate `data/processed/*_y{1,2,3}.parquet`.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
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
from src.eval.plots import plot_calibration
from src.models.calibration import IsotonicCalibrator, PlattCalibrator


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

    # Ensure subgroup axes are string-like for reporting.
    for col in ("state", "sex", "race"):
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


def _group_split(groups: pd.Series, *, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
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


def _train_cal_test_split(*, groups: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fitcal_idx, test_idx = _group_split(groups, test_size=0.2, seed=RANDOM_SEED)
    fit_idx, cal_idx = _group_split(groups.iloc[fitcal_idx], test_size=0.25, seed=RANDOM_SEED + 1)
    return fitcal_idx[fit_idx], fitcal_idx[cal_idx], test_idx


def train_xgb_and_predict(
    *,
    x: np.ndarray,
    y: np.ndarray,
    groups: pd.Series,
    n_trials: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | str]]:
    from src.models.xgb import train_xgb, tune_xgb

    fit_idx, cal_idx, test_idx = _train_cal_test_split(groups=groups)

    # Tune on a subset of fit to reduce runtime (still grouped).
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
    model = train_xgb(x_train=x[fit_idx], y_train=y[fit_idx], params=tune.best_params, seed=RANDOM_SEED)

    p_cal_raw = model.predict_proba(x[cal_idx])[:, 1]
    p_test_raw = model.predict_proba(x[test_idx])[:, 1]

    platt = PlattCalibrator().fit(p_cal_raw, y[cal_idx])
    isotonic = IsotonicCalibrator().fit(p_cal_raw, y[cal_idx])
    preds = {
        "raw": np.clip(p_test_raw, 1e-6, 1.0 - 1e-6),
        "platt": platt.predict(p_test_raw),
        "isotonic": isotonic.predict(p_test_raw),
    }

    # Pick best calibration by test-set Brier.
    best_name = min(preds.keys(), key=lambda k: brier_score(y[test_idx].tolist(), preds[k].tolist()))
    best_pred = preds[best_name]

    metrics: dict[str, float | str] = {"calibration": str(best_name), **_evaluate(y[test_idx], best_pred)}
    return test_idx, best_pred, metrics


def _subgroup_table_lines(
    name: str,
    y_true: list[int],
    y_prob: list[float],
    groups: list[str],
    bootstrap_n: int = 200,
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

        summary = threshold_gap_summary(y_true, y_prob, group=groups)
        lines.extend(
            [
                "",
                f"- Max FPR gap across thresholds: {_format_float(float(summary.get('fpr_gap_max', float('nan'))))}",
                f"- Max FNR gap across thresholds: {_format_float(float(summary.get('fnr_gap_max', float('nan'))))}",
            ]
        )

    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id-mod", type=int, default=int(os.getenv("ID_MOD", "200")))
    parser.add_argument("--id-rem", type=int, default=int(os.getenv("ID_REM", "0")))
    parser.add_argument("--xgb-trials", type=int, default=int(os.getenv("PRECRIME_XGB_TRIALS", "16")))
    args = parser.parse_args()

    horizons = ["y1", "y2", "y3"]
    lines = [
        "# NCRP Fairness / Subgroup Audit (ICPSR 37973 term-record benchmark)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Variant: `terms_mod{args.id_mod}_r{args.id_rem}`.",
        "",
        "- Label family: **reincarceration** (return to prison)",
        "- Timing: **year granularity** (ADMITYR/RELEASEYR only in public-use extract)",
        "- Model: tuned **XGBoost** with post-hoc calibration; splits grouped by **ABT_INMATE_ID**",
        "",
    ]

    for horizon in horizons:
        df = _add_age_group(_sanitize_frame(load_horizon(horizon=horizon, id_mod=args.id_mod, id_rem=args.id_rem)))
        y = pd.to_numeric(df["y"], errors="coerce").fillna(0).astype(int).to_numpy()
        groups = df["ABT_INMATE_ID"].astype(str)

        feature_frame = df.drop(columns=["y"]).copy()
        x_df, _ = _prepare_feature_matrix(feature_frame)
        x = x_df.to_numpy(dtype=float)

        try:
            test_idx, p_test, metrics = train_xgb_and_predict(x=x, y=y, groups=groups, n_trials=int(args.xgb_trials))
        except Exception as exc:
            raise RuntimeError(f"XGBoost fairness run failed for {horizon}: {exc}") from exc

        y_test = y[test_idx]

        lines.extend(
            [
                f"## {horizon.upper()}",
                "",
                "- " + ", ".join([f"{k}={_format_float(float(v))}" if k != "calibration" else f"calibration={v}" for k, v in metrics.items()]),
                "",
            ]
        )

        # Plot specs (calibration by group).
        for group_name, col in (("race", "race"), ("sex", "sex"), ("age_group", "age_group")):
            group_vals = df.iloc[test_idx][col].fillna("Unknown").astype(str).tolist()
            fig = plot_calibration(
                y_true=y_test.astype(int).tolist(),
                y_prob=np.asarray(p_test, dtype=float).tolist(),
                group=group_vals,
                n_bins=10,
                title=f"NCRP 37973 Calibration {horizon} (XGBoost, {metrics['calibration']}) by {group_name}",
            )
            (_plots_dir() / f"fairness_ncrp37973_{horizon}_{group_name}_calibration.json").write_text(
                json.dumps(fig, indent=2)
            )

        # Threshold gap specs (race + sex).
        for group_name, col in (("race", "race"), ("sex", "sex")):
            group_vals = df.iloc[test_idx][col].fillna("Unknown").astype(str).tolist()
            summary = threshold_gap_summary(y_test.astype(int).tolist(), np.asarray(p_test, dtype=float).tolist(), group=group_vals)
            (_plots_dir() / f"fairness_ncrp37973_{horizon}_{group_name}_threshold_gap.json").write_text(
                json.dumps(summary, indent=2)
            )

        # Subgroup tables.
        lines.extend(
            _subgroup_table_lines(
                "By race",
                y_test.astype(int).tolist(),
                np.asarray(p_test, dtype=float).tolist(),
                df.iloc[test_idx]["race"].fillna("Unknown").astype(str).tolist(),
            )
        )
        lines.append("")
        lines.extend(
            _subgroup_table_lines(
                "By sex",
                y_test.astype(int).tolist(),
                np.asarray(p_test, dtype=float).tolist(),
                df.iloc[test_idx]["sex"].fillna("Unknown").astype(str).tolist(),
            )
        )
        lines.append("")
        lines.extend(
            _subgroup_table_lines(
                "By age group",
                y_test.astype(int).tolist(),
                np.asarray(p_test, dtype=float).tolist(),
                df.iloc[test_idx]["age_group"].fillna("Unknown").astype(str).tolist(),
            )
        )
        lines.append("")

        # Overall gap summary at threshold=0.5 (quick headline).
        race_vals = df.iloc[test_idx]["race"].fillna("Unknown").astype(str).tolist()
        lines.append(f"- FPR gap@0.5 (race): {_format_float(float(fpr_gap(y_test.tolist(), p_test.tolist(), race_vals, threshold=0.5)))}")
        lines.append(f"- FNR gap@0.5 (race): {_format_float(float(fnr_gap(y_test.tolist(), p_test.tolist(), race_vals, threshold=0.5)))}")
        lines.append("")

    out_path = _reports_dir() / "ncrp_37973_fairness_report.md"
    out_path.write_text("\n".join(lines))
    print(f"wrote ncrp fairness report: {out_path}")


if __name__ == "__main__":
    main()
