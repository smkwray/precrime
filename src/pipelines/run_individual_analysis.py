"""Profile individuals with lowest predictions and largest model errors.

Generates aggregate trait profiles (not row-level data) for:
  1. Lowest-predicted individuals (predicted probability closest to 0)
  2. Worst model errors — confident false negatives and confident false positives

Usage:
    python -m src.pipelines.run_individual_analysis
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.calibration import IsotonicCalibrator, PlattCalibrator


RANDOM_SEED = 42


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _reports_dir() -> Path:
    path = _repo_root() / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _apply_calibrator(name: str, p_cal: np.ndarray, y_cal: np.ndarray, p_eval: np.ndarray) -> np.ndarray:
    if name == "platt":
        return PlattCalibrator().fit(p_cal, y_cal).predict(p_eval)
    if name == "isotonic":
        return IsotonicCalibrator().fit(p_cal, y_cal).predict(p_eval)
    return np.clip(p_eval, 1e-6, 1.0 - 1e-6)


def _trait_profile(df: pd.DataFrame, profile_cols: list[str]) -> list[dict]:
    """Summarize trait distributions for a subset of individuals."""
    rows = []
    for col in profile_cols:
        if col not in df.columns:
            continue
        vc = df[col].astype(str).value_counts(normalize=True).head(5)
        for val, pct in vc.items():
            rows.append({"trait": col, "value": str(val), "pct": round(float(pct) * 100, 1), "n": int((vc * len(df)).get(val, 0))})
    return rows


def _profile_table_lines(profile: list[dict], title: str) -> list[str]:
    """Format trait profile as markdown table lines."""
    lines = [f"#### {title}", "", "| Trait | Value | % of Group | N |", "|---|---|---:|---:|"]
    for r in profile:
        lines.append(f"| {r['trait']} | {r['value']} | {r['pct']}% | {r['n']} |")
    lines.append("")
    return lines


def _summary_stats_lines(preds: np.ndarray, y: np.ndarray, label: str) -> list[str]:
    """Summary statistics for a prediction subset."""
    return [
        f"**{label}** — N={len(preds)}, "
        f"pred mean={np.mean(preds):.4f}, pred median={np.median(preds):.4f}, "
        f"pred min={np.min(preds):.4f}, pred max={np.max(preds):.4f}, "
        f"actual rearrest rate={np.mean(y):.3f}",
        "",
    ]


# ---------------------------------------------------------------------------
# NIJ analysis
# ---------------------------------------------------------------------------

def _run_nij_analysis() -> list[str]:
    from sklearn.model_selection import train_test_split

    from src.features.build_nij_static import build_static_datasets
    from src.features.build_nij_dynamic import build_dynamic_datasets
    from src.models.xgb import train_xgb

    best_cfg = json.loads((_repo_root() / "reports" / "xgb_best_models.json").read_text())
    all_sets = {"static": build_static_datasets(), "dynamic": build_dynamic_datasets()}

    nij_profile_cols = [
        "Age_at_Release", "Gender", "Race", "Gang_Affiliated",
        "Supervision_Risk_Score_First", "Prison_Years", "Prison_Offense",
        "Prior_Arrest_Episodes_Felony", "Prior_Arrest_Episodes_Property",
        "Prior_Arrest_Episodes_Violent", "Prior_Arrest_Episodes_Drug",
        "Condition_MH_SA", "Condition_Cog_Ed",
    ]

    lines = ["## NIJ Georgia Parole", ""]

    for horizon in sorted(best_cfg.keys()):
        cfg = best_cfg[horizon]
        dataset_key = str(cfg["dataset"])
        calibration_name = str(cfg["calibration"])
        best_params = dict(cfg["tuning"]["best_params"])

        ds = all_sets[dataset_key][horizon].copy()
        target_col = next(c for c in ("target", "y") if c in ds.columns)
        y = pd.to_numeric(ds[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
        feature_frame = ds.drop(columns=[target_col]).copy()

        x_df = pd.get_dummies(feature_frame.drop(columns=["ID"], errors="ignore"), dummy_na=True)
        x_df = x_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        x = x_df.to_numpy(dtype=float)

        idx = np.arange(len(ds))
        train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
        fit_idx, cal_idx = train_test_split(train_idx, test_size=0.25, random_state=RANDOM_SEED, stratify=y[train_idx])

        model = train_xgb(x_train=x[fit_idx], y_train=y[fit_idx], params=best_params, seed=RANDOM_SEED)
        p_cal = model.predict_proba(x[cal_idx])[:, 1]
        p_test_raw = model.predict_proba(x[test_idx])[:, 1]
        p_test = _apply_calibrator(calibration_name, p_cal, y[cal_idx], p_test_raw)

        y_test = y[test_idx]
        test_df = feature_frame.iloc[test_idx].copy()
        test_df["_pred"] = p_test
        test_df["_actual"] = y_test
        test_df["_error"] = (p_test - y_test) ** 2
        test_df["_signed_error"] = p_test - y_test

        lines.extend([f"### {horizon.upper()} (N_test={len(test_idx)}, base rate={np.mean(y_test):.3f})", ""])

        # --- Lowest predicted (bottom 5%) ---
        cutoff_low = np.percentile(p_test, 5)
        lowest = test_df[test_df["_pred"] <= cutoff_low].copy()
        lines.extend(_summary_stats_lines(lowest["_pred"].to_numpy(), lowest["_actual"].to_numpy(), f"Bottom 5% predicted (p ≤ {cutoff_low:.4f})"))
        lines.extend(_profile_table_lines(_trait_profile(lowest, nij_profile_cols), "Trait profile — lowest predicted"))

        # --- Lowest predicted (bottom 1%) ---
        cutoff_1pct = np.percentile(p_test, 1)
        lowest_1 = test_df[test_df["_pred"] <= cutoff_1pct].copy()
        lines.extend(_summary_stats_lines(lowest_1["_pred"].to_numpy(), lowest_1["_actual"].to_numpy(), f"Bottom 1% predicted (p ≤ {cutoff_1pct:.4f})"))
        lines.extend(_profile_table_lines(_trait_profile(lowest_1, nij_profile_cols), "Trait profile — bottom 1% predicted"))

        # --- Worst false negatives (actual=1 but lowest predicted) ---
        fn_df = test_df[test_df["_actual"] == 1].nsmallest(max(1, int(0.05 * test_df["_actual"].sum())), "_pred")
        lines.extend(_summary_stats_lines(fn_df["_pred"].to_numpy(), fn_df["_actual"].to_numpy(), f"Worst false negatives (top 5% of positives by lowest prediction)"))
        lines.extend(_profile_table_lines(_trait_profile(fn_df, nij_profile_cols), "Trait profile — worst false negatives"))

        # --- Worst false positives (actual=0 but highest predicted) ---
        fp_df = test_df[test_df["_actual"] == 0].nlargest(max(1, int(0.05 * (1 - test_df["_actual"]).sum())), "_pred")
        lines.extend(_summary_stats_lines(fp_df["_pred"].to_numpy(), fp_df["_actual"].to_numpy(), f"Worst false positives (top 5% of negatives by highest prediction)"))
        lines.extend(_profile_table_lines(_trait_profile(fp_df, nij_profile_cols), "Trait profile — worst false positives"))

        # --- Highest individual error ---
        worst_err = test_df.nlargest(max(1, int(0.05 * len(test_df))), "_error")
        lines.extend(_summary_stats_lines(worst_err["_pred"].to_numpy(), worst_err["_actual"].to_numpy(), f"Highest individual Brier error (top 5%)"))
        lines.extend(_profile_table_lines(_trait_profile(worst_err, nij_profile_cols), "Trait profile — highest individual error"))

    return lines


# ---------------------------------------------------------------------------
# COMPAS analysis
# ---------------------------------------------------------------------------

def _run_compas_analysis() -> list[str]:
    from sklearn.model_selection import train_test_split

    from src.features.build_compas import build_compas_2yr
    from src.models.xgb import train_xgb, tune_xgb

    df = build_compas_2yr()
    target_col = "y"
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
    feature_frame = df.drop(columns=[target_col]).copy()

    x_df = pd.get_dummies(feature_frame.drop(columns=["id"], errors="ignore"), dummy_na=True)
    x_df = x_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    x = x_df.to_numpy(dtype=float)

    idx = np.arange(len(df))
    fit_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    fit_idx, cal_idx = train_test_split(fit_idx, test_size=0.25, random_state=RANDOM_SEED, stratify=y[fit_idx])
    tune_train_idx, tune_val_idx = train_test_split(fit_idx, test_size=0.2, random_state=RANDOM_SEED, stratify=y[fit_idx])

    tune = tune_xgb(x_train=x[tune_train_idx], y_train=y[tune_train_idx], x_valid=x[tune_val_idx], y_valid=y[tune_val_idx], seed=RANDOM_SEED, n_trials=32)
    model = train_xgb(x_train=x[fit_idx], y_train=y[fit_idx], params=tune.best_params, seed=RANDOM_SEED)

    p_cal = model.predict_proba(x[cal_idx])[:, 1]
    p_test_raw = model.predict_proba(x[test_idx])[:, 1]
    p_test = _apply_calibrator("platt", p_cal, y[cal_idx], p_test_raw)

    y_test = y[test_idx]
    test_df = feature_frame.iloc[test_idx].copy()
    test_df["_pred"] = p_test
    test_df["_actual"] = y_test
    test_df["_error"] = (p_test - y_test) ** 2

    compas_profile_cols = ["age_cat", "sex", "race", "c_charge_degree", "priors_count", "score_text", "v_score_text"]

    lines = [
        "## COMPAS (Broward County, 2-year)", "",
        f"N_test={len(test_idx)}, base rate={np.mean(y_test):.3f}", "",
    ]

    cutoff_low = np.percentile(p_test, 5)
    lowest = test_df[test_df["_pred"] <= cutoff_low]
    lines.extend(_summary_stats_lines(lowest["_pred"].to_numpy(), lowest["_actual"].to_numpy(), f"Bottom 5% predicted (p ≤ {cutoff_low:.4f})"))
    lines.extend(_profile_table_lines(_trait_profile(lowest, compas_profile_cols), "Trait profile — lowest predicted"))

    fn_df = test_df[test_df["_actual"] == 1].nsmallest(max(1, int(0.05 * test_df["_actual"].sum())), "_pred")
    lines.extend(_summary_stats_lines(fn_df["_pred"].to_numpy(), fn_df["_actual"].to_numpy(), f"Worst false negatives"))
    lines.extend(_profile_table_lines(_trait_profile(fn_df, compas_profile_cols), "Trait profile — worst false negatives"))

    fp_df = test_df[test_df["_actual"] == 0].nlargest(max(1, int(0.05 * (1 - test_df["_actual"]).sum())), "_pred")
    lines.extend(_summary_stats_lines(fp_df["_pred"].to_numpy(), fp_df["_actual"].to_numpy(), f"Worst false positives"))
    lines.extend(_profile_table_lines(_trait_profile(fp_df, compas_profile_cols), "Trait profile — worst false positives"))

    worst_err = test_df.nlargest(max(1, int(0.05 * len(test_df))), "_error")
    lines.extend(_summary_stats_lines(worst_err["_pred"].to_numpy(), worst_err["_actual"].to_numpy(), "Highest individual Brier error (top 5%)"))
    lines.extend(_profile_table_lines(_trait_profile(worst_err, compas_profile_cols), "Trait profile — highest individual error"))

    return lines


# ---------------------------------------------------------------------------
# NCRP analysis
# ---------------------------------------------------------------------------

def _run_ncrp_analysis(*, id_mod: int, id_rem: int) -> list[str]:
    from src.models.xgb import train_xgb, tune_xgb
    from src.pipelines.run_ncrp_37973_benchmark import (
        _add_age_group,
        _group_split,
        _prepare_feature_matrix,
        _sanitize_frame,
        load_horizon,
    )

    ncrp_profile_cols = ["sex", "race", "age_group", "ADMTYPE", "RELTYPE", "OFFGENERAL", "state"]

    lines = ["## NCRP (ICPSR 37973, reincarceration)", ""]

    for horizon in ["y1"]:
        df = _add_age_group(_sanitize_frame(load_horizon(horizon=horizon, id_mod=id_mod, id_rem=id_rem)))
        y = pd.to_numeric(df["y"], errors="coerce").fillna(0).astype(int).to_numpy()
        groups = df["ABT_INMATE_ID"].astype(str)
        feature_frame = df.drop(columns=["y"]).copy()

        x_df, _ = _prepare_feature_matrix(feature_frame)
        x = x_df.to_numpy(dtype=float)

        fitcal_idx, test_idx = _group_split(groups, test_size=0.2, seed=RANDOM_SEED)
        fit_idx, cal_idx = _group_split(groups.iloc[fitcal_idx], test_size=0.25, seed=RANDOM_SEED + 1)
        fit_idx = fitcal_idx[fit_idx]
        cal_idx = fitcal_idx[cal_idx]

        tune_train_idx, tune_val_idx = _group_split(groups.iloc[fit_idx], test_size=0.2, seed=RANDOM_SEED + 2)
        tune_train_idx = fit_idx[tune_train_idx]
        tune_val_idx = fit_idx[tune_val_idx]

        tune = tune_xgb(x_train=x[tune_train_idx], y_train=y[tune_train_idx], x_valid=x[tune_val_idx], y_valid=y[tune_val_idx], seed=RANDOM_SEED, n_trials=32)
        model = train_xgb(x_train=x[fit_idx], y_train=y[fit_idx], params=tune.best_params, seed=RANDOM_SEED)

        p_cal = model.predict_proba(x[cal_idx])[:, 1]
        p_test_raw = model.predict_proba(x[test_idx])[:, 1]
        p_test = _apply_calibrator("isotonic", p_cal, y[cal_idx], p_test_raw)

        y_test = y[test_idx]
        test_df = feature_frame.iloc[test_idx].copy()
        test_df["_pred"] = p_test
        test_df["_actual"] = y_test
        test_df["_error"] = (p_test - y_test) ** 2

        lines.extend([f"### {horizon.upper()} (N_test={len(test_idx)}, base rate={np.mean(y_test):.3f})", ""])

        cutoff_low = np.percentile(p_test, 5)
        lowest = test_df[test_df["_pred"] <= cutoff_low]
        lines.extend(_summary_stats_lines(lowest["_pred"].to_numpy(), lowest["_actual"].to_numpy(), f"Bottom 5% predicted (p ≤ {cutoff_low:.4f})"))
        lines.extend(_profile_table_lines(_trait_profile(lowest, ncrp_profile_cols), "Trait profile — lowest predicted"))

        fn_df = test_df[test_df["_actual"] == 1].nsmallest(max(1, int(0.05 * test_df["_actual"].sum())), "_pred")
        lines.extend(_summary_stats_lines(fn_df["_pred"].to_numpy(), fn_df["_actual"].to_numpy(), "Worst false negatives"))
        lines.extend(_profile_table_lines(_trait_profile(fn_df, ncrp_profile_cols), "Trait profile — worst false negatives"))

        fp_df = test_df[test_df["_actual"] == 0].nlargest(max(1, int(0.05 * (1 - test_df["_actual"]).sum())), "_pred")
        lines.extend(_summary_stats_lines(fp_df["_pred"].to_numpy(), fp_df["_actual"].to_numpy(), "Worst false positives"))
        lines.extend(_profile_table_lines(_trait_profile(fp_df, ncrp_profile_cols), "Trait profile — worst false positives"))

        worst_err = test_df.nlargest(max(1, int(0.05 * len(test_df))), "_error")
        lines.extend(_summary_stats_lines(worst_err["_pred"].to_numpy(), worst_err["_actual"].to_numpy(), "Highest individual Brier error (top 5%)"))
        lines.extend(_profile_table_lines(_trait_profile(worst_err, ncrp_profile_cols), "Trait profile — highest individual error"))

    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_report(nij_lines: list[str], compas_lines: list[str], ncrp_lines: list[str]) -> Path:
    lines = [
        "# Individual Prediction Analysis",
        "",
        "Profiles of individuals with the **lowest model-predicted probability** (closest to 0) "
        "and individuals where the **model performs worst** (largest prediction errors).",
        "",
        "All results are from the held-out test set (20% split, seed 42). "
        "Trait profiles show the top-5 most common values for each characteristic within the subset. "
        "These are **aggregate summaries** — no individual-level predictions are exported.",
        "",
        "> **Terminology:**",
        "> - *Lowest predicted*: individuals the model assigns the smallest rearrest/reincarceration probability.",
        "> - *Worst false negatives*: individuals who **were** rearrested but had the **lowest** predicted probability (model missed them).",
        "> - *Worst false positives*: individuals who were **not** rearrested but had the **highest** predicted probability (model was wrong about them).",
        "> - *Highest individual error*: individuals with the largest (p − y)² regardless of direction.",
        "",
        "---",
        "",
    ]
    lines.extend(nij_lines)
    lines.extend(["", "---", ""])
    lines.extend(compas_lines)
    lines.extend(["", "---", ""])
    lines.extend(ncrp_lines)
    lines.extend([
        "",
        "---",
        "",
        "*Generated by `src/pipelines/run_individual_analysis.py`. "
        "All caveats about observational data, label definitions, and generalizability apply.*",
    ])

    out_path = _reports_dir() / "individual_analysis.md"
    out_path.write_text("\n".join(lines))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Individual prediction analysis")
    parser.add_argument("--id-mod", type=int, default=int(os.getenv("ID_MOD", "200")))
    parser.add_argument("--id-rem", type=int, default=int(os.getenv("ID_REM", "0")))
    parser.add_argument("--dataset", choices=["nij", "compas", "ncrp", "all"], default="all")
    args = parser.parse_args()

    nij_lines = _run_nij_analysis() if args.dataset in ("nij", "all") else []
    compas_lines = _run_compas_analysis() if args.dataset in ("compas", "all") else []
    ncrp_lines = _run_ncrp_analysis(id_mod=args.id_mod, id_rem=args.id_rem) if args.dataset in ("ncrp", "all") else []

    report_path = write_report(nij_lines, compas_lines, ncrp_lines)
    print(f"wrote individual analysis report: {report_path}")


if __name__ == "__main__":
    main()
