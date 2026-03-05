"""Run a dependency-free model sweep on NIJ tasks and write an aggregate report.

The goal is to compare several model families under:
- Overall probabilistic metrics (Brier, AUROC, AUPRC, ECE, log loss)
- NIJ-style terms: sex-specific Brier and "FairAcc" index components.

By default, this runs:
- Y1 on the static feature set
- Y2/Y3 on the dynamic feature set
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
from src.eval.nij_scoring import brier_by_sex, fpr_by_race_at_threshold, nij_fair_and_accurate, nij_fp_term
from src.features.build_nij_dynamic import build_dynamic_datasets
from src.features.build_nij_static import build_static_datasets
from src.models.calibration import IsotonicCalibrator, PlattCalibrator
from src.models.logistic import LogisticRegressionGD
from src.models.xgb import train_xgb, tune_xgb


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


def _n_jobs() -> int:
    cap = int(os.getenv("PRECRIME_N_JOBS", "16"))
    detected = os.cpu_count() or 4
    return max(1, min(cap, int(detected)))


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


def _no_race_columns(frame: pd.DataFrame) -> pd.DataFrame:
    cols = [
        c
        for c in frame.columns
        if c != "Race" and not str(c).startswith("Race_") and "race" not in str(c).lower()
    ]
    return frame[cols].copy()


def _split_train_cal_test(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    fit_idx, cal_idx = train_test_split(
        train_idx,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=y[train_idx],
    )
    return fit_idx, cal_idx, test_idx


def _standardize(x_train: np.ndarray, x_other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.mean(x_train, axis=0)
    sigma = np.std(x_train, axis=0)
    sigma[sigma == 0.0] = 1.0
    return (x_train - mu) / sigma, (x_other - mu) / sigma


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


def _nij_terms(
    y_true: list[int],
    y_prob: list[float],
    sex: list[str],
    race: list[str],
) -> dict[str, float]:
    # Sex-specific Brier.
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


def _calibrate(calibration: str, p_cal_raw: np.ndarray, y_cal: np.ndarray, p_test_raw: np.ndarray) -> np.ndarray:
    if calibration == "platt":
        return PlattCalibrator().fit(p_cal_raw, y_cal).predict(p_test_raw)
    if calibration == "isotonic":
        return IsotonicCalibrator().fit(p_cal_raw, y_cal).predict(p_test_raw)
    return np.clip(p_test_raw, 1e-6, 1.0 - 1e-6)


def _fit_predict_logistic(
    x: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_fit_s, x_cal_s = _standardize(x[fit_idx], x[cal_idx])
    _, x_test_s = _standardize(x[fit_idx], x[test_idx])
    model = LogisticRegressionGD(learning_rate=0.05, n_iter=400, l2=1e-3).fit(x_fit_s, y[fit_idx])
    p_cal = model.predict_proba(x_cal_s)
    p_test = model.predict_proba(x_test_s)
    return p_cal, p_test


def _fit_predict_rf(
    x: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

    if kind == "rf":
        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
            n_jobs=_n_jobs(),
        )
    elif kind == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
            n_jobs=_n_jobs(),
        )
    else:
        raise ValueError(f"Unknown kind: {kind}")

    model.fit(x[fit_idx], y[fit_idx])
    p_cal = model.predict_proba(x[cal_idx])[:, 1]
    p_test = model.predict_proba(x[test_idx])[:, 1]
    return np.clip(p_cal, 1e-6, 1.0 - 1e-6), np.clip(p_test, 1e-6, 1.0 - 1e-6)


def _fit_predict_histgb(
    x: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=300,
        random_state=RANDOM_SEED,
    )
    model.fit(x[fit_idx], y[fit_idx])
    p_cal = model.predict_proba(x[cal_idx])[:, 1]
    p_test = model.predict_proba(x[test_idx])[:, 1]
    return np.clip(p_cal, 1e-6, 1.0 - 1e-6), np.clip(p_test, 1e-6, 1.0 - 1e-6)


def _fit_predict_xgb_best(
    x: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    params: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    model = train_xgb(x_train=x[fit_idx], y_train=y[fit_idx], params=params, seed=RANDOM_SEED)
    p_cal = model.predict_proba(x[cal_idx])[:, 1]
    p_test = model.predict_proba(x[test_idx])[:, 1]
    return np.clip(p_cal, 1e-6, 1.0 - 1e-6), np.clip(p_test, 1e-6, 1.0 - 1e-6)


def _fit_predict_xgb_retuned(
    x: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    n_trials: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    from sklearn.model_selection import train_test_split

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

    p_cal, p_test = _fit_predict_xgb_best(
        x=x,
        y=y,
        fit_idx=fit_idx,
        cal_idx=cal_idx,
        test_idx=test_idx,
        params=dict(tune.best_params),
    )
    return p_cal, p_test, dict(tune.best_params)


def write_model_sweep_report(
    best_models_path: Path | None = None,
    out_md: Path | None = None,
    out_json: Path | None = None,
    xgb_trials: int = 128,
    retune_horizons: set[str] | None = None,
    include_without_race: bool = False,
    render_only: bool = False,
) -> tuple[Path, Path]:
    best = _load_best_models(best_models_path)
    out_md = out_md or (_reports_dir() / "model_sweep.md")
    out_json = out_json or (_reports_dir() / "model_sweep.json")

    all_sets = {
        "static": build_static_datasets(),
        "dynamic": build_dynamic_datasets(),
    }

    default_tasks = [
        ("y1", "static"),
        ("y2", "dynamic"),
        ("y3", "dynamic"),
    ]

    if retune_horizons is None:
        retune_horizons = {"y1"}

    if render_only:
        if not out_json.exists():
            raise FileNotFoundError(f"Missing {out_json}. Run this pipeline once without --render-only first.")
        records = json.loads(out_json.read_text())
        df = pd.DataFrame(records)
        return _render_model_sweep_md(df=df, out_md=out_md, out_json=out_json)

    records: list[dict[str, float | str | int]] = []

    for horizon, dataset_key in default_tasks:
        ds = _add_age_group(all_sets[dataset_key][horizon])
        target_col = _extract_target_column(ds)
        y = pd.to_numeric(ds[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
        feature_frame = ds.drop(columns=[target_col]).copy()
        fit_idx, cal_idx, test_idx = _split_train_cal_test(y)

        test_frame = ds.iloc[test_idx].copy()
        y_test = y[test_idx].astype(int).tolist()
        sex_vals = test_frame["Gender"].fillna("Unknown").astype(str).tolist() if "Gender" in test_frame else ["Unknown"] * len(test_frame)
        race_vals = test_frame["Race"].fillna("Unknown").astype(str).tolist() if "Race" in test_frame else ["Unknown"] * len(test_frame)

        variants = [("with_race", feature_frame)]
        if include_without_race:
            variants.append(("without_race", _no_race_columns(feature_frame)))

        for variant_name, variant_features in variants:
            x_df = _prepare_feature_matrix(variant_features)
            x = x_df.to_numpy(dtype=float)

            def add_record(model: str, calibration: str, p_test: np.ndarray, p_cal_raw: np.ndarray, y_cal: np.ndarray, extra: dict[str, object] | None = None) -> None:
                p = _calibrate(calibration, p_cal_raw=p_cal_raw, y_cal=y_cal, p_test_raw=p_test)
                overall = _evaluate_overall(y[test_idx], p)
                nij = _nij_terms(y_test, p.astype(float).tolist(), sex_vals, race_vals)
                row: dict[str, float | str | int] = {
                    "horizon": horizon,
                    "dataset": dataset_key,
                    "variant": variant_name,
                    "model": model,
                    "calibration": calibration,
                    "brier": float(overall["brier"]),
                    "auroc": float(overall["auroc"]),
                    "auprc": float(overall["auprc"]),
                    "log_loss": float(overall["log_loss"]),
                    "ece": float(overall["ece"]),
                    "bs_m": float(nij["bs_m"]),
                    "bs_f": float(nij["bs_f"]),
                    "bs_sex_avg": float(nij["bs_sex_avg"]),
                    "fairacc_m": float(nij["fairacc_m"]),
                    "fairacc_f": float(nij["fairacc_f"]),
                    "fairacc_sex_avg": float(nij["fairacc_sex_avg"]),
                }
                if extra:
                    for k, v in extra.items():
                        row[k] = v  # type: ignore[assignment]
                records.append(row)

            # Logistic (raw + calibrated variants).
            p_cal_logit, p_test_logit = _fit_predict_logistic(x, y, fit_idx, cal_idx, test_idx)
            for cal_name in ("raw", "platt", "isotonic"):
                add_record("logistic_gd", cal_name, p_test_logit, p_cal_logit, y[cal_idx])

            # RF / ExtraTrees / HistGB.
            p_cal_rf, p_test_rf = _fit_predict_rf(x, y, fit_idx, cal_idx, test_idx, kind="rf")
            for cal_name in ("raw", "platt", "isotonic"):
                add_record("random_forest", cal_name, p_test_rf, p_cal_rf, y[cal_idx])

            p_cal_et, p_test_et = _fit_predict_rf(x, y, fit_idx, cal_idx, test_idx, kind="extra_trees")
            for cal_name in ("raw", "platt", "isotonic"):
                add_record("extra_trees", cal_name, p_test_et, p_cal_et, y[cal_idx])

            p_cal_hg, p_test_hg = _fit_predict_histgb(x, y, fit_idx, cal_idx, test_idx)
            for cal_name in ("raw", "platt", "isotonic"):
                add_record("hist_gb", cal_name, p_test_hg, p_cal_hg, y[cal_idx])

            # XGB baseline = best models JSON for this horizon.
            cfg = best[horizon]
            best_params = dict(cfg.get("tuning", {}).get("best_params", {}))
            p_cal_xgb, p_test_xgb = _fit_predict_xgb_best(x, y, fit_idx, cal_idx, test_idx, params=best_params)
            add_record(
                "xgb_best",
                str(cfg.get("calibration", "raw")),
                p_test_xgb,
                p_cal_xgb,
                y[cal_idx],
                extra={"xgb_trials": int(cfg.get("tuning", {}).get("n_trials", 0))},
            )

            # Retuned XGB (optional per horizon).
            if horizon in retune_horizons:
                p_cal_rt, p_test_rt, rt_params = _fit_predict_xgb_retuned(
                    x=x,
                    y=y,
                    fit_idx=fit_idx,
                    cal_idx=cal_idx,
                    test_idx=test_idx,
                    n_trials=xgb_trials,
                )
                (_reports_dir() / f"xgb_retuned_{dataset_key}_{horizon}.json").write_text(
                    json.dumps({"n_trials": xgb_trials, "best_params": rt_params}, indent=2)
                )
                for cal_name in ("raw", "platt", "isotonic"):
                    add_record(
                        "xgb_retuned",
                        cal_name,
                        p_test_rt,
                        p_cal_rt,
                        y[cal_idx],
                        extra={"xgb_trials": int(xgb_trials)},
                    )

    out_json.write_text(json.dumps(records, indent=2))
    df = pd.DataFrame(records)
    return _render_model_sweep_md(df=df, out_md=out_md, out_json=out_json)


def _render_model_sweep_md(df: pd.DataFrame, out_md: Path, out_json: Path) -> tuple[Path, Path]:
    # Sort mainly by sex-avg Brier, then overall Brier.
    ordered = df.sort_values(["horizon", "variant", "bs_sex_avg", "brier"], ascending=[True, True, True, True]).copy()

    lines: list[str] = [
        "# NIJ Model Sweep (Dependency-Free)",
        "",
        f"Generated: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}",
        "",
        "This sweep compares several model families on a fixed seeded split (seed 42).",
        "Primary sorting is by sex-average Brier (NIJ-style accuracy view).",
        "A secondary NIJ-style view (“FairAcc”) is included to show how rankings can change if you factor in an explicit parity term.",
        "",
        "Key NIJ-style terms:",
        "- `bs_sex_avg`: average of male/female Brier errors (lower is better).",
        "- `fairacc_sex_avg`: average of male/female FairAcc indices (higher is better).",
        "",
    ]

    for (horizon, variant), block_unsorted in ordered.groupby(["horizon", "variant"], sort=False):
        block = block_unsorted.sort_values(["bs_sex_avg", "brier"], ascending=[True, True]).copy()
        block_fair = block_unsorted.sort_values(
            ["fairacc_sex_avg", "bs_sex_avg", "brier"],
            ascending=[False, True, True],
        ).copy()

        top_acc = block.iloc[0]
        top_fair = block_fair.iloc[0]
        top_note = f"Top by prediction (`bs_sex_avg`): `{top_acc['model']} ({top_acc['calibration']})`."
        if str(top_acc["model"]) != str(top_fair["model"]) or str(top_acc["calibration"]) != str(top_fair["calibration"]):
            top_note += f" Top by FairAcc: `{top_fair['model']} ({top_fair['calibration']})`."

        lines.extend(
            [
                f"## {str(horizon).upper()} — {str(variant)}",
                "",
                top_note,
                "",
                "| Dataset | Model | Calibration | bs_sex_avg | fairacc_sex_avg | Overall Brier | AUROC | AUPRC | ECE |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in block.iterrows():
            lines.append(
                "| "
                + f"{row['dataset']} | {row['model']} | {row['calibration']} | "
                + f"{float(row['bs_sex_avg']):.5f} | {float(row['fairacc_sex_avg']):.5f} | {float(row['brier']):.5f} | "
                + f"{float(row['auroc']):.5f} | {float(row['auprc']):.5f} | {float(row['ece']):.5f} |"
            )
        lines.append("")

        lines.extend(
            [
                "### Alternate ranking (FairAcc-first; NIJ-style “fair-and-accurate” view)",
                "",
                "This is the same block sorted by `fairacc_sex_avg` (descending).",
                "",
                "| Dataset | Model | Calibration | fairacc_sex_avg | bs_sex_avg | Overall Brier | AUROC | AUPRC | ECE |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in block_fair.head(8).iterrows():
            lines.append(
                "| "
                + f"{row['dataset']} | {row['model']} | {row['calibration']} | "
                + f"{float(row['fairacc_sex_avg']):.5f} | {float(row['bs_sex_avg']):.5f} | {float(row['brier']):.5f} | "
                + f"{float(row['auroc']):.5f} | {float(row['auprc']):.5f} | {float(row['ece']):.5f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- These are *not* official NIJ leaderboard scores (different test set).",
            "- `xgb_best` uses the params from `reports/xgb_best_models.json`.",
            "- `xgb_retuned` trials are controlled by `--xgb-trials` (default 128) and are run only for selected horizons.",
            "",
            f"Machine-readable metrics: `{out_json.name}`",
        ]
    )

    out_md.write_text("\n".join(lines))
    return out_md, out_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-models", type=Path, default=None, help="Path to xgb_best_models.json")
    parser.add_argument("--out-md", type=Path, default=None, help="Output markdown path")
    parser.add_argument("--out-json", type=Path, default=None, help="Output json path")
    parser.add_argument("--xgb-trials", type=int, default=int(os.getenv("PRECRIME_XGB_TRIALS", "128")))
    parser.add_argument(
        "--retune-horizons",
        type=str,
        default=os.getenv("PRECRIME_RETUNE_HORIZONS", "y1"),
        help="Comma-separated horizons to retune XGB for (default: y1).",
    )
    parser.add_argument("--include-without-race", action="store_true", help="Also run variants excluding Race from training.")
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Skip model training and only render markdown from an existing JSON metrics file.",
    )
    args = parser.parse_args()

    retune = {h.strip() for h in str(args.retune_horizons).split(",") if h.strip()}

    out_md, out_json = write_model_sweep_report(
        best_models_path=args.best_models,
        out_md=args.out_md,
        out_json=args.out_json,
        xgb_trials=int(args.xgb_trials),
        retune_horizons=retune,
        include_without_race=bool(args.include_without_race),
        render_only=bool(args.render_only),
    )
    print(f"wrote model sweep markdown: {out_md}")
    print(f"wrote model sweep json: {out_json}")


if __name__ == "__main__":
    main()
