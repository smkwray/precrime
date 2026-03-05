"""Seed-sweep stability report for NIJ best models.

Runs the best-per-horizon model config across multiple random seeds (split + fit)
and summarizes variability in aggregate metrics and a simple race error-rate gap
under a top-k selection policy.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.metrics import auprc, auroc, brier_score, expected_calibration_error, log_loss
from src.features.build_nij_dynamic import build_dynamic_datasets
from src.features.build_nij_static import build_static_datasets
from src.models.calibration import IsotonicCalibrator, PlattCalibrator
from src.models.xgb import train_xgb


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


def _extract_target_column(ds: pd.DataFrame) -> str:
    for candidate in ("y", "target"):
        if candidate in ds.columns:
            return candidate
    raise ValueError("Unable to detect target column in dataset")


def _prepare_feature_matrix(feature_frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    model_frame = feature_frame.drop(columns=["ID"], errors="ignore")
    x_matrix = pd.get_dummies(model_frame, dummy_na=True)
    x_matrix = x_matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return x_matrix, x_matrix.columns.astype(str).tolist()


def _split_train_cal_test(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
    fit_idx, cal_idx = train_test_split(train_idx, test_size=0.25, random_state=seed, stratify=y[train_idx])
    return fit_idx, cal_idx, test_idx


def _train_predict_xgb(
    x: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    params: dict[str, object],
    calibration: str,
    seed: int,
) -> np.ndarray:
    model = train_xgb(x_train=x[fit_idx], y_train=y[fit_idx], params=params, seed=seed)
    p_cal_raw = model.predict_proba(x[cal_idx])[:, 1]
    p_test_raw = model.predict_proba(x[test_idx])[:, 1]

    if calibration == "platt":
        return PlattCalibrator().fit(p_cal_raw, y[cal_idx]).predict(p_test_raw)
    if calibration == "isotonic":
        return IsotonicCalibrator().fit(p_cal_raw, y[cal_idx]).predict(p_test_raw)
    return np.clip(p_test_raw, 1e-6, 1.0 - 1e-6)


def _binary_rates(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y = y_true.astype(int)
    pred = (y_prob >= threshold).astype(int)
    tp = float(np.sum((pred == 1) & (y == 1)))
    fp = float(np.sum((pred == 1) & (y == 0)))
    tn = float(np.sum((pred == 0) & (y == 0)))
    fn = float(np.sum((pred == 0) & (y == 1)))
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    fnr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    return {"fpr": fpr, "fnr": fnr, "ppv": ppv}


def _threshold_top_k(y_prob: np.ndarray, k: float) -> float:
    n = len(y_prob)
    m = int(np.ceil(k * n))
    if m <= 0:
        return 1.0
    if m >= n:
        return 0.0
    return float(np.partition(y_prob, n - m)[n - m])


def _format_float(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value:.5f}"


def write_stability_report(best_models_path: Path | None = None, out_path: Path | None = None, seeds: list[int] | None = None) -> Path:
    seeds = seeds or [42, 43, 44]
    best = _load_best_models(best_models_path)
    out = out_path or (_reports_dir() / "stability_eval.md")

    all_sets = {"static": build_static_datasets(), "dynamic": build_dynamic_datasets()}

    lines: list[str] = [
        "# Stability Evaluation (Seed Sweep)",
        "",
        f"Generated: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}",
        "",
        "This report reruns the best-per-horizon NIJ model configs across multiple random seeds (split + fit) and summarizes variability.",
        "",
        f"Seeds: `{', '.join(str(s) for s in seeds)}`.",
        "",
        "Policy for error-rate summary: `top10%` (flag top 10% highest predicted risks on the test set).",
        "",
    ]

    for horizon in sorted(best.keys()):
        cfg = best[horizon]
        dataset_key = str(cfg["dataset"])
        calibration = str(cfg["calibration"])
        params = cfg.get("tuning", {}).get("best_params", {})

        ds = all_sets[dataset_key][horizon].copy()
        target_col = _extract_target_column(ds)
        y_full = pd.to_numeric(ds[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
        feature_frame = ds.drop(columns=[target_col]).copy()

        x_df, _ = _prepare_feature_matrix(feature_frame)
        x_full = x_df.to_numpy(dtype=float)

        rows: list[dict[str, float]] = []
        for seed in seeds:
            fit_idx, cal_idx, test_idx = _split_train_cal_test(y_full, seed=seed)
            p_test = _train_predict_xgb(
                x=x_full,
                y=y_full,
                fit_idx=fit_idx,
                cal_idx=cal_idx,
                test_idx=test_idx,
                params=dict(params),
                calibration=calibration,
                seed=seed,
            )
            y_test = y_full[test_idx].astype(int)
            metrics = {
                "brier": float(brier_score(y_test.tolist(), p_test.tolist())),
                "auroc": float(auroc(y_test.tolist(), p_test.tolist())),
                "auprc": float(auprc(y_test.tolist(), p_test.tolist())),
                "log_loss": float(log_loss(y_test.tolist(), p_test.tolist())),
                "ece": float(expected_calibration_error(y_test.tolist(), p_test.tolist(), n_bins=10)),
            }

            # Race gap at top10% (if Race present)
            race_gap_fpr = float("nan")
            race_gap_fnr = float("nan")
            if "Race" in ds.columns:
                race = ds.iloc[test_idx]["Race"].fillna("Unknown").astype(str).to_numpy()
                thr = _threshold_top_k(p_test, 0.10)
                fprs = []
                fnrs = []
                for label in sorted(set(race.tolist())):
                    idx = np.where(race == label)[0]
                    r = _binary_rates(y_test[idx], p_test[idx], thr)
                    if not np.isnan(r["fpr"]):
                        fprs.append(float(r["fpr"]))
                    if not np.isnan(r["fnr"]):
                        fnrs.append(float(r["fnr"]))
                if fprs:
                    race_gap_fpr = float(max(fprs) - min(fprs))
                if fnrs:
                    race_gap_fnr = float(max(fnrs) - min(fnrs))

            rows.append({**metrics, "race_gap_fpr_top10": race_gap_fpr, "race_gap_fnr_top10": race_gap_fnr})

        def _mean_std(key: str) -> tuple[float, float]:
            vals = np.array([r[key] for r in rows], dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                return float("nan"), float("nan")
            return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

        lines.extend(
            [
                f"## {dataset_key.upper()} {horizon.upper()} (calibration: {calibration})",
                "",
                "| Metric | Mean | Std |",
                "|---|---:|---:|",
            ]
        )
        for key, label in [
            ("brier", "Brier"),
            ("auroc", "AUROC"),
            ("auprc", "AUPRC"),
            ("log_loss", "Log loss"),
            ("ece", "ECE"),
            ("race_gap_fpr_top10", "Race FPR gap @top10%"),
            ("race_gap_fnr_top10", "Race FNR gap @top10%"),
        ]:
            m, s = _mean_std(key)
            lines.append(f"| {label} | {_format_float(m)} | {_format_float(s)} |")
        lines.append("")

    out.write_text("\n".join(lines))
    return out


def _parse_seeds(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-models", type=Path, default=None, help="Path to xgb_best_models.json")
    parser.add_argument("--out", type=Path, default=None, help="Output path for markdown report")
    parser.add_argument("--seeds", type=str, default="42,43,44", help="Comma-separated seeds")
    args = parser.parse_args()

    report = write_stability_report(
        best_models_path=args.best_models,
        out_path=args.out,
        seeds=_parse_seeds(str(args.seeds)),
    )
    print(f"wrote stability report: {report}")


if __name__ == "__main__":
    main()

