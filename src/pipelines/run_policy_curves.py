"""Generate NIJ Y1 policy tradeoff curves (selection budget vs error metrics).

This pipeline writes an aggregate JSON plot spec under `reports/plots/`.
Rendering into a PNG under `docs/figures/` is handled by `scripts/render_figures.py`,
so this pipeline does not require optional visualization dependencies.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build_nij_static import build_static_datasets
from src.pipelines._split_utils import (
    extract_target_column,
    prepare_feature_matrix,
    split_train_cal_select_test,
)
from src.pipelines.run_operational_eval import (
    _binary_rates,
    _load_best_models,
    _threshold_top_k,
    _train_predict_xgb,
)


RANDOM_SEED = 42


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _reports_dir() -> Path:
    out = _repo_root() / "reports"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _plots_dir() -> Path:
    out = _reports_dir() / "plots"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _race_mask(values: pd.Series, label: str) -> np.ndarray:
    v = values.fillna("Unknown").astype(str).str.strip().str.upper()
    return (v == label.upper()).to_numpy()

def _relpath(path: Path) -> str:
    try:
        return str(path.relative_to(_repo_root()))
    except ValueError:
        return str(path)


def write_policy_curves(
    best_models_path: Path | None = None,
    out_json: Path | None = None,
    out_report: Path | None = None,
) -> tuple[Path, Path]:
    best = _load_best_models(best_models_path)
    if "y1" not in best:
        raise ValueError("Missing y1 config in xgb_best_models.json")

    y1_cfg = best["y1"]
    if str(y1_cfg["dataset"]) != "static":
        raise ValueError("Policy curves task expects Y1 static best model configuration.")

    calibration = str(y1_cfg["calibration"])
    params = dict(y1_cfg.get("tuning", {}).get("best_params", {}))

    ds = build_static_datasets()["y1"].copy()
    target_col = extract_target_column(ds)
    y_full = pd.to_numeric(ds[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
    feature_frame = ds.drop(columns=[target_col]).copy()

    # 4-way split: thresholds derived from select split
    fit_idx, cal_idx, select_idx, test_idx = split_train_cal_select_test(y_full, seed=RANDOM_SEED)
    x_df, _ = prepare_feature_matrix(feature_frame)
    x_np = x_df.to_numpy(dtype=float)

    p_select = _train_predict_xgb(
        x=x_np, y=y_full,
        fit_idx=fit_idx, cal_idx=cal_idx, eval_idx=select_idx,
        params=params, calibration=calibration, seed=RANDOM_SEED,
    )
    p_test = _train_predict_xgb(
        x=x_np, y=y_full,
        fit_idx=fit_idx, cal_idx=cal_idx, eval_idx=test_idx,
        params=params, calibration=calibration, seed=RANDOM_SEED,
    )

    y_test = y_full[test_idx].astype(int)
    test_frame = ds.iloc[test_idx].copy()
    race_vals = test_frame["Race"].fillna("Unknown").astype(str) if "Race" in test_frame else pd.Series(["Unknown"] * len(test_frame))
    black_mask = _race_mask(race_vals, "BLACK")
    white_mask = _race_mask(race_vals, "WHITE")

    ks = np.arange(0.01, 0.301, 0.01)
    xs: list[float] = []
    overall_fpr: list[float] = []
    overall_fnr: list[float] = []
    overall_ppv: list[float] = []
    black_fpr: list[float] = []
    black_fnr: list[float] = []
    black_ppv: list[float] = []
    white_fpr: list[float] = []
    white_fnr: list[float] = []
    white_ppv: list[float] = []

    for k in ks:
        # Threshold derived from select split, applied to test split
        thr = _threshold_top_k(p_select, float(k))
        r_all = _binary_rates(y_test, p_test, threshold=thr)
        r_black = _binary_rates(y_test[black_mask], p_test[black_mask], threshold=thr) if np.any(black_mask) else {"fpr": np.nan, "fnr": np.nan, "ppv": np.nan}
        r_white = _binary_rates(y_test[white_mask], p_test[white_mask], threshold=thr) if np.any(white_mask) else {"fpr": np.nan, "fnr": np.nan, "ppv": np.nan}

        xs.append(float(k))
        overall_fpr.append(float(r_all["fpr"]))
        overall_fnr.append(float(r_all["fnr"]))
        overall_ppv.append(float(r_all["ppv"]))
        black_fpr.append(float(r_black["fpr"]))
        black_fnr.append(float(r_black["fnr"]))
        black_ppv.append(float(r_black["ppv"]))
        white_fpr.append(float(r_white["fpr"]))
        white_fnr.append(float(r_white["fnr"]))
        white_ppv.append(float(r_white["ppv"]))

    plot_spec = {
        "kind": "policy_curves",
        "title": "NIJ Y1 static policy tradeoffs",
        "x_label": "Selection rate (top-k fraction)",
        "metrics": {
            "fpr": {
                "overall": overall_fpr,
                "black": black_fpr,
                "white": white_fpr,
            },
            "fnr": {
                "overall": overall_fnr,
                "black": black_fnr,
                "white": white_fnr,
            },
            "ppv": {
                "overall": overall_ppv,
                "black": black_ppv,
                "white": white_ppv,
            },
        },
        "x": xs,
        "metadata": {
            "horizon": "y1",
            "dataset": "static",
            "seed": RANDOM_SEED,
            "calibration": calibration,
            "group_counts": {
                "BLACK": int(np.sum(black_mask)),
                "WHITE": int(np.sum(white_mask)),
            },
            "caution": "Subgroup curves can be unstable for small-N groups; interpret trends with uncertainty.",
        },
    }

    json_path = out_json or (_plots_dir() / "nij_y1_policy_curves.json")
    json_path.write_text(json.dumps(plot_spec, indent=2))

    report_path = out_report or (_reports_dir() / "policy_curves.md")
    report_lines = [
        "# Policy Curves (NIJ Y1 Static)",
        "",
        f"Generated: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}",
        "",
        "Artifacts:",
        f"- JSON plot spec: `{_relpath(json_path)}`",
        "- PNG figure (rendered from JSON): `docs/figures/nij_y1_policy_tradeoffs.png`",
        "",
        "Configuration:",
        f"- Model: XGBoost best config for Y1 static (`calibration={calibration}`)",
        f"- Split seed: `{RANDOM_SEED}` (same fit/cal/select/test protocol as other NIJ reports)",
        f"- Group counts in test split: BLACK={int(np.sum(black_mask))}, WHITE={int(np.sum(white_mask))}",
        "",
        "Caution:",
        "- Curves summarize threshold-policy tradeoffs (budget vs error rates), not a recommended operational policy.",
        "- Subgroup curves can vary with sample size and split randomness; treat small differences as uncertain.",
    ]
    report_path.write_text("\n".join(report_lines))

    return json_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-models", type=Path, default=None, help="Path to xgb_best_models.json")
    parser.add_argument("--out-json", type=Path, default=None, help="Output path for JSON plot spec")
    parser.add_argument("--out-report", type=Path, default=None, help="Output path for markdown summary")
    args = parser.parse_args()

    json_path, report_path = write_policy_curves(
        best_models_path=args.best_models,
        out_json=args.out_json,
        out_report=args.out_report,
    )
    print(f"wrote policy-curve json: {json_path}")
    print(f"wrote policy-curve report: {report_path}")


if __name__ == "__main__":
    main()
