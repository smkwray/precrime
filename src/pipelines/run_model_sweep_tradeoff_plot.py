"""Create an aggregate plot spec showing prediction vs NIJ FairAcc tradeoffs.

Reads `reports/model_sweep.json` (aggregate per-model metrics) and writes a compact
scatter-plot spec to `reports/plots/model_sweep_tradeoff.json`.

Rendering into a PNG is handled by `scripts/render_figures.py`.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


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


def write_tradeoff_plot_spec(
    in_json: Path | None = None,
    out_json: Path | None = None,
) -> Path:
    in_json = in_json or (_reports_dir() / "model_sweep.json")
    if not in_json.exists():
        raise FileNotFoundError(f"Missing {in_json}. Run `python -m src.pipelines.run_model_sweep` first.")

    out_json = out_json or (_plots_dir() / "model_sweep_tradeoff.json")

    records = json.loads(in_json.read_text())
    df = pd.DataFrame(records)
    required = {"horizon", "variant", "model", "calibration", "bs_sex_avg", "fairacc_sex_avg"}
    missing = sorted(required - set(df.columns.astype(str).tolist()))
    if missing:
        raise ValueError(f"model_sweep.json missing columns: {missing}")

    df = df.copy()
    df["horizon"] = df["horizon"].astype(str).str.lower()
    df["variant"] = df["variant"].astype(str)
    df["model"] = df["model"].astype(str)
    df["calibration"] = df["calibration"].astype(str)
    df["bs_sex_avg"] = pd.to_numeric(df["bs_sex_avg"], errors="coerce")
    df["fairacc_sex_avg"] = pd.to_numeric(df["fairacc_sex_avg"], errors="coerce")

    horizons = [h for h in ["y1", "y2", "y3"] if h in set(df["horizon"].tolist())]
    variants = sorted(df["variant"].dropna().unique().tolist())

    panels: list[dict[str, object]] = []
    for horizon in horizons:
        for variant in variants:
            block = df[(df["horizon"] == horizon) & (df["variant"] == variant)].copy()
            block = block.dropna(subset=["bs_sex_avg", "fairacc_sex_avg"])
            if block.empty:
                continue

            idx_best_pred = int(block["bs_sex_avg"].idxmin())
            idx_best_fair = int(block["fairacc_sex_avg"].idxmax())

            points: list[dict[str, object]] = []
            for i, row in block.iterrows():
                points.append(
                    {
                        "model": str(row["model"]),
                        "calibration": str(row["calibration"]),
                        "x": float(row["bs_sex_avg"]),
                        "y": float(row["fairacc_sex_avg"]),
                        "best_prediction": bool(i == idx_best_pred),
                        "best_fairacc": bool(i == idx_best_fair),
                    }
                )

            panels.append(
                {
                    "horizon": horizon,
                    "variant": variant,
                    "points": points,
                }
            )

    spec = {
        "kind": "model_sweep_tradeoff",
        "title": "Prediction vs NIJ-style FairAcc (sex-averaged)",
        "subtitle": "Primary objective: minimize sex-avg Brier; FairAcc shown as an alternate NIJ-style view.",
        "x_label": "Sex-average Brier error (lower is better)",
        "y_label": "Sex-average FairAcc (higher is better)",
        "panels": panels,
        "generated": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        "source": str(in_json.name),
        "notes": [
            "FairAcc = (1 - BS) * FP where FP = 1 - |FPR_black@0.5 - FPR_white@0.5| (computed within sex, then averaged).",
            "This is a diagnostic view; it does not recommend an operational threshold or policy.",
        ],
    }

    out_json.write_text(json.dumps(spec, indent=2))
    return out_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-json", type=Path, default=None, help="Input model_sweep.json path")
    parser.add_argument("--out-json", type=Path, default=None, help="Output plot spec path")
    args = parser.parse_args()

    out = write_tradeoff_plot_spec(in_json=args.in_json, out_json=args.out_json)
    print(f"wrote tradeoff plot spec: {out}")


if __name__ == "__main__":
    main()

