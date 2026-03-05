"""Generate a compact, public-friendly 'what drives predictions' report for NIJ.

This produces:
- Top SHAP features table for the Y1 static XGBoost model (from `reports/plots/xgb_shap_static_y1.json`)
- Simple association tables (observed rearrest rates) for Y1/Y2/Y3 on the static-at-release dataset

All outputs are aggregate tables (no row-level data).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


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


def _load_nij_static(horizon: str) -> pd.DataFrame:
    path = _processed_dir() / f"nij_static_{horizon}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed NIJ static parquet: {path}")
    return pd.read_parquet(path)


def _risk_bin(score: pd.Series) -> pd.Series:
    s = pd.to_numeric(score, errors="coerce")
    out = pd.Series(index=score.index, dtype="object")
    out[(s >= 1) & (s <= 3)] = "1–3"
    out[(s >= 4) & (s <= 6)] = "4–6"
    out[(s >= 7) & (s <= 10)] = "7–10"
    return out.fillna("Unknown").astype(str)


def _rate_table(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in frame.columns:
        return pd.DataFrame(columns=["group", "n", "rearrest_rate"])
    g = frame[col].copy()
    if pd.api.types.is_numeric_dtype(g):
        g = pd.to_numeric(g, errors="coerce")
    g = g.astype("object").fillna("Unknown").astype(str)
    y = pd.to_numeric(frame["y"], errors="coerce").fillna(0).astype(int)

    tmp = pd.DataFrame({"group": g, "y": y})
    out = tmp.groupby("group", dropna=False).agg(n=("y", "size"), rearrest_rate=("y", "mean")).reset_index()
    out = out.sort_values(["n", "rearrest_rate"], ascending=[False, False], kind="mergesort")
    return out


def _md_table(df: pd.DataFrame, *, rate_fmt: str = "{:.3f}") -> list[str]:
    if df.empty:
        return ["(missing)"]
    lines = ["| Group | N | Rearrest rate |", "|---|---:|---:|"]
    for _, row in df.iterrows():
        lines.append(f"| {row['group']} | {int(row['n'])} | {rate_fmt.format(float(row['rearrest_rate']))} |")
    return lines


def _binary_lift_lines(frame: pd.DataFrame, col: str, *, positive: str = "Yes", negative: str = "No") -> list[str]:
    if col not in frame.columns:
        return ["(missing)"]

    y = pd.to_numeric(frame["y"], errors="coerce").fillna(0).astype(int)
    g = frame[col].astype(str).fillna("Unknown")
    mask = g.isin([positive, negative])
    if int(mask.sum()) == 0:
        return ["(missing)"]

    y_sub = y[mask]
    g_sub = g[mask]
    rates = pd.DataFrame({"g": g_sub, "y": y_sub}).groupby("g")["y"].mean()
    n = pd.DataFrame({"g": g_sub, "y": y_sub}).groupby("g")["y"].size()

    p_pos = float(rates.get(positive, np.nan))
    p_neg = float(rates.get(negative, np.nan))
    rd = p_pos - p_neg
    rr = (p_pos / p_neg) if (np.isfinite(p_pos) and np.isfinite(p_neg) and p_neg > 0) else np.nan

    lines = [
        "| Variable | N(Yes) | P(Y=1 \\| Yes) | N(No) | P(Y=1 \\| No) | Δ (pp) | RR |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| "
        + f"{col} | {int(n.get(positive, 0))} | {p_pos:.3f} | {int(n.get(negative, 0))} | {p_neg:.3f} | {rd*100:.1f} | {rr:.3f} |",
    ]
    return lines


def _top_binary_lifts(
    frame: pd.DataFrame,
    *,
    min_n: int = 300,
    top_k: int = 10,
) -> pd.DataFrame:
    y = pd.to_numeric(frame["y"], errors="coerce").fillna(0).astype(int)
    rows: list[dict[str, float | int | str]] = []

    for col in frame.columns:
        if col in {"ID", "y"}:
            continue
        s = frame[col]
        # Normalize values.
        as_str = s.astype("object").where(s.notna(), None).astype(str).replace({"nan": "Unknown", "<NA>": "Unknown"})
        uniq = set(as_str.dropna().unique().tolist())

        # Handle Yes/No style.
        if uniq.issubset({"Yes", "No", "Unknown"}):
            mask = as_str.isin(["Yes", "No"])
            if int(mask.sum()) == 0:
                continue
            g = as_str[mask]
            y_sub = y[mask]
            counts = g.value_counts()
            if int(counts.get("Yes", 0)) < min_n or int(counts.get("No", 0)) < min_n:
                continue
            p_yes = float(y_sub[g == "Yes"].mean())
            p_no = float(y_sub[g == "No"].mean())
            rr = (p_yes / p_no) if p_no > 0 else np.nan
            rd = p_yes - p_no
            rows.append(
                {
                    "variable": col,
                    "n_yes": int(counts.get("Yes", 0)),
                    "p_yes": p_yes,
                    "n_no": int(counts.get("No", 0)),
                    "p_no": p_no,
                    "delta_pp": rd * 100.0,
                    "rr": rr,
                }
            )
            continue

        # Handle numeric 0/1.
        num = pd.to_numeric(s, errors="coerce")
        uniq_num = set(num.dropna().unique().tolist())
        if uniq_num.issubset({0, 1}) and len(uniq_num) > 1:
            mask = num.isin([0, 1])
            counts = num[mask].value_counts()
            if int(counts.get(1, 0)) < min_n or int(counts.get(0, 0)) < min_n:
                continue
            p_yes = float(y[mask & (num == 1)].mean())
            p_no = float(y[mask & (num == 0)].mean())
            rr = (p_yes / p_no) if p_no > 0 else np.nan
            rd = p_yes - p_no
            rows.append(
                {
                    "variable": col,
                    "n_yes": int(counts.get(1, 0)),
                    "p_yes": p_yes,
                    "n_no": int(counts.get(0, 0)),
                    "p_no": p_no,
                    "delta_pp": rd * 100.0,
                    "rr": rr,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["rr", "delta_pp"], ascending=[False, False], kind="mergesort").head(int(top_k))
    return out.reset_index(drop=True)


def _md_binary_lifts(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["(none)"]

    def _pretty(name: str) -> str:
        mapping = {
            "Gang_Affiliated": "Gang affiliated",
            "Condition_MH_SA": "MH/SA condition",
            "Condition_Cog_Ed": "Cognitive/education condition",
            "Condition_Other": "Other condition",
            "Prior_Revocations_Parole": "Prior parole revocations",
            "Prior_Revocations_Probation": "Prior probation revocations",
            "Prior_Arrest_Episodes_DVCharges": "Any prior domestic-violence-charge arrest episode",
            "Prior_Arrest_Episodes_GunCharges": "Any prior gun-charge arrest episode",
            "Prior_Conviction_Episodes_Viol": "Any prior violent conviction episode",
            "Prior_Conviction_Episodes_Drug": "Any prior drug conviction episode",
            "_v2": "Any prior PP-violation-charge conviction episode",
            "_v3": "Any prior domestic-violence-charge conviction episode",
            "_v4": "Any prior gun-charge conviction episode",
            "Supervision_Risk_Score_First__missing": "Missing: supervision risk score",
            "Supervision_Level_First__missing": "Missing: supervision level",
            "Prison_Offense__missing": "Missing: prison offense",
            "Gang_Affiliated__missing": "Missing: gang affiliation",
        }
        return mapping.get(name, name)

    lines = [
        "| Variable | N(Yes) | P(Y=1\\|Yes) | N(No) | P(Y=1\\|No) | Δ (pp) | RR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        lines.append(
            "| "
            + f"{_pretty(str(row['variable']))} | {int(row['n_yes'])} | {float(row['p_yes']):.3f} | {int(row['n_no'])} | {float(row['p_no']):.3f} | "
            + f"{float(row['delta_pp']):.1f} | {float(row['rr']):.3f} |"
        )
    return lines


def _load_shap_top(path: Path, top_k: int = 15) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])
    obj = json.loads(path.read_text())

    # Two supported formats:
    # (A) plot-spec dict: {"x":[features], "y":[values], ...}
    # (B) list of dicts: [{"feature":..., "mean_abs_shap":...}, ...]
    if isinstance(obj, dict) and "x" in obj and "y" in obj:
        feats = [str(x) for x in obj.get("x", [])]
        vals = [float(y) for y in obj.get("y", [])]
        df = pd.DataFrame({"feature": feats, "mean_abs_shap": vals})
    elif isinstance(obj, list):
        df = pd.DataFrame(obj)
    else:
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])

    if "feature" not in df.columns or "mean_abs_shap" not in df.columns:
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])

    df["mean_abs_shap"] = pd.to_numeric(df["mean_abs_shap"], errors="coerce")
    df = df.dropna(subset=["mean_abs_shap"]).sort_values("mean_abs_shap", ascending=False).head(int(top_k))
    return df.reset_index(drop=True)


def main() -> None:
    out_path = _reports_dir() / "nij_predictive_factors.md"

    shap_path = _plots_dir() / "xgb_shap_static_y1.json"
    shap_top = _load_shap_top(shap_path, top_k=15)

    lines: list[str] = [
        "# NIJ — Predictive Factors and Simple Associations",
        "",
        "This report is intentionally descriptive: it shows **which features the model uses** (SHAP) and **how rearrest rates vary** by a few key variables (unadjusted associations).",
        "",
        "Notes:",
        "- The Y2 and Y3 tasks in this repo are **conditional** (only people not rearrested in prior horizons are included).",
        "- SHAP shows association with the model’s predictions, not causality.",
        "",
        "## Top Predictive Factors (SHAP, Year 1 Static)",
        "",
        "From `reports/plots/xgb_shap_static_y1.json` (XGBoost, static-at-release features).",
        "",
    ]

    if shap_top.empty:
        lines.append("(missing — run the NIJ XGBoost pipeline to generate SHAP specs)")
    else:
        lines.extend(["| Rank | Feature | Mean \\|SHAP\\| |", "|---:|---|---:|"])
        for i, (_, row) in enumerate(shap_top.iterrows(), start=1):
            lines.append(f"| {i} | {str(row['feature'])} | {float(row['mean_abs_shap']):.4f} |")

    lines.extend(
        [
            "",
            "## Observed Rearrest Rates by Key Variables (Static-at-release)",
            "",
            "These are raw subgroup rates (not adjusted for other variables).",
            "",
        ]
    )

    horizons = ["y1", "y2", "y3"]
    key_vars = [
        ("Age at release", "Age_at_Release", None),
        ("Gang affiliated", "Gang_Affiliated", None),
        ("MH/SA condition", "Condition_MH_SA", None),
        ("Supervision risk score (binned)", "Supervision_Risk_Score_First", _risk_bin),
        ("Prior property-arrest episodes", "Prior_Arrest_Episodes_Property", None),
        ("Prison years", "Prison_Years", None),
    ]

    for h in horizons:
        df = _load_nij_static(h)
        y = pd.to_numeric(df["y"], errors="coerce").fillna(0).astype(int)
        base = float(y.mean()) if len(y) else float("nan")

        lines.extend([f"### {h.upper()} (N={len(df):,}, base rate={base:.3f})", ""])

        # Quick “how much more likely” summary for key binary variables.
        lines.append("#### Lift (binary variables; unadjusted)")
        lines.append("")
        lines.extend(_binary_lift_lines(df, "Condition_MH_SA"))
        lines.append("")
        lines.extend(_binary_lift_lines(df, "Gang_Affiliated"))
        lines.append("")

        lines.append("#### Top Unadjusted Risk Lifts (Binary Indicators)")
        lines.append("")
        top = _top_binary_lifts(df, min_n=300, top_k=10)
        lines.extend(_md_binary_lifts(top))
        lines.append("")

        for title, col, transform in key_vars:
            view = df.copy()
            if transform is not None and col in view.columns:
                view[col] = transform(view[col])
            tbl = _rate_table(view, col)
            lines.append(f"#### {title}")
            lines.append("")
            lines.extend(_md_table(tbl, rate_fmt="{:.3f}"))
            lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"wrote NIJ factors report: {out_path}")


if __name__ == "__main__":
    main()
