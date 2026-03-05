"""Generate a compact 'what drives predictions' report for COMPAS (Broward County).

Produces:
- Top SHAP features table (from `reports/plots/xgb_shap_compas.json`)
- Simple association tables (observed rearrest rates) for key variables

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


def _load_compas() -> pd.DataFrame:
    path = _processed_dir() / "compas_2yr.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed COMPAS parquet: {path}")
    return pd.read_parquet(path)


def _rate_table(frame: pd.DataFrame, col: str, *, label: str = "Rearrest rate") -> pd.DataFrame:
    if col not in frame.columns:
        return pd.DataFrame(columns=["group", "n", "rate"])
    g = frame[col].copy()
    if pd.api.types.is_numeric_dtype(g):
        g = pd.to_numeric(g, errors="coerce")
    g = g.astype("object").fillna("Unknown").astype(str)
    y = pd.to_numeric(frame["y"], errors="coerce").fillna(0).astype(int)

    tmp = pd.DataFrame({"group": g, "y": y})
    out = tmp.groupby("group", dropna=False).agg(n=("y", "size"), rate=("y", "mean")).reset_index()
    out = out.sort_values(["n", "rate"], ascending=[False, False], kind="mergesort")
    return out


def _md_table(df: pd.DataFrame, *, rate_fmt: str = "{:.3f}", label: str = "Rearrest rate") -> list[str]:
    if df.empty:
        return ["(missing)"]
    lines = [f"| Group | N | {label} |", "|---|---:|---:|"]
    for _, row in df.iterrows():
        lines.append(f"| {row['group']} | {int(row['n'])} | {rate_fmt.format(float(row['rate']))} |")
    return lines


def _decile_bin(score: pd.Series) -> pd.Series:
    s = pd.to_numeric(score, errors="coerce")
    out = pd.Series(index=score.index, dtype="object")
    out[(s >= 1) & (s <= 3)] = "1–3 (low)"
    out[(s >= 4) & (s <= 6)] = "4–6 (medium)"
    out[(s >= 7) & (s <= 10)] = "7–10 (high)"
    return out.fillna("Unknown").astype(str)


def _priors_bin(count: pd.Series) -> pd.Series:
    s = pd.to_numeric(count, errors="coerce")
    out = pd.Series(index=count.index, dtype="object")
    out[s == 0] = "0"
    out[(s >= 1) & (s <= 2)] = "1–2"
    out[(s >= 3) & (s <= 5)] = "3–5"
    out[(s >= 6) & (s <= 10)] = "6–10"
    out[s > 10] = "11+"
    return out.fillna("Unknown").astype(str)


def _load_shap_top(path: Path, top_k: int = 10) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])
    obj = json.loads(path.read_text())
    if isinstance(obj, list):
        df = pd.DataFrame(obj)
    elif isinstance(obj, dict) and "x" in obj and "y" in obj:
        df = pd.DataFrame({"feature": obj["x"], "mean_abs_shap": obj["y"]})
    else:
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])

    if "feature" not in df.columns or "mean_abs_shap" not in df.columns:
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])
    df["mean_abs_shap"] = pd.to_numeric(df["mean_abs_shap"], errors="coerce")
    df = df.dropna(subset=["mean_abs_shap"]).sort_values("mean_abs_shap", ascending=False).head(int(top_k))
    return df.reset_index(drop=True)


def _top_binary_lifts(
    frame: pd.DataFrame,
    *,
    min_n: int = 100,
    top_k: int = 10,
) -> pd.DataFrame:
    y = pd.to_numeric(frame["y"], errors="coerce").fillna(0).astype(int)
    rows: list[dict] = []

    for col in frame.columns:
        if col in {"y", "ID"}:
            continue
        num = pd.to_numeric(frame[col], errors="coerce")
        uniq = set(num.dropna().unique().tolist())
        if not uniq.issubset({0, 1}) or len(uniq) < 2:
            continue
        mask = num.isin([0, 1])
        counts = num[mask].value_counts()
        if int(counts.get(1, 0)) < min_n or int(counts.get(0, 0)) < min_n:
            continue
        p_yes = float(y[mask & (num == 1)].mean())
        p_no = float(y[mask & (num == 0)].mean())
        rr = (p_yes / p_no) if p_no > 0 else np.nan
        rd = p_yes - p_no
        rows.append({
            "variable": col,
            "n_yes": int(counts.get(1, 0)),
            "p_yes": p_yes,
            "n_no": int(counts.get(0, 0)),
            "p_no": p_no,
            "delta_pp": rd * 100.0,
            "rr": rr,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["rr", "delta_pp"], ascending=[False, False]).head(int(top_k))
    return out.reset_index(drop=True)


def _pretty_feature(name: str) -> str:
    mapping = {
        "is_violent_recid": "Prior violent recidivism",
        "sex_Female": "Sex: female",
        "c_charge_degree_F": "Charge degree: felony",
        "race_African-American": "Race: African-American",
        "race_Caucasian": "Race: Caucasian",
    }
    return mapping.get(name, name)


def _md_binary_lifts(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["(none)"]
    lines = [
        "| Variable | N(Yes) | P(Y=1\\|Yes) | N(No) | P(Y=1\\|No) | Δ (pp) | RR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {_pretty_feature(str(row['variable']))} | {int(row['n_yes'])} | {float(row['p_yes']):.3f} | "
            f"{int(row['n_no'])} | {float(row['p_no']):.3f} | {float(row['delta_pp']):.1f} | {float(row['rr']):.3f} |"
        )
    return lines


def main() -> None:
    out_path = _reports_dir() / "compas_predictive_factors.md"
    df = _load_compas()
    y = pd.to_numeric(df["y"], errors="coerce").fillna(0).astype(int)
    base = float(y.mean())

    shap_top = _load_shap_top(_plots_dir() / "xgb_shap_compas.json", top_k=10)

    lines: list[str] = [
        "# COMPAS — Predictive Factors and Simple Associations",
        "",
        "This report shows **which features the model uses** (SHAP) and **how rearrest rates vary** by key variables (unadjusted associations) for the COMPAS (ProPublica Broward County) dataset.",
        "",
        "- Single horizon: two-year rearrest.",
        "- SHAP shows association with predictions, not causality.",
        "- Not directly comparable to NIJ (different jurisdiction, population, features, label).",
        "",
        "## Top Predictive Factors (SHAP)",
        "",
        "From `reports/plots/xgb_shap_compas.json` (XGBoost).",
        "",
    ]

    if shap_top.empty:
        lines.append("(missing — run the COMPAS XGBoost pipeline to generate SHAP specs)")
    else:
        lines.extend(["| Rank | Feature | Mean \\|SHAP\\| |", "|---:|---|---:|"])
        for i, (_, row) in enumerate(shap_top.iterrows(), start=1):
            lines.append(f"| {i} | {_pretty_feature(str(row['feature']))} | {float(row['mean_abs_shap']):.4f} |")

    lines.extend([
        "",
        f"## Observed Rearrest Rates by Key Variables (N={len(df):,}, base rate={base:.3f})",
        "",
        "These are raw subgroup rates (not adjusted for other variables).",
        "",
    ])

    # Binary lifts
    lines.append("### Top Unadjusted Risk Lifts (Binary Indicators)")
    lines.append("")
    top_lifts = _top_binary_lifts(df, min_n=100, top_k=10)
    lines.extend(_md_binary_lifts(top_lifts))
    lines.append("")

    # Key categorical breakdowns
    key_vars = [
        ("Race", "race", None),
        ("Sex", "sex", None),
        ("Age category", "age_cat", None),
        ("COMPAS decile score (binned)", "decile_score", _decile_bin),
        ("COMPAS violence decile score (binned)", "v_decile_score", _decile_bin),
        ("Prior offense count (binned)", "priors_count", _priors_bin),
        ("Charge degree", "c_charge_degree", None),
        ("COMPAS risk level", "score_text", None),
    ]

    for title, col, transform in key_vars:
        view = df.copy()
        if transform is not None and col in view.columns:
            view[col] = transform(view[col])
        tbl = _rate_table(view, col)
        lines.append(f"### {title}")
        lines.append("")
        lines.extend(_md_table(tbl))
        lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"wrote COMPAS factors report: {out_path}")


if __name__ == "__main__":
    main()
