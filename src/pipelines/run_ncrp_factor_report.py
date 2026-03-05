"""Generate a compact 'what drives predictions' report for NCRP (ICPSR 37973).

Produces:
- Top SHAP features table (from `reports/plots/xgb_shap_ncrp37973_y1.json`)
- Simple association tables (observed reincarceration rates) for Y1/Y2/Y3

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


def _load_ncrp(horizon: str, variant: str = "terms_mod200_r0") -> pd.DataFrame:
    path = _processed_dir() / f"ncrp_icpsr_37973_{variant}_{horizon}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed NCRP parquet: {path}")
    return pd.read_parquet(path)


# ICPSR 37973 codebook mappings
ADMTYPE_MAP = {"1": "New court commitment", "2": "Parole revocation/return", "3": "Other/transfer", "4": "Unknown"}
RELTYPE_MAP = {"1": "Conditional release (parole)", "2": "Unconditional release (max-out)", "3": "Other/transfer", "4": "Unknown"}
OFFGENERAL_MAP = {"1": "Violent", "2": "Property", "3": "Drug", "4": "Public order", "5": "Other", "9": "Unknown"}


def _decode_col(series: pd.Series, mapping: dict) -> pd.Series:
    return series.astype(str).map(lambda x: mapping.get(str(x).split(".")[0], x))


# ICPSR 37973 coded-value mappings (these columns are already binned in the public-use extract)
TIMESRVD_MAP = {"0": "< 6 months", "1": "6–11 months", "2": "1–2 years", "3": "2–5 years", "4": "5+ years"}
SENTLGTH_MAP = {
    "0": "< 1 year", "1": "1–2 years", "2": "2–3 years", "3": "3–5 years",
    "4": "5–10 years", "5": "10–25 years", "6": "25+ years / life", "9": "Unknown",
}
AGE_MAP = {"1": "18–24", "2": "25–34", "3": "35–44", "4": "45–54", "5": "55+", "9": "Unknown"}


def _rate_table(frame: pd.DataFrame, col: str) -> pd.DataFrame:
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


def _md_table(df: pd.DataFrame, *, rate_fmt: str = "{:.3f}", label: str = "Reincarceration rate") -> list[str]:
    if df.empty:
        return ["(missing)"]
    lines = [f"| Group | N | {label} |", "|---|---:|---:|"]
    for _, row in df.iterrows():
        lines.append(f"| {row['group']} | {int(row['n'])} | {rate_fmt.format(float(row['rate']))} |")
    return lines


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


def _pretty_feature(name: str) -> str:
    mapping = {
        "RELEASEYR": "Release year",
        "ADMITYR": "Admission year",
        "TIMESRVD": "Time served (months)",
        "SENTLGTH": "Sentence length (months)",
        "PROJ_PRISREL_YEAR": "Projected prison release year",
        "MAND_PRISREL_YEAR": "Mandatory prison release year",
        "PARELIG_YEAR": "Parole eligibility year",
        "age_at_release": "Age at release",
        "age_at_admit": "Age at admission",
        "sex_Female": "Sex: female",
    }
    if name in mapping:
        return mapping[name]
    # Coded categories
    for prefix, codebook, label in [
        ("ADMTYPE_", ADMTYPE_MAP, "Admission type"),
        ("RELTYPE_", RELTYPE_MAP, "Release type"),
        ("OFFGENERAL_", OFFGENERAL_MAP, "Offense category"),
    ]:
        if name.startswith(prefix):
            code = name[len(prefix):]
            decoded = codebook.get(code, code)
            return f"{label}: {decoded}"
    if name.startswith("state_"):
        fips = name[len("state_"):]
        state_names = {
            "6": "California", "12": "Florida", "13": "Georgia",
            "29": "Missouri", "40": "Oklahoma", "42": "Pennsylvania",
            "45": "South Carolina", "8": "Colorado",
        }
        return f"State: {state_names.get(fips, f'FIPS {fips}')}"
    if name.startswith("race_"):
        return f"Race: {name[len('race_'):]}"
    if name.startswith("OFFDETAIL_"):
        return f"Offense detail code {name[len('OFFDETAIL_'):]}"
    return name


def main() -> None:
    out_path = _reports_dir() / "ncrp_37973_predictive_factors.md"

    shap_top = _load_shap_top(_plots_dir() / "xgb_shap_ncrp37973_y1.json", top_k=10)

    lines: list[str] = [
        "# NCRP (ICPSR 37973) — Predictive Factors and Simple Associations",
        "",
        "This report shows **which features the model uses** (SHAP) and **how reincarceration rates vary** by key variables (unadjusted associations) for the NCRP dataset.",
        "",
        "- Label: **return to prison / reincarceration** (not rearrest).",
        "- Y2/Y3 are conditional (only individuals not reincarcerated in prior horizons).",
        "- Event timing: **year granularity** (ICPSR 37973 public-use extract).",
        "- SHAP shows association with predictions, not causality.",
        "- Not directly comparable to NIJ (different label, jurisdiction, features).",
        "",
        "## Top Predictive Factors (SHAP, Year 1)",
        "",
        "From `reports/plots/xgb_shap_ncrp37973_y1.json` (XGBoost).",
        "",
    ]

    if shap_top.empty:
        lines.append("(missing — run the NCRP XGBoost pipeline to generate SHAP specs)")
    else:
        lines.extend(["| Rank | Feature | Mean \\|SHAP\\| |", "|---:|---|---:|"])
        for i, (_, row) in enumerate(shap_top.iterrows(), start=1):
            lines.append(f"| {i} | {_pretty_feature(str(row['feature']))} | {float(row['mean_abs_shap']):.4f} |")

    lines.extend([
        "",
        "## Observed Reincarceration Rates by Key Variables",
        "",
        "These are raw subgroup rates (not adjusted for other variables).",
        "",
    ])

    horizons = ["y1", "y2", "y3"]
    key_vars = [
        ("Race", "race", None),
        ("Sex", "sex", None),
        ("Age at release", "age_at_release", lambda s: _decode_col(s, AGE_MAP)),
        ("Admission type", "ADMTYPE", lambda s: _decode_col(s, ADMTYPE_MAP)),
        ("Release type", "RELTYPE", lambda s: _decode_col(s, RELTYPE_MAP)),
        ("Offense category", "OFFGENERAL", lambda s: _decode_col(s, OFFGENERAL_MAP)),
        ("Time served", "TIMESRVD", lambda s: _decode_col(s, TIMESRVD_MAP)),
        ("Sentence length", "SENTLGTH", lambda s: _decode_col(s, SENTLGTH_MAP)),
    ]

    for h in horizons:
        df = _load_ncrp(h)
        y = pd.to_numeric(df["y"], errors="coerce").fillna(0).astype(int)
        base = float(y.mean()) if len(y) else float("nan")

        lines.extend([f"### {h.upper()} (N={len(df):,}, base rate={base:.3f})", ""])

        for title, col, transform in key_vars:
            view = df.copy()
            if transform is not None and col in view.columns:
                view[col] = transform(view[col])
            tbl = _rate_table(view, col)
            lines.append(f"#### {title}")
            lines.append("")
            lines.extend(_md_table(tbl))
            lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"wrote NCRP factors report: {out_path}")


if __name__ == "__main__":
    main()
