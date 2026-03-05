"""Build a reincarceration benchmark from NCRP selected variables (ICPSR 37973).

Important limitation:
This public-use extract provides event timing at *year* granularity (e.g., ADMITYR/RELEASEYR).
We therefore define Y1/Y2/Y3 horizons using year-differences, not exact day windows.

We also treat "reincarceration" as "next prison term admission after a given term's release"
within the same person ID (ABT_INMATE_ID), derived from term records.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_SLUG = "ncrp_icpsr_37973"

SEX_LABELS = {
    1: "Male",
    2: "Female",
}

RACE_LABELS = {
    1: "White, non-Hispanic",
    2: "Black, non-Hispanic",
    3: "Hispanic, any race",
    4: "Other race(s), non-Hispanic",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _processed_dir() -> Path:
    path = _repo_root() / "data" / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _term_records_path() -> Path:
    return (
        _repo_root()
        / "data"
        / "raw"
        / DATASET_SLUG
        / "original"
        / "ICPSR_37973"
        / "DS0001"
        / "37973-0001-Data.tsv"
    )


@dataclass(frozen=True)
class SampleSpec:
    id_mod: int
    id_rem: int


def _sample_mask(ids: pd.Series, spec: SampleSpec) -> pd.Series:
    # Deterministic sampling without needing to hold a full set of IDs.
    # NCRP ABT_INMATE_ID values are often string-like (e.g., "A0120...").
    # Prefer using the numeric substring; fall back to -1 if parsing fails.
    as_str = ids.astype(str)
    digits = as_str.str.replace(r"\D", "", regex=True)
    values = pd.to_numeric(digits, errors="coerce").fillna(-1).astype("int64")
    mod = int(spec.id_mod)
    rem = int(spec.id_rem)
    if mod <= 0:
        raise ValueError("--id-mod must be >= 1")
    if rem < 0 or rem >= mod:
        raise ValueError("--id-rem must satisfy 0 <= id_rem < id_mod")
    return (values % mod) == rem


def load_term_records_sample(*, spec: SampleSpec, chunksize: int = 400_000) -> pd.DataFrame:
    path = _term_records_path()
    if not path.exists():
        raise FileNotFoundError(f"Missing raw NCRP term file: {path}")

    usecols = [
        "ABT_INMATE_ID",
        "STATE",
        "SEX",
        "RACE",
        "EDUCATION",
        "ADMTYPE",
        "OFFGENERAL",
        "OFFDETAIL",
        "ADMITYR",
        "RELEASEYR",
        "SENTLGTH",
        "TIMESRVD",
        "RELTYPE",
        "AGEADMIT",
        "AGERELEASE",
        "MAND_PRISREL_YEAR",
        "PROJ_PRISREL_YEAR",
        "PARELIG_YEAR",
    ]

    # Read in chunks and keep only sampled IDs.
    kept: list[pd.DataFrame] = []
    reader = pd.read_csv(path, sep="\t", usecols=usecols, chunksize=int(chunksize), low_memory=False)
    for chunk in reader:
        mask = _sample_mask(chunk["ABT_INMATE_ID"], spec)
        sub = chunk.loc[mask].copy()
        if len(sub) > 0:
            kept.append(sub)

    if not kept:
        return pd.DataFrame(columns=usecols)

    out = pd.concat(kept, ignore_index=True)
    return out


def _next_admit_year_by_term(frame: pd.DataFrame) -> pd.Series:
    # Compute next admission year per term record after sorting within person.
    out = frame.copy()
    # ABT_INMATE_ID is commonly an alphanumeric string; use numeric substring for grouping.
    out["ABT_INMATE_ID"] = (
        out["ABT_INMATE_ID"]
        .astype(str)
        .str.replace(r"\D", "", regex=True)
        .pipe(lambda s: pd.to_numeric(s, errors="coerce"))
        .astype("Int64")
    )
    out["ADMITYR"] = pd.to_numeric(out["ADMITYR"], errors="coerce").astype("Int64")
    out["RELEASEYR"] = pd.to_numeric(out["RELEASEYR"], errors="coerce").astype("Int64")

    # Treat sentinel/invalid years as missing (ICPSR codebooks commonly use 9999).
    for col in ("ADMITYR", "RELEASEYR"):
        series = out[col]
        series = series.mask(series == 9999)
        series = series.mask(series < 1900)
        series = series.mask(series > 2025)
        out[col] = series.astype("Int64")

    out = out.dropna(subset=["ABT_INMATE_ID", "ADMITYR", "RELEASEYR"]).copy()
    out = out.sort_values(["ABT_INMATE_ID", "ADMITYR", "RELEASEYR"], kind="mergesort")

    next_adm = out.groupby("ABT_INMATE_ID", sort=False)["ADMITYR"].shift(-1)
    # Enforce "after release" at year granularity.
    next_adm = next_adm.where(next_adm >= out["RELEASEYR"])
    next_adm.name = "NEXT_ADMITYR"
    # Reindex to original row order (some rows dropped).
    return next_adm.reindex(out.index)


def _derive_horizons_years(release_year: pd.Series, next_admit_year: pd.Series) -> pd.DataFrame:
    rel = pd.to_numeric(release_year, errors="coerce")
    nxt = pd.to_numeric(next_admit_year, errors="coerce")
    delta = nxt - rel
    out = pd.DataFrame(index=release_year.index)
    out["y1"] = ((delta >= 0) & (delta <= 1)).fillna(False).astype(int)
    out["y2"] = (delta == 2).fillna(False).astype(int)
    out["y3"] = (delta == 3).fillna(False).astype(int)
    return out


def build_release_cohort(*, spec: SampleSpec, chunksize: int) -> pd.DataFrame:
    df = load_term_records_sample(spec=spec, chunksize=chunksize)
    if df.empty:
        return df

    # Compute next admission year within each person.
    next_adm = _next_admit_year_by_term(df)
    df = df.loc[next_adm.index].copy()
    df["NEXT_ADMITYR"] = next_adm.values

    # Replace common ICPSR sentinel values with missing.
    # (The codebook uses 9999 to denote missing for several *_YEAR fields.)
    for col in ("MAND_PRISREL_YEAR", "PROJ_PRISREL_YEAR", "PARELIG_YEAR"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").mask(lambda s: s == 9999).astype("Int64")

    # Derive y1/y2/y3 using year granularity.
    horizons = _derive_horizons_years(df["RELEASEYR"], df["NEXT_ADMITYR"])
    df = pd.concat([df, horizons], axis=1)

    # Standardize a few column names for downstream use.
    df = df.rename(
        columns={
            "STATE": "state",
            "SEX": "sex",
            "RACE": "race",
            "AGERELEASE": "age_at_release",
            "AGEADMIT": "age_at_admit",
        }
    )

    # Map key demographic codes to human-readable labels (per ICPSR 37973 codebook).
    if "sex" in df.columns:
        sex_code = pd.to_numeric(df["sex"], errors="coerce")
        df["sex"] = sex_code.map(SEX_LABELS).fillna("Unknown").astype(str)
    if "race" in df.columns:
        race_code = pd.to_numeric(df["race"], errors="coerce")
        df["race"] = race_code.map(RACE_LABELS).fillna("Unknown").astype(str)
    return df


def write_outputs(frame: pd.DataFrame, *, spec: SampleSpec, out_dir: Path | None = None) -> None:
    out_dir = _processed_dir() if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"terms_mod{spec.id_mod}_r{spec.id_rem}"

    release_path = out_dir / f"{DATASET_SLUG}_{suffix}_release.parquet"
    frame.to_parquet(release_path, index=False)

    # Modeling tables per horizon with conditioning, following NIJ convention.
    # NOTE: exclude label-source columns to avoid leakage.
    leakage_cols = {"NEXT_ADMITYR"}
    base_cols = [c for c in frame.columns if c not in ("y1", "y2", "y3") and c not in leakage_cols]

    y1 = frame[base_cols].copy()
    y1["y"] = frame["y1"].astype(int)
    y1.to_parquet(out_dir / f"{DATASET_SLUG}_{suffix}_y1.parquet", index=False)

    y2 = frame.loc[frame["y1"].astype(int) == 0, base_cols].copy()
    y2["y"] = frame.loc[y2.index, "y2"].astype(int).values
    y2.to_parquet(out_dir / f"{DATASET_SLUG}_{suffix}_y2.parquet", index=False)

    y3 = frame.loc[(frame["y1"].astype(int) == 0) & (frame["y2"].astype(int) == 0), base_cols].copy()
    y3["y"] = frame.loc[y3.index, "y3"].astype(int).values
    y3.to_parquet(out_dir / f"{DATASET_SLUG}_{suffix}_y3.parquet", index=False)

    manifest = {
        "dataset": DATASET_SLUG,
        "variant": suffix,
        "label_family": "reincarceration",
        "timing_granularity": "year",
        "horizons": {
            "y1": "NEXT_ADMITYR in {RELEASEYR, RELEASEYR+1}",
            "y2": "NEXT_ADMITYR == RELEASEYR+2 (conditional on y1==0)",
            "y3": "NEXT_ADMITYR == RELEASEYR+3 (conditional on y1==0 and y2==0)",
        },
        "rows": int(len(frame)),
        "unique_ids": int(frame["ABT_INMATE_ID"].astype(str).nunique()),
    }
    (out_dir / f"{DATASET_SLUG}_{suffix}_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[ncrp] wrote: {release_path}")
    print(f"[ncrp] wrote: {out_dir / f'{DATASET_SLUG}_{suffix}_y1.parquet'} (and y2/y3)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id-mod", type=int, default=50, help="Keep rows where ABT_INMATE_ID % id_mod == id_rem")
    parser.add_argument("--id-rem", type=int, default=0, help="Remainder for deterministic ID sampling")
    parser.add_argument("--chunksize", type=int, default=400_000)
    args = parser.parse_args()

    spec = SampleSpec(id_mod=int(args.id_mod), id_rem=int(args.id_rem))
    df = build_release_cohort(spec=spec, chunksize=int(args.chunksize))
    if df.empty:
        print("[ncrp] no rows selected (try a smaller --id-mod)")
        return
    write_outputs(df, spec=spec)


if __name__ == "__main__":
    main()
