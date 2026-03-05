from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.data.load_compas import load_compas_2yr_analysis

if TYPE_CHECKING:
    import pandas as pd


FEATURE_COLUMNS = [
    "id",
    "sex",
    "race",
    "age",
    "age_cat",
    "c_charge_degree",
    "c_charge_desc",
    "juv_fel_count",
    "juv_misd_count",
    "juv_other_count",
    "priors_count",
    "days_b_screening_arrest",
    "decile_score",
    "score_text",
    "v_decile_score",
    "v_score_text",
    "is_violent_recid",
    "two_year_recid",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_output_path() -> Path:
    path = _repo_root() / "data" / "processed" / "compas_2yr.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_compas_2yr(frame: "pd.DataFrame | None" = None) -> "pd.DataFrame":
    """
    Build the COMPAS 2-year benchmark table used by modeling.

    Notes:
    - NIJ and COMPAS are not directly comparable: NIJ outcomes are year-1/2/3 arrest flags on
      Georgia parole release cohorts, while COMPAS here is a Broward County pretrial sample with
      one two-year recidivism target.
    - ProPublica documents staged narrowing from 18,610 scored people (raw COMPAS export)
      to a pretrial subset and then to the 6,172-row two-year analysis sample. This builder
      reproduces the final analysis filter starting from `compas-scores-two-years.csv`.
    """
    source = frame if frame is not None else load_compas_2yr_analysis()
    output = source[FEATURE_COLUMNS].copy()
    output = output.rename(columns={"two_year_recid": "y"})
    return output


def write_compas_parquet(output_path: Path | None = None) -> Path:
    dataset = build_compas_2yr()
    path = output_path or _default_output_path()
    dataset.to_parquet(path, index=False)
    return path


def main() -> None:
    output_path = write_compas_parquet()
    print(f"wrote compas_2yr: {output_path}")


if __name__ == "__main__":
    main()
