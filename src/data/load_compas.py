from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


RAW_FILENAME = "compas-scores-two-years.csv"
EXPECTED_RAW_ROWS = 7214
EXPECTED_ANALYSIS_ROWS = 6172


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_raw_dir() -> Path:
    return _repo_root() / "data" / "raw" / "compas"


def _require_pandas():
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise ImportError("pandas is required for COMPAS data loading functions") from exc
    return pd


def _coalesce_duplicate_columns(frame):
    if "decile_score.1" in frame.columns:
        frame["decile_score"] = frame["decile_score.1"]
        frame = frame.drop(columns=["decile_score.1"])
    if "priors_count.1" in frame.columns:
        frame["priors_count"] = frame["priors_count.1"]
        frame = frame.drop(columns=["priors_count.1"])
    return frame


def _cast_compas_types(frame):
    pd = _require_pandas()
    out = frame.copy()

    int_columns = [
        "id",
        "age",
        "juv_fel_count",
        "juv_misd_count",
        "juv_other_count",
        "priors_count",
        "days_b_screening_arrest",
        "decile_score",
        "v_decile_score",
        "is_recid",
        "is_violent_recid",
        "two_year_recid",
    ]
    for column in int_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")

    categorical_columns = [
        "sex",
        "race",
        "age_cat",
        "c_charge_degree",
        "c_charge_desc",
        "score_text",
        "v_score_text",
    ]
    for column in categorical_columns:
        out[column] = out[column].astype("category")

    return out


def propublica_two_year_mask(frame) -> "pd.Series":
    pd = _require_pandas()
    days = pd.to_numeric(frame["days_b_screening_arrest"], errors="coerce")

    mask = days.between(-30, 30, inclusive="both")
    mask &= frame["is_recid"].astype(str) != "-1"
    mask &= frame["c_charge_degree"].astype(str) != "O"
    mask &= frame["score_text"].astype(str) != "N/A"
    return mask


def load_compas_raw(raw_dir: Path | None = None, validate: bool = True) -> "pd.DataFrame":
    pd = _require_pandas()
    path = (raw_dir or _default_raw_dir()) / RAW_FILENAME

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame = _coalesce_duplicate_columns(frame)

    if validate and len(frame) != EXPECTED_RAW_ROWS:
        raise ValueError(f"Expected {EXPECTED_RAW_ROWS} raw rows, found {len(frame)}")

    return frame


def load_compas_2yr_analysis(raw_dir: Path | None = None, validate: bool = True) -> "pd.DataFrame":
    frame = load_compas_raw(raw_dir=raw_dir, validate=validate)
    filtered = frame.loc[propublica_two_year_mask(frame)].copy()

    if validate and len(filtered) != EXPECTED_ANALYSIS_ROWS:
        raise ValueError(f"Expected {EXPECTED_ANALYSIS_ROWS} analysis rows, found {len(filtered)}")

    filtered = _cast_compas_types(filtered)
    return filtered
