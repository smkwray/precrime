"""Generic helpers for release-cohort style datasets (index date + first event date)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HorizonSpec:
    name: str
    start_day_exclusive: int
    end_day_inclusive: int


DEFAULT_HORIZONS: list[HorizonSpec] = [
    HorizonSpec("y1", 0, 365),
    HorizonSpec("y2", 365, 730),
    HorizonSpec("y3", 730, 1095),
]


def to_datetime(series: pd.Series, fmt: str | None = None) -> pd.Series:
    if fmt:
        return pd.to_datetime(series, errors="coerce", format=fmt)
    return pd.to_datetime(series, errors="coerce")


def derive_horizon_targets(
    index_date: pd.Series,
    event_date: pd.Series,
    *,
    horizons: Iterable[HorizonSpec] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    idx = pd.to_datetime(index_date, errors="coerce")
    evt = pd.to_datetime(event_date, errors="coerce")
    delta_days = (evt - idx).dt.total_seconds() / (24 * 3600)

    out = pd.DataFrame(index=index_date.index)
    for h in horizons:
        out[h.name] = (
            (delta_days > float(h.start_day_exclusive)) & (delta_days <= float(h.end_day_inclusive))
        ).astype(int)

    # If index date is missing, set all targets to 0 (and let upstream filtering drop missing index dates).
    missing_index = idx.isna()
    if missing_index.any():
        for h in horizons:
            out.loc[missing_index, h.name] = 0
    return out


def horizon_condition_mask(frame: pd.DataFrame, horizon: str) -> pd.Series:
    if horizon == "y1":
        return pd.Series(True, index=frame.index)
    if horizon == "y2":
        return frame.get("y1", 0).astype(int) == 0
    if horizon == "y3":
        return (frame.get("y1", 0).astype(int) == 0) & (frame.get("y2", 0).astype(int) == 0)
    raise ValueError("horizon must be one of {'y1','y2','y3'}")


def make_episode_id(
    person_id: pd.Series | None,
    index_date: pd.Series,
    *,
    prefix: str,
) -> pd.Series:
    idx = pd.to_datetime(index_date, errors="coerce").dt.strftime("%Y-%m-%d").fillna("unknown_date")
    if person_id is None:
        return prefix + "_" + idx + "_" + pd.Series(np.arange(len(index_date)), index=index_date.index).astype(str)
    pid = person_id.fillna("unknown_id").astype(str)
    return prefix + "_" + pid + "_" + idx

