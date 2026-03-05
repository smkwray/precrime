"""Helpers for working with locally-downloaded ICPSR/NACJD datasets.

This repo intentionally avoids auto-downloading many datasets because they are
account-gated and/or covered by specific data use agreements. These utilities
assume you've already downloaded and extracted an ICPSR package under
`data/raw/<dataset_slug>/original/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


_TABULAR_SUFFIXES = (
    ".csv",
    ".tsv",
    ".txt",
    ".dta",  # Stata
    ".sas7bdat",
    ".xpt",  # SAS transport
    ".sav",  # SPSS (requires pyreadstat)
    ".por",  # SPSS portable (requires pyreadstat)
)


@dataclass(frozen=True)
class CandidateTable:
    path: Path
    size_bytes: int


def find_candidate_tables(root: Path) -> list[CandidateTable]:
    if not root.exists():
        return []

    tables: list[CandidateTable] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _TABULAR_SUFFIXES:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        tables.append(CandidateTable(path=path, size_bytes=int(size)))

    tables.sort(key=lambda t: (t.size_bytes, str(t.path)), reverse=True)
    return tables


def _read_text(path: Path, nrows: int | None, sep: str | None) -> pd.DataFrame:
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t", nrows=nrows, low_memory=False)
    if sep is not None:
        return pd.read_csv(path, sep=sep, nrows=nrows, low_memory=False)
    # Sniff delimiter (comma vs tab) using python engine.
    return pd.read_csv(path, sep=None, engine="python", nrows=nrows, low_memory=False)


def load_table(path: Path, *, nrows: int | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix in (".csv", ".tsv", ".txt"):
        return _read_text(path, nrows=nrows, sep=None)

    if suffix == ".dta":
        # pandas supports chunksize for Stata; use it to avoid loading full files in inspect mode.
        if nrows is not None:
            it = pd.read_stata(path, chunksize=max(1, int(nrows)))
            return next(iter(it))
        return pd.read_stata(path)

    if suffix in (".sas7bdat", ".xpt"):
        if nrows is not None:
            it = pd.read_sas(path, chunksize=max(1, int(nrows)))
            return next(iter(it))
        return pd.read_sas(path)

    if suffix in (".sav", ".por"):
        # Requires pyreadstat; pandas will raise an ImportError if missing.
        if nrows is not None:
            return pd.read_spss(path, nrows=max(1, int(nrows)))
        return pd.read_spss(path)

    raise ValueError(f"Unsupported table type: {path}")


def pick_table(
    tables: Iterable[CandidateTable],
    *,
    preferred_globs: list[str] | None = None,
) -> CandidateTable | None:
    tables_list = list(tables)
    if not tables_list:
        return None

    if preferred_globs:
        expanded: list[CandidateTable] = []
        for pattern in preferred_globs:
            for cand in tables_list:
                if cand.path.match(pattern) or cand.path.as_posix().endswith(pattern.strip("*")):
                    expanded.append(cand)
        if expanded:
            expanded.sort(key=lambda t: (t.size_bytes, str(t.path)), reverse=True)
            return expanded[0]

    return tables_list[0]

