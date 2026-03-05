"""Ingest locally-downloaded release-cohort datasets into a standard parquet format.

This pipeline is intentionally mapping-driven because ICPSR packages vary widely
in file format and column naming conventions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.data.icpsr_local import find_candidate_tables, load_table, pick_table
from src.data.release_cohort import derive_horizon_targets, make_episode_id, to_datetime


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _processed_dir() -> Path:
    path = _repo_root() / "data" / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _raw_dir(dataset_slug: str) -> Path:
    return _repo_root() / "data" / "raw" / dataset_slug / "original"


def _config_path(dataset_slug: str) -> Path:
    return _repo_root() / "configs" / "datasets" / f"{dataset_slug}.yaml"


def _load_config(dataset_slug: str) -> dict[str, Any]:
    path = _config_path(dataset_slug)
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset config: {path}")
    return yaml.safe_load(path.read_text())


def _require(cfg: dict[str, Any], key_path: str) -> str:
    cur: Any = cfg
    for part in key_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Missing required config key: {key_path}")
        cur = cur[part]
    if cur in (None, "", "__FILL_ME__"):
        raise ValueError(f"Config key must be set: {key_path}")
    return str(cur)


def _optional(cfg: dict[str, Any], key_path: str) -> str | None:
    cur: Any = cfg
    for part in key_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    if cur in (None, "", "__FILL_ME__"):
        return None
    return str(cur)


def cmd_inspect(dataset_slug: str) -> None:
    raw_dir = _raw_dir(dataset_slug)
    tables = find_candidate_tables(raw_dir)
    if not tables:
        print(f"[inspect] no tabular files found under: {raw_dir}")
        return

    print(f"[inspect] candidate tables under: {raw_dir}")
    for i, cand in enumerate(tables[:15]):
        print(f"  {i+1:02d}. {cand.path.relative_to(raw_dir)} ({cand.size_bytes/1e6:.1f} MB)")

    cfg = None
    try:
        cfg = _load_config(dataset_slug)
    except FileNotFoundError:
        pass

    preferred = None
    if cfg:
        preferred = cfg.get("tables", {}).get("preferred_globs")
        if isinstance(preferred, str):
            preferred = [preferred]

    chosen = pick_table(tables, preferred_globs=preferred)
    if chosen is None:
        return

    print(f"[inspect] loading head from: {chosen.path.relative_to(raw_dir)}")
    try:
        head = load_table(chosen.path, nrows=25)
    except Exception as exc:  # noqa: BLE001
        print(f"[inspect] unable to load table head: {exc}")
        print("[inspect] tip: convert the file to CSV locally (R haven / Python pyreadstat) and re-run.")
        return

    print(f"[inspect] shape(head): {head.shape}")
    print("[inspect] columns:")
    for name in list(head.columns.astype(str)):
        print(f"  - {name}")


def _build_release_table(dataset_slug: str, cfg: dict[str, Any]) -> pd.DataFrame:
    raw_dir = _raw_dir(dataset_slug)
    tables = find_candidate_tables(raw_dir)
    if not tables:
        raise FileNotFoundError(f"No supported tabular files found under {raw_dir}")

    preferred = cfg.get("tables", {}).get("preferred_globs")
    if isinstance(preferred, str):
        preferred = [preferred]
    chosen = pick_table(tables, preferred_globs=preferred)
    if chosen is None:
        raise FileNotFoundError(f"Unable to pick a table under {raw_dir}")

    frame = load_table(chosen.path, nrows=None)
    frame.columns = frame.columns.astype(str)
    return frame


def _load_table_by_globs(dataset_slug: str, globs: list[str]) -> pd.DataFrame:
    raw_dir = _raw_dir(dataset_slug)
    tables = find_candidate_tables(raw_dir)
    chosen = pick_table(tables, preferred_globs=globs)
    if chosen is None:
        raise FileNotFoundError(f"Unable to find a matching table under {raw_dir} for {globs}")
    frame = load_table(chosen.path, nrows=None)
    frame.columns = frame.columns.astype(str)
    return frame


def _link_first_event_date(
    releases: pd.DataFrame,
    admissions: pd.DataFrame,
    *,
    release_person_col: str,
    admission_person_col: str,
    release_date_col: str,
    admission_date_col: str,
    allow_same_day: bool,
) -> pd.Series:
    rel = releases[[release_person_col, release_date_col]].copy()
    rel["_row_id"] = releases.index
    adm = admissions[[admission_person_col, admission_date_col]].copy()

    rel[release_person_col] = rel[release_person_col].astype(str)
    adm[admission_person_col] = adm[admission_person_col].astype(str)

    rel_date = pd.to_datetime(rel[release_date_col], errors="coerce")
    adm_date = pd.to_datetime(adm[admission_date_col], errors="coerce")
    rel = rel.assign(_rel_date=rel_date)
    adm = adm.assign(_adm_date=adm_date)

    rel = rel.dropna(subset=["_rel_date"]).sort_values([release_person_col, "_rel_date"])
    adm = adm.dropna(subset=["_adm_date"]).sort_values([admission_person_col, "_adm_date"])

    # Next admission on/after release (within person).
    merged = pd.merge_asof(
        rel,
        adm,
        left_on="_rel_date",
        right_on="_adm_date",
        left_by=release_person_col,
        right_by=admission_person_col,
        direction="forward",
        allow_exact_matches=bool(allow_same_day),
    )
    linked = merged.set_index("_row_id")["_adm_date"]
    return linked.reindex(releases.index)


def cmd_process(dataset_slug: str) -> None:
    cfg = _load_config(dataset_slug)
    label_family = _require(cfg, "label_family")

    frame = _build_release_table(dataset_slug, cfg)
    date_fmt = _optional(cfg, "dates.format")

    person_id_col = _optional(cfg, "columns.person_id")
    episode_id_col = _optional(cfg, "columns.episode_id")
    release_date_col = _require(cfg, "columns.release_date")

    rearrest_col = _optional(cfg, "columns.event_date_rearrest")
    reconviction_col = _optional(cfg, "columns.event_date_reconviction")
    reincarceration_col = _optional(cfg, "columns.event_date_reincarceration")

    if label_family == "rearrest":
        event_col = rearrest_col
    elif label_family == "reconviction":
        event_col = reconviction_col
    elif label_family == "reincarceration":
        event_col = reincarceration_col
    else:
        raise ValueError("label_family must be one of {'rearrest','reconviction','reincarceration'}")

    # Optional linkage mode (primarily for NCRP): compute event date by linking releases to admissions.
    if event_col is None and label_family == "reincarceration" and isinstance(cfg.get("linkage"), dict):
        linkage = cfg.get("linkage") or {}
        release_globs = linkage.get("release_globs") or []
        admission_globs = linkage.get("admission_globs") or []
        admission_date_col = linkage.get("admission_date")
        admission_person_col = linkage.get("admission_person_id") or person_id_col
        allow_same_day = bool(linkage.get("allow_same_day", True))

        if not release_globs or not admission_globs:
            raise ValueError("linkage mode requires linkage.release_globs and linkage.admission_globs")
        if admission_date_col in (None, "", "__FILL_ME__"):
            raise ValueError("linkage mode requires linkage.admission_date")
        if not person_id_col or person_id_col in (None, "", "__FILL_ME__"):
            raise ValueError("linkage mode requires columns.person_id")
        if not admission_person_col or admission_person_col in (None, "", "__FILL_ME__"):
            raise ValueError("linkage mode requires linkage.admission_person_id or columns.person_id")

        releases = _load_table_by_globs(dataset_slug, [str(g) for g in release_globs])
        admissions = _load_table_by_globs(dataset_slug, [str(g) for g in admission_globs])
        if release_date_col not in releases.columns:
            raise KeyError(f"Missing release date column in release table: {release_date_col}")
        if person_id_col not in releases.columns:
            raise KeyError(f"Missing person id column in release table: {person_id_col}")
        if admission_date_col not in admissions.columns:
            raise KeyError(f"Missing admission date column in admission table: {admission_date_col}")
        if admission_person_col not in admissions.columns:
            raise KeyError(f"Missing person id column in admission table: {admission_person_col}")

        linked_event = _link_first_event_date(
            releases=releases,
            admissions=admissions,
            release_person_col=person_id_col,
            admission_person_col=str(admission_person_col),
            release_date_col=release_date_col,
            admission_date_col=str(admission_date_col),
            allow_same_day=allow_same_day,
        )

        # Replace `frame` with the release table; fill in computed event date.
        frame = releases.copy()
        frame["_linked_event_date"] = linked_event
        event_col = "_linked_event_date"

    if event_col is None:
        raise ValueError(
            f"label_family={label_family} requires an event date mapping "
            "(columns.event_date_*) or linkage configuration for reincarceration."
        )

    missing = [c for c in [release_date_col, event_col] if c not in frame.columns]
    if missing:
        raise KeyError(f"Missing required columns in raw table: {missing}")

    out = frame.copy()
    out["release_date"] = to_datetime(out[release_date_col], fmt=date_fmt)
    out["event_date"] = to_datetime(out[event_col], fmt=date_fmt)

    if person_id_col and person_id_col in out.columns:
        out["person_id"] = out[person_id_col].astype(str)
    else:
        out["person_id"] = pd.Series([None] * len(out))

    if episode_id_col and episode_id_col in out.columns:
        out["episode_id"] = out[episode_id_col].astype(str)
    else:
        out["episode_id"] = make_episode_id(
            out["person_id"] if person_id_col else None,
            out["release_date"],
            prefix=dataset_slug,
        )

    # Standardize common sensitive columns (best-effort).
    for std_name, key in [
        ("race", "columns.race"),
        ("sex", "columns.sex"),
        ("ethnicity", "columns.ethnicity"),
        ("age_at_release", "columns.age_at_release"),
        ("dob", "columns.dob"),
        ("state", "columns.state"),
    ]:
        src = _optional(cfg, key)
        if src and src in out.columns:
            out[std_name] = out[src]

    # If DOB exists but age_at_release is missing, derive it.
    if "age_at_release" not in out.columns and "dob" in out.columns:
        dob = to_datetime(out["dob"], fmt=date_fmt)
        rel = out["release_date"]
        out["age_at_release"] = ((rel - dob).dt.days / 365.25).round(1)

    # Derive y1/y2/y3 from index date + event date.
    targets = derive_horizon_targets(out["release_date"], out["event_date"])
    out = pd.concat([out, targets], axis=1)

    # Drop rows with missing release date.
    out = out[~out["release_date"].isna()].copy()

    # Build feature list: everything except raw label columns and derived bookkeeping.
    exclude = set(
        [
            release_date_col,
            event_col,
            "release_date",
            "event_date",
            "person_id",
            "episode_id",
            "y1",
            "y2",
            "y3",
        ]
    )
    # Do not leak obvious post-outcome date fields if present.
    for col in list(out.columns):
        lowered = str(col).lower()
        if "rearrest" in lowered or "reconvict" in lowered or "readmit" in lowered or "return" in lowered:
            if "date" in lowered or "time" in lowered:
                exclude.add(col)

    feature_exclude_cfg = cfg.get("feature", {}).get("exclude", []) or []
    exclude.update([str(c) for c in feature_exclude_cfg])

    include_cfg = cfg.get("feature", {}).get("include")
    if include_cfg:
        features = [c for c in [str(x) for x in include_cfg] if c in out.columns and c not in exclude]
    else:
        features = [c for c in out.columns if c not in exclude]

    # Write a canonical release table for downstream processing.
    processed_dir = _processed_dir()
    release_path = processed_dir / f"{dataset_slug}_release.parquet"
    keep_cols = ["episode_id", "person_id", "release_date", "event_date", "y1", "y2", "y3"]
    for extra in ("race", "sex", "ethnicity", "age_at_release", "state"):
        if extra in out.columns:
            keep_cols.append(extra)
    keep_cols.extend(features)

    out[keep_cols].to_parquet(release_path, index=False)
    manifest_path = processed_dir / f"{dataset_slug}_feature_manifest.json"
    manifest_path.write_text(json.dumps({"dataset": dataset_slug, "label_family": label_family, "features": features}, indent=2))

    # Write horizon-specific modeling tables (matching the repo convention of `y` as target).
    for horizon in ("y1", "y2", "y3"):
        subset = out[keep_cols].copy()
        if horizon != "y1":
            if horizon == "y2":
                subset = subset[subset["y1"] == 0]
            elif horizon == "y3":
                subset = subset[(subset["y1"] == 0) & (subset["y2"] == 0)]

        subset = subset.drop(columns=["y1", "y2", "y3"]).assign(y=out.loc[subset.index, horizon].astype(int).values)
        horizon_path = processed_dir / f"{dataset_slug}_{horizon}.parquet"
        subset.to_parquet(horizon_path, index=False)

    print(f"[process] wrote: {release_path}")
    print(f"[process] wrote: {manifest_path}")
    print(f"[process] wrote: {processed_dir / f'{dataset_slug}_y1.parquet'} (and y2/y3)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="dataset slug under data/raw/<slug>/original/")
    parser.add_argument("--task", choices=["inspect", "process"], required=True)
    args = parser.parse_args()

    if args.task == "inspect":
        cmd_inspect(args.dataset)
    else:
        cmd_process(args.dataset)


if __name__ == "__main__":
    main()
