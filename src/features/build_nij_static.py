from __future__ import annotations

from pathlib import Path
from typing import Mapping, TYPE_CHECKING

from src.data.load_nij import _load_schema, load_training

if TYPE_CHECKING:
    import pandas as pd


HORIZON_TARGETS = {
    "y1": "Recidivism_Arrest_Year1",
    "y2": "Recidivism_Arrest_Year2",
    "y3": "Recidivism_Arrest_Year3",
}

HORIZON_FILTERS = {
    "y1": tuple(),
    "y2": ("Recidivism_Arrest_Year1",),
    "y3": ("Recidivism_Arrest_Year1", "Recidivism_Arrest_Year2"),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _output_dir() -> Path:
    path = _repo_root() / "data" / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _family_columns(*families: str) -> list[str]:
    wanted = set(families)
    fields = _load_schema()
    return [str(field["name"]) for field in fields if str(field["family"]) in wanted]


def get_static_feature_columns() -> list[str]:
    return _family_columns("id", "sensitive", "static_release")


def row_passes_horizon(row: Mapping[str, object], horizon: str) -> bool:
    if horizon not in HORIZON_FILTERS:
        raise ValueError("horizon must be one of {'y1', 'y2', 'y3'}")

    for prereq in HORIZON_FILTERS[horizon]:
        value = row[prereq]
        if value not in (0, "0", 0.0, False):
            return False
    return True


def _apply_horizon_filter(frame, horizon: str):
    if horizon not in HORIZON_FILTERS:
        raise ValueError("horizon must be one of {'y1', 'y2', 'y3'}")

    filtered = frame
    for prereq in HORIZON_FILTERS[horizon]:
        filtered = filtered[filtered[prereq] == 0]
    return filtered


def _append_missing_indicators(frame, base_columns: list[str]) -> list[str]:
    indicators = [f"{column}__missing" for column in base_columns if f"{column}__missing" in frame.columns]
    return base_columns + indicators


def build_static_datasets(training_frame: "pd.DataFrame | None" = None) -> dict[str, "pd.DataFrame"]:
    frame = training_frame if training_frame is not None else load_training()
    base_columns = get_static_feature_columns()
    feature_columns = _append_missing_indicators(frame, base_columns)

    outputs: dict[str, "pd.DataFrame"] = {}
    for horizon, target_column in HORIZON_TARGETS.items():
        subset = _apply_horizon_filter(frame, horizon)
        dataset = subset[feature_columns + [target_column]].copy()
        dataset = dataset.rename(columns={target_column: "y"})
        outputs[horizon] = dataset
    return outputs


def write_static_parquet(output_dir: Path | None = None) -> dict[str, Path]:
    datasets = build_static_datasets()
    out_dir = output_dir or _output_dir()

    paths: dict[str, Path] = {}
    for horizon, frame in datasets.items():
        path = out_dir / f"nij_static_{horizon}.parquet"
        frame.to_parquet(path, index=False)
        paths[horizon] = path
    return paths


def main() -> None:
    paths = write_static_parquet()
    for horizon, path in sorted(paths.items()):
        print(f"wrote static {horizon}: {path}")


if __name__ == "__main__":
    main()
