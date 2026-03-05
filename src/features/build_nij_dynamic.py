from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .build_nij_static import HORIZON_TARGETS, _family_columns, _output_dir, _apply_horizon_filter
from src.data.load_nij import load_training

if TYPE_CHECKING:
    import pandas as pd


def _dynamic_family_columns() -> list[str]:
    return _family_columns("dynamic_supervision")


def get_dynamic_feature_columns(horizon: str) -> list[str]:
    if horizon not in HORIZON_TARGETS:
        raise ValueError("horizon must be one of {'y1', 'y2', 'y3'}")

    static_columns = _family_columns("id", "sensitive", "static_release")
    if horizon == "y1":
        return static_columns
    return static_columns + _dynamic_family_columns()


def _append_missing_indicators(frame, base_columns: list[str]) -> list[str]:
    indicators = [f"{column}__missing" for column in base_columns if f"{column}__missing" in frame.columns]
    return base_columns + indicators


def build_dynamic_datasets(training_frame: "pd.DataFrame | None" = None) -> dict[str, "pd.DataFrame"]:
    frame = training_frame if training_frame is not None else load_training()

    outputs: dict[str, "pd.DataFrame"] = {}
    for horizon, target_column in HORIZON_TARGETS.items():
        subset = _apply_horizon_filter(frame, horizon)
        base_columns = get_dynamic_feature_columns(horizon)
        feature_columns = _append_missing_indicators(subset, base_columns)
        dataset = subset[feature_columns + [target_column]].copy()
        dataset = dataset.rename(columns={target_column: "y"})
        outputs[horizon] = dataset
    return outputs


def write_dynamic_parquet(output_dir: Path | None = None) -> dict[str, Path]:
    datasets = build_dynamic_datasets()
    out_dir = output_dir or _output_dir()

    paths: dict[str, Path] = {}
    for horizon, frame in datasets.items():
        path = out_dir / f"nij_dynamic_{horizon}.parquet"
        frame.to_parquet(path, index=False)
        paths[horizon] = path
    return paths


def main() -> None:
    paths = write_dynamic_parquet()
    for horizon, path in sorted(paths.items()):
        print(f"wrote dynamic {horizon}: {path}")


if __name__ == "__main__":
    main()
