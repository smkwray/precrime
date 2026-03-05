from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_schema_path() -> Path:
    return _repo_root() / "schemas" / "nij.yml"


def _default_raw_dir() -> Path:
    return _repo_root() / "data" / "raw" / "nij"


def _parse_scalar(raw: str) -> str:
    value = raw.strip()
    if value.startswith('"') and value.endswith('"') and len(value) >= 2:
        return value[1:-1]
    if value.startswith("'") and value.endswith("'") and len(value) >= 2:
        return value[1:-1]
    return value


def _fallback_parse_schema(schema_path: Path) -> list[dict[str, str | int]]:
    fields: list[dict[str, str | int]] = []
    current: dict[str, str | int] | None = None

    for line in schema_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- position:"):
            if current is not None:
                fields.append(current)
            current = {"position": int(stripped.split(":", 1)[1].strip())}
            continue
        if current is None or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        current[key.strip()] = _parse_scalar(value)

    if current is not None:
        fields.append(current)
    return fields


def _load_schema(schema_path: Path | None = None) -> list[dict[str, str | int]]:
    resolved = schema_path or _default_schema_path()
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(resolved.read_text())
        fields = data.get("fields", []) if isinstance(data, dict) else []
        return [dict(field) for field in fields]
    except Exception:
        return _fallback_parse_schema(resolved)


def _as_nullable_numeric(series, kind: str):
    import pandas as pd  # type: ignore

    converted = pd.to_numeric(series, errors="coerce")
    if kind == "integer":
        return converted.astype("Int64")
    return converted.astype("Float64")


def _apply_type_casts(df, fields: list[dict[str, str | int]]):
    import pandas as pd  # type: ignore

    for field in fields:
        col = str(field["name"])
        typ = str(field["type"])
        missing_encoding = str(field.get("missing_value_encoding", "none"))
        missing_is_blank = "blank string" in missing_encoding.lower()
        missing_mask = df[col].eq("")

        if missing_is_blank:
            df[f"{col}__missing"] = missing_mask.astype("int8")

        if typ in {"integer_identifier", "ordinal_integer"}:
            df[col] = _as_nullable_numeric(df[col], kind="integer")
            continue
        if typ in {"continuous_float", "continuous_proportion"}:
            df[col] = _as_nullable_numeric(df[col], kind="float")
            continue
        if typ == "target_binary":
            mapped = df[col].map({"Yes": 1, "No": 0})
            df[col] = mapped.astype("Int8")
            continue

        prepared = df[col].mask(missing_mask, pd.NA) if missing_is_blank else df[col]
        df[col] = prepared.astype("category")

    return df


def _validate_frame(df, fields: list[dict[str, str | int]], expected_rows: int | None = None) -> None:
    schema_columns = [str(field["name"]) for field in fields]
    if list(df.columns[: len(schema_columns)]) != schema_columns:
        raise ValueError("Input columns do not match schema field order")
    if expected_rows is not None and len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(df)}")


def _load_csv(path: Path):
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise ImportError("pandas is required for NIJ data loading functions") from exc

    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_training(
    raw_dir: Path | None = None,
    schema_path: Path | None = None,
    validate: bool = True,
) -> "pd.DataFrame":
    fields = _load_schema(schema_path)
    csv_path = (raw_dir or _default_raw_dir()) / "nij-challenge2021_training_dataset.csv"
    df = _load_csv(csv_path)

    if validate:
        _validate_frame(df, fields, expected_rows=18028)

    df = _apply_type_casts(df, fields)
    return df


def load_test(
    dataset: int = 1,
    raw_dir: Path | None = None,
    schema_path: Path | None = None,
    validate: bool = True,
) -> "pd.DataFrame":
    if dataset not in (1, 2, 3):
        raise ValueError("dataset must be one of {1, 2, 3}")

    fields = _load_schema(schema_path)
    if dataset == 1:
        selected_fields = fields[:33]
        expected_rows = 7807
    elif dataset == 2:
        selected_fields = fields[:49]
        expected_rows = 5460
    else:
        selected_fields = fields[:49]
        expected_rows = 4146

    csv_path = (raw_dir or _default_raw_dir()) / f"nij-challenge2021_test_dataset_{dataset}.csv"
    df = _load_csv(csv_path)

    if validate:
        _validate_frame(df, selected_fields, expected_rows=expected_rows)

    df = _apply_type_casts(df, selected_fields)
    return df
