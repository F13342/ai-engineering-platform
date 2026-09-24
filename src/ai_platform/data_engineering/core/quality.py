from collections.abc import Mapping, Sequence

import pandas as pd


def validate_dataset(
    data: pd.DataFrame,
    required_columns: Sequence[str] = (),
    unique_key: str | None = None,
    numeric_ranges: Mapping[str, tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate a generic dataset and separate valid and rejected records."""
    data = data.copy()

    for column in required_columns:
        if column not in data.columns:
            raise KeyError(f"Missing required column: {column}")

    if unique_key is not None and unique_key not in data.columns:
        raise KeyError(f"Missing unique key column: {unique_key}")

    if numeric_ranges is not None:
        for column in numeric_ranges:
            if column not in data.columns:
                raise KeyError(f"Missing numeric column: {column}")

    duplicate_mask = pd.Series(False, index=data.index)

    if unique_key is not None:
        duplicate_mask = (
            data[unique_key].notna()
            & data[unique_key].duplicated(keep="first")
        )

    rejected: list[dict[str, object]] = []
    valid_indices: list[int] = []

    for index, row in data.iterrows():
        reasons: list[str] = []

        for column in required_columns:
            if pd.isna(row[column]):
                reasons.append(f"missing {column}")

        if unique_key is not None and duplicate_mask.loc[index]:
            reasons.append(f"duplicate {unique_key}")

        if numeric_ranges is not None:
            for column, (minimum, maximum) in numeric_ranges.items():
                value = row[column]

                if pd.isna(value):
                    continue

                numeric_value = pd.to_numeric(
                    value,
                    errors="coerce",
                )

                if pd.isna(numeric_value):
                    reasons.append(f"invalid {column}")
                elif not minimum <= numeric_value <= maximum:
                    reasons.append(f"invalid {column}")

        if reasons:
            record = row.to_dict()
            record["rejection_reason"] = "; ".join(reasons)
            rejected.append(record)
        else:
            valid_indices.append(index)

    return (
        data.loc[valid_indices].reset_index(drop=True),
        pd.DataFrame(rejected).reset_index(drop=True),
    )
