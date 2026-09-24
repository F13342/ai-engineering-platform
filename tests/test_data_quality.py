import pandas as pd
import pytest

from ai_platform.data_engineering.core.quality import validate_dataset


def test_data_quality_validation() -> None:
    """Test required values, uniqueness, and numeric ranges."""
    data = pd.DataFrame(
        {
            "record_id": [1, 2, 2, None, 5],
            "value": [10, 50, 120, 30, "invalid"],
            "name": ["A", "B", "C", "D", "E"],
        }
    )

    valid, rejected = validate_dataset(
        data,
        required_columns=["record_id"],
        unique_key="record_id",
        numeric_ranges={"value": (0, 100)},
    )

    assert len(valid) == 2
    assert valid["record_id"].tolist() == [1.0, 2.0]

    assert len(rejected) == 3

    reasons = rejected["rejection_reason"].tolist()

    assert any("duplicate record_id" in reason for reason in reasons)
    assert "missing record_id" in reasons
    assert "invalid value" in reasons


def test_data_quality_requires_existing_columns() -> None:
    """Test that missing schema columns fail clearly."""
    data = pd.DataFrame({"id": [1, 2]})

    with pytest.raises(KeyError):
        validate_dataset(
            data,
            required_columns=["record_id"],
        )
