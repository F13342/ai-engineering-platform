import pandas as pd
import pytest

from ai_platform.data_engineering.core.integration import (
    concatenate_sources,
    merge_sources,
)


def test_concatenate_multiple_sources() -> None:
    """Test combining multiple compatible datasets."""
    source_a = pd.DataFrame(
        {
            "record_id": [1, 2],
            "value": ["A", "B"],
        }
    )

    source_b = pd.DataFrame(
        {
            "record_id": [3],
            "value": ["C"],
        }
    )

    result = concatenate_sources(
        [source_a, source_b],
        ["source_a", "source_b"],
    )

    assert len(result) == 3
    assert result["record_id"].tolist() == [1, 2, 3]
    assert result["source"].tolist() == [
        "source_a",
        "source_a",
        "source_b",
    ]


def test_merge_sources_by_key() -> None:
    """Test merging complementary datasets by a shared key."""
    left = pd.DataFrame(
        {
            "record_id": [1, 2, 3],
            "name": ["A", "B", "C"],
        }
    )

    right = pd.DataFrame(
        {
            "record_id": [1, 2],
            "score": [90, 85],
        }
    )

    result = merge_sources(
        left,
        right,
        key="record_id",
    )

    assert len(result) == 3
    assert result.loc[result["record_id"] == 1, "score"].iloc[0] == 90
    assert pd.isna(
        result.loc[result["record_id"] == 3, "score"].iloc[0]
    )


def test_merge_sources_requires_shared_key() -> None:
    """Test that integration fails clearly when the key is missing."""
    left = pd.DataFrame({"record_id": [1]})
    right = pd.DataFrame({"other_id": [1]})

    with pytest.raises(KeyError):
        merge_sources(left, right, key="record_id")
