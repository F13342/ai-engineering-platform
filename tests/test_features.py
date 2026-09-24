import pandas as pd
import pytest

from ai_platform.data_engineering.core.features import (
    add_ratio_feature,
    build_ai_platform_features,
)


def test_add_ratio_feature() -> None:
    """Test generic safe ratio feature creation."""
    data = pd.DataFrame(
        {
            "messages": [10, 5],
            "conversations": [2, 0],
        }
    )

    result = add_ratio_feature(
        data,
        numerator="messages",
        denominator="conversations",
        output_column="messages_per_conversation",
    )

    assert result["messages_per_conversation"].tolist() == [5.0, 0.0]


def test_build_ai_platform_features() -> None:
    """Test AI Platform analytical feature construction."""
    data = pd.DataFrame(
        {
            "message_count": [20],
            "conversation_count": [4],
            "document_count": [6],
            "project_count": [2],
        }
    )

    result = build_ai_platform_features(data)

    assert result["messages_per_conversation"].iloc[0] == 5.0
    assert result["documents_per_project"].iloc[0] == 3.0


def test_ratio_feature_requires_columns() -> None:
    """Test clear failure when required columns are missing."""
    data = pd.DataFrame({"messages": [10]})

    with pytest.raises(KeyError):
        add_ratio_feature(
            data,
            numerator="messages",
            denominator="conversations",
            output_column="ratio",
        )
