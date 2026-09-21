from pathlib import Path

from ai_platform.data_engineering.integration.integration import (
    aggregate_database_data,
)
from ai_platform.data_engineering.sources.database_source import (
    load_database_data,
)


def test_database_integration() -> None:
    """Test successful aggregation of SQLite student data."""
    database_path = Path("database/students.db")

    data = load_database_data(database_path)
    result = aggregate_database_data(data)

    assert not result.empty
    assert list(result.columns) == [
    "student_id",
    "course_count",
    "average_score",
    "score",
]
    assert result.loc[
        result["student_id"] == "1001", "course_count"
    ].iloc[0] == 2