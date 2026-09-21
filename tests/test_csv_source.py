from pathlib import Path

from ai_platform.data_engineering.sources.csv_source import load_csv


def test_load_csv() -> None:
    """Test loading student data from a CSV file."""
    file_path = Path("data/raw/students.csv")

    data = load_csv(file_path)

    assert not data.empty
    assert list(data.columns) == [
        "student_id",
        "student_name",
        "age",
        "major",
        "city",
    ]