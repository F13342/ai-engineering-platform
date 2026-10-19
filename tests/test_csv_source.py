from pathlib import Path

from ai_platform.data_engineering.sources.csv_source import load_csv


def test_load_csv(tmp_path: Path) -> None:
    """Test loading student data from a temporary CSV file."""
    file_path = tmp_path / "students.csv"
    file_path.write_text(
        "student_id,student_name,age,major,city\n"
        "1,Ahmed,21,Computer Science,Dhamar\n"
        "2,Sara,22,Information Systems,Sanaa\n",
        encoding="utf-8",
    )

    data = load_csv(file_path)

    assert not data.empty
    assert list(data.columns) == [
        "student_id",
        "student_name",
        "age",
        "major",
        "city",
    ]
    assert len(data) == 2
    assert data.iloc[0]["student_name"] == "Ahmed"
