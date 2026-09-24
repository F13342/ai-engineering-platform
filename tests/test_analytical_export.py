from pathlib import Path

from ai_platform.database.analytics.export import write_analytical_csv


def test_analytical_csv_export(tmp_path: Path) -> None:
    """Test exporting analytical data to CSV."""
    output_file = tmp_path / "analytics.csv"

    data = [
        {
            "user_id": 1,
            "username": "AI User",
            "activity_total": 5,
        }
    ]

    write_analytical_csv(data, output_file)

    assert output_file.exists()

    content = output_file.read_text(encoding="utf-8")

    assert "user_id,username,activity_total" in content
    assert "1,AI User,5" in content
