from pathlib import Path
import csv

from ai_platform.database.analytics.user_activity import (
    build_user_analytical_dataset,
)


def write_analytical_csv(
    data: list[dict[str, object]],
    file_path: Path,
) -> None:
    """Write analytical records to a CSV file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if not data:
        return

    fieldnames = list(data[0].keys())

    with file_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def export_user_analytical_dataset(
    file_path: Path,
) -> None:
    """Build and export the AI Platform user analytical dataset."""
    data = build_user_analytical_dataset()
    write_analytical_csv(data, file_path)
