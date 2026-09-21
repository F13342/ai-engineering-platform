from pathlib import Path

import pandas as pd


def write_csv(data: pd.DataFrame, file_path: Path) -> None:
    """Write a DataFrame to CSV."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(file_path, index=False)