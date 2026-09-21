from pathlib import Path

import pandas as pd


def load_csv(file_path: Path) -> pd.DataFrame:
    """Load student data from a CSV file."""
    return pd.read_csv(file_path)