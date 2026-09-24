from pathlib import Path

import pandas as pd

from ai_platform.database.analytics.user_activity import (
    build_user_analytical_dataset,
)


def load_sql_analytical_data() -> pd.DataFrame:
    """Load AI Platform analytical data into a DataFrame."""
    records = build_user_analytical_dataset()

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def save_sql_analytical_data(
    output_path: Path,
) -> None:
    """Save the SQL analytical dataset as a Data Engineering input."""
    data = load_sql_analytical_data()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        output_path,
        index=False,
    )
