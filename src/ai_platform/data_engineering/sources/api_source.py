from typing import Any

import pandas as pd
import requests


def load_api_data(url: str, timeout: int = 10) -> pd.DataFrame:
    """Load student data from a REST API."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    data: Any = response.json()

    if not isinstance(data, list):
        raise ValueError("API response must be a list of records.")

    return pd.DataFrame(data)