from collections.abc import Sequence

import pandas as pd


def concatenate_sources(
    sources: Sequence[pd.DataFrame],
    source_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Combine compatible data sources into one dataset."""
    if not sources:
        return pd.DataFrame()

    frames = [data.copy() for data in sources]

    if source_names is not None:
        if len(source_names) != len(frames):
            raise ValueError(
                "source_names must match the number of sources."
            )

        for frame, source_name in zip(frames, source_names):
            frame["source"] = source_name

    return pd.concat(
        frames,
        ignore_index=True,
    )


def merge_sources(
    left: pd.DataFrame,
    right: pd.DataFrame,
    key: str,
    how: str = "left",
) -> pd.DataFrame:
    """Merge two data sources using a shared key."""
    if key not in left.columns:
        raise KeyError(f"Missing key in left dataset: {key}")

    if key not in right.columns:
        raise KeyError(f"Missing key in right dataset: {key}")

    return left.merge(
        right,
        on=key,
        how=how,
    )
