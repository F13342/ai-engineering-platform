from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd


ExtractStage = Callable[[], pd.DataFrame]
DataStage = Callable[[pd.DataFrame], pd.DataFrame]
ValidateStage = Callable[
    [pd.DataFrame],
    tuple[pd.DataFrame, pd.DataFrame],
]
LoadStage = Callable[[pd.DataFrame], None]


@dataclass(frozen=True)
class PipelineResult:
    """Represent the result of a data pipeline run."""

    data: pd.DataFrame
    rejected: pd.DataFrame


class DataPipeline:
    """Execute a reusable data processing pipeline."""

    def __init__(
        self,
        extract: ExtractStage,
        clean: DataStage,
        transform: DataStage,
        validate: ValidateStage,
        load: LoadStage,
    ) -> None:
        self._extract = extract
        self._clean = clean
        self._transform = transform
        self._validate = validate
        self._load = load

    def run(self) -> PipelineResult:
        """Run extraction, cleaning, transformation, validation, and load."""
        data = self._extract()
        data = self._clean(data)
        data = self._transform(data)

        valid_data, rejected_data = self._validate(data)

        self._load(valid_data)

        return PipelineResult(
            data=valid_data,
            rejected=rejected_data,
        )
