import pandas as pd

from ai_platform.data_engineering.core.pipeline import DataPipeline


def test_data_pipeline_executes_all_stages() -> None:
    """Test the reusable pipeline stage sequence."""
    loaded_data: list[pd.DataFrame] = []

    def extract() -> pd.DataFrame:
        return pd.DataFrame({"value": [1, 2, 3]})

    def clean(data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        result["cleaned"] = True
        return result

    def transform(data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        result["doubled"] = result["value"] * 2
        return result

    def validate(
        data: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        valid = data[data["value"] >= 2].copy()
        rejected = data[data["value"] < 2].copy()
        return valid, rejected

    def load(data: pd.DataFrame) -> None:
        loaded_data.append(data.copy())

    pipeline = DataPipeline(
        extract=extract,
        clean=clean,
        transform=transform,
        validate=validate,
        load=load,
    )

    result = pipeline.run()

    assert len(result.data) == 2
    assert len(result.rejected) == 1

    assert result.data["doubled"].tolist() == [4, 6]
    assert result.data["cleaned"].all()

    assert len(loaded_data) == 1
    assert loaded_data[0]["value"].tolist() == [2, 3]
