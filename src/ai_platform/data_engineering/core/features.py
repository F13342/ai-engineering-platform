import pandas as pd


def add_ratio_feature(
    data: pd.DataFrame,
    numerator: str,
    denominator: str,
    output_column: str,
) -> pd.DataFrame:
    """Add a safe ratio-based feature to a dataset."""
    if numerator not in data.columns:
        raise KeyError(f"Missing numerator column: {numerator}")

    if denominator not in data.columns:
        raise KeyError(f"Missing denominator column: {denominator}")

    result = data.copy()

    denominator_values = pd.to_numeric(
        result[denominator],
        errors="coerce",
    ).fillna(0)

    numerator_values = pd.to_numeric(
        result[numerator],
        errors="coerce",
    ).fillna(0)

    result[output_column] = (
        numerator_values
        .div(denominator_values.where(denominator_values != 0))
        .fillna(0.0)
    )

    return result


def build_ai_platform_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Build reusable features from the AI Platform analytical dataset."""
    result = data.copy()

    result = add_ratio_feature(
        result,
        numerator="message_count",
        denominator="conversation_count",
        output_column="messages_per_conversation",
    )

    result = add_ratio_feature(
        result,
        numerator="document_count",
        denominator="project_count",
        output_column="documents_per_project",
    )

    return result
