import pandas as pd


def aggregate_database_data(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate course data for each student."""
    if data.empty:
        return pd.DataFrame(
            columns=[
                "student_id",
                "course_count",
                "average_score",
                "score",
            ]
        )

    data = data.copy()
    data["student_id"] = data["student_id"].astype("string")

    summary = (
        data.groupby("student_id", as_index=False)
        .agg(
            course_count=("course_name", "count"),
            average_score=("score", "mean"),
        )
    )

    # Keep the student's highest score for record-level validation.
    max_score = (
        data.groupby("student_id", as_index=False)["score"]
        .max()
        .rename(columns={"score": "score"})
    )

    return summary.merge(max_score, on="student_id", how="left")


def integrate_student_data(
    csv_data: pd.DataFrame,
    api_data: pd.DataFrame,
    database_data: pd.DataFrame,
) -> pd.DataFrame:
    """Merge CSV, API, and SQLite data by student_id."""
    csv_data = csv_data.copy()
    api_data = api_data.copy()

    csv_data["student_id"] = (
        pd.to_numeric(csv_data["student_id"], errors="coerce")
        .astype("Int64")
        .astype("string")
    )

    api_data["student_id"] = api_data["student_id"].astype("string")

    database_summary = aggregate_database_data(database_data)

    result = csv_data.merge(
        api_data,
        on="student_id",
        how="left",
    )

    return result.merge(
        database_summary,
        on="student_id",
        how="left",
    )