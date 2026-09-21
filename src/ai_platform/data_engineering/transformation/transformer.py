import pandas as pd


def transform_student_data(data: pd.DataFrame) -> pd.DataFrame:
    """Apply types and create derived columns."""
    data = data.copy()

    data["student_id"] = data["student_id"].astype("string")
    data["age"] = pd.to_numeric(data["age"], errors="coerce")
    data["gpa"] = pd.to_numeric(data["gpa"], errors="coerce")
    data["attendance"] = pd.to_numeric(
        data["attendance"], errors="coerce"
    )
    data["score"] = pd.to_numeric(data["score"], errors="coerce")

    data["performance_level"] = pd.cut(
        data["gpa"],
        bins=[-float("inf"), 2, 3, float("inf")],
        labels=["Low", "Medium", "High"],
    )

    data["attendance_status"] = data["attendance"].apply(
        lambda value: "Good" if value >= 75 else "Low"
    )

    return data