import pandas as pd


def validate_records(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate records and separate valid and rejected rows."""
    rejected = []
    valid_indices = []

    duplicate_ids = (
        data["student_id"].notna()
        & data["student_id"].duplicated(keep="first")
    )

    for index, row in data.iterrows():
        reasons = []

        if pd.isna(row["student_id"]):
            reasons.append("missing student_id")
        elif duplicate_ids.loc[index]:
            reasons.append("duplicate student_id")

        if not pd.isna(row["age"]) and not 16 <= row["age"] <= 80:
            reasons.append("invalid age")

        if not pd.isna(row["gpa"]) and not 0 <= row["gpa"] <= 4:
            reasons.append("invalid gpa")

        if not pd.isna(row["attendance"]) and not 0 <= row["attendance"] <= 100:
            reasons.append("invalid attendance")

        if not pd.isna(row["score"]) and not 0 <= row["score"] <= 100:
            reasons.append("invalid score")

        if reasons:
            record = row.to_dict()
            record["reason"] = "; ".join(reasons)
            rejected.append(record)
        else:
            valid_indices.append(index)

    valid_df = data.loc[valid_indices].copy()
    rejected_df = pd.DataFrame(rejected)

    return (
        valid_df.reset_index(drop=True),
        rejected_df.reset_index(drop=True),
    )