import pandas as pd


def clean_student_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize student data."""
    data = data.copy()

    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    text_columns = ["student_name", "major", "city", "status"]

    for column in text_columns:
        if column in data.columns:
            data[column] = (
                data[column]
                .astype("string")
                .str.strip()
            )

    if "major" in data.columns:
        data["major"] = data["major"].str.title()

    if "city" in data.columns:
        data["city"] = data["city"].replace(
            {"Sana'a": "Sanaa"}
        )

    return data