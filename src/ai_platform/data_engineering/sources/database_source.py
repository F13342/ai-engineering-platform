import sqlite3
from pathlib import Path

import pandas as pd


def load_database_data(database_path: Path) -> pd.DataFrame:
    """Load student enrollment data from SQLite."""
    query = """
        SELECT
            e.student_id,
            c.course_name,
            e.score
        FROM enrollments AS e
        JOIN courses AS c
            ON e.course_id = c.course_id
        ORDER BY e.student_id
    """

    with sqlite3.connect(database_path) as connection:
        return pd.read_sql_query(query, connection)