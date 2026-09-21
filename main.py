import logging
from pathlib import Path

from ai_platform.data_engineering.integration.integration import (
    integrate_student_data,
)
from ai_platform.data_engineering.output.csv_writer import write_csv
from ai_platform.data_engineering.sources.api_source import load_api_data
from ai_platform.data_engineering.sources.csv_source import load_csv
from ai_platform.data_engineering.sources.database_source import (
    load_database_data,
)
from ai_platform.data_engineering.transformation.cleaner import (
    clean_student_data,
)
from ai_platform.data_engineering.transformation.transformer import (
    transform_student_data,
)
from ai_platform.data_engineering.validation.quality import validate_records


CSV_PATH = Path("data/raw/students.csv")
DATABASE_PATH = Path("database/students.db")

FINAL_PATH = Path("data/processed/final_dataset.csv")
REJECTED_PATH = Path("data/rejected/rejected_records.csv")

API_URL = "http://localhost:8000/students"


def setup_logging() -> None:
    """Configure pipeline logging."""
    Path("logs").mkdir(exist_ok=True)

    logging.basicConfig(
        filename="logs/pipeline.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def run_pipeline() -> None:
    """Run the complete data engineering pipeline."""
    logging.info("Pipeline started.")

    logging.info("Extracting CSV data.")
    csv_data = load_csv(CSV_PATH)

    logging.info("Extracting API data.")
    api_data = load_api_data(API_URL)

    logging.info("Extracting SQLite data.")
    database_data = load_database_data(DATABASE_PATH)

    logging.info("Cleaning source data.")
    csv_data = clean_student_data(csv_data)
    api_data = clean_student_data(api_data)

    logging.info("Integrating data sources.")
    integrated_data = integrate_student_data(
        csv_data,
        api_data,
        database_data,
    )

    logging.info("Transforming integrated data.")
    transformed_data = transform_student_data(integrated_data)

    logging.info("Validating final records.")
    valid_data, rejected_data = validate_records(transformed_data)

    write_csv(valid_data, FINAL_PATH)
    write_csv(rejected_data, REJECTED_PATH)

    logging.info(
        "Pipeline completed. Valid=%s Rejected=%s",
        len(valid_data),
        len(rejected_data),
    )


if __name__ == "__main__":
    setup_logging()
    run_pipeline()