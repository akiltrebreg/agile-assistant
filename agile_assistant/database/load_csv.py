"""Load CSV data files from S3 into PostgreSQL.

Downloads ``report_agile_dashboard.csv`` and ``report_agile_dashboard_metrics.csv``
from S3 (``S3_DATA_BUCKET``/``S3_DATA_PATH``) and loads them via psycopg2's
``copy_expert`` (server-side ``COPY FROM STDIN``).

Usage::

    python -m agile_assistant.database.load_csv
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

import boto3
import psycopg2

from agile_assistant.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# Tables to load: (table_name, csv_filename, truncate_first)
_TABLES = [
    ("report_agile_dashboard", "report_agile_dashboard.csv"),
    ("report_agile_dashboard_metrics", "report_agile_dashboard_metrics.csv"),
]


def _download_csvs_from_s3() -> Path:
    """Download all CSVs from S3 to a temp directory."""
    bucket = settings.s3_data_bucket
    prefix = settings.s3_data_path.rstrip("/") + "/"
    endpoint = settings.s3_endpoint

    logger.info("Downloading CSVs from s3://%s/%s ...", bucket, prefix)

    session = boto3.session.Session(
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "ru-central1"),
    )
    s3 = session.client("s3", endpoint_url=endpoint)

    local_dir = Path(tempfile.mkdtemp(prefix="csv_data_"))
    for _, filename in _TABLES:
        key = f"{prefix}{filename}"
        local_path = local_dir / filename
        logger.info("  s3://%s/%s → %s", bucket, key, local_path)
        s3.download_file(bucket, key, str(local_path))

    return local_dir


def _load_table(conn: psycopg2.extensions.connection, table: str, csv_path: Path) -> int:
    """Truncate *table* and bulk-load *csv_path* via COPY FROM STDIN."""
    with conn.cursor() as cur, csv_path.open("r", encoding="utf-8") as f:
        cur.execute(f"TRUNCATE TABLE {table};")
        cur.copy_expert(
            f"COPY {table} FROM STDIN WITH (FORMAT csv, HEADER true, DELIMITER ',', NULL '')",
            f,
        )
        cur.execute(f"SELECT COUNT(*) FROM {table};")
        count = cur.fetchone()[0]
    conn.commit()
    return count


def main() -> None:
    """Download configured CSVs from S3 and load each into PostgreSQL.

    Exits with code 1 when ``S3_DATA_BUCKET`` is not configured. Each
    table is truncated and rebuilt via ``COPY FROM STDIN`` inside a
    single connection that is closed in a ``finally`` block.
    """
    if not settings.s3_data_bucket:
        logger.error("S3_DATA_BUCKET is not configured")
        sys.exit(1)

    csv_dir = _download_csvs_from_s3()

    logger.info("Connecting to PostgreSQL: %s", settings.database_url.split("@")[-1])
    conn = psycopg2.connect(settings.database_url)
    try:
        for table, filename in _TABLES:
            csv_path = csv_dir / filename
            count = _load_table(conn, table, csv_path)
            logger.info("[OK] %s: %d rows loaded", table, count)
    finally:
        conn.close()

    logger.info("Done.")


if __name__ == "__main__":
    main()
