#!/usr/bin/env python3
"""Print a human-readable summary of the auto face ID SQLite database."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from hailo_apps.my_projects.auto_face_id.sqlite_db_handler import SQLiteDatabaseHandler
except ImportError:
    from sqlite_db_handler import SQLiteDatabaseHandler


PROJECT_DIR = Path(__file__).resolve().parent
DATABASE_DIR = PROJECT_DIR / "database"
SAMPLES_DIR = PROJECT_DIR / "samples"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect the auto face ID SQLite database.")
    parser.add_argument("--clear", action="store_true", help="Delete all people and face samples.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    db = SQLiteDatabaseHandler(
        db_name="persons.sqlite3",
        threshold=0.55,
        database_dir=DATABASE_DIR,
        samples_dir=SAMPLES_DIR,
    )

    if args.clear:
        db.clear_table()

    records = db.get_all_records()
    print(f"Database: {db.db_path}")
    print(f"Total people: {len(records)}")
    print(f"Total entered: {db.get_total_entered()}")
    for record in records:
        print(
            f"- {record['label']}: global_id={record['global_id']} "
            f"samples={len(record['samples_json'])} "
            f"visits={len(record.get('visits_json') or [])} "
            f"entered={record.get('entered', 0)}"
        )


if __name__ == "__main__":
    main()
