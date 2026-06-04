#!/usr/bin/env python3
"""Migrate the auto face ID LanceDB records into SQLite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parents[2]
DATABASE_DIR = PROJECT_DIR / "database"
SAMPLES_DIR = PROJECT_DIR / "samples"

if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from hailo_apps.my_projects.auto_face_id.sqlite_db_handler import SQLiteDatabaseHandler
from hailo_apps.python.core.common.db_handler import DatabaseHandler, Record


def main() -> None:
    lance_db = DatabaseHandler(
        db_name="persons.db",
        table_name="persons",
        schema=Record,
        threshold=0.55,
        database_dir=str(DATABASE_DIR),
        samples_dir=str(SAMPLES_DIR),
    )
    sqlite_db = SQLiteDatabaseHandler(
        db_name="persons.sqlite3",
        threshold=0.55,
        database_dir=DATABASE_DIR,
        samples_dir=SAMPLES_DIR,
    )

    migrated_people = 0
    migrated_samples = 0
    for record in lance_db.get_all_records():
        if sqlite_db.get_record_by_id(record["global_id"]) is not None:
            print(f"Skipping existing person: {record['label']} ({record['global_id']})")
            continue

        samples = record["samples_json"]
        if not samples:
            print(f"Skipping person without samples: {record['label']}")
            continue

        first_sample = samples[0]
        sqlite_db.create_record(
            embedding=np.asarray(first_sample["embedding"], dtype=np.float32),
            sample=first_sample["sample_path"],
            timestamp=record["last_sample_recieved_time"],
            label=record["label"],
            global_id=record["global_id"],
            threshold=record["classificaiton_confidence_threshold"],
        )
        migrated_samples += 1

        for sample in samples[1:]:
            sqlite_db.insert_new_sample(
                record=sqlite_db.get_record_by_id(record["global_id"]),
                embedding=np.asarray(sample["embedding"], dtype=np.float32),
                sample=sample["sample_path"],
                timestamp=record["last_sample_recieved_time"],
                sample_id=sample["id"],
            )
            migrated_samples += 1

        migrated_people += 1
        print(f"Migrated: {record['label']} ({len(samples)} samples)")

    print(
        f"Migration complete: people={migrated_people}, samples={migrated_samples}, "
        f"database={sqlite_db.db_path}"
    )


if __name__ == "__main__":
    main()
