#!/usr/bin/env python3
"""Maintenance CLI for the whole-person body ID database and sample files."""

from __future__ import annotations

try:
    from hailo_apps.my_projects.auto_face_id import manage_database
except ImportError:
    import manage_database


DB_NAME = "persons_body.sqlite3"
SAMPLES_DIR = manage_database.PROJECT_DIR / "body_samples"


def main() -> None:
    manage_database.main(
        db_name=DB_NAME,
        samples_dir=SAMPLES_DIR,
        description="Inspect, delete, and repair the person body ID SQLite database.",
    )


if __name__ == "__main__":
    main()
