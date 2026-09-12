from pathlib import Path
from unittest.mock import patch

from hailo_apps.my_projects.auto_face_id import manage_body_database


def test_body_cli_uses_body_database_and_samples() -> None:
    with patch.object(manage_body_database.manage_database, "main") as shared_main:
        manage_body_database.main()

    shared_main.assert_called_once_with(
        db_name="persons_body.sqlite3",
        samples_dir=manage_body_database.manage_database.PROJECT_DIR / "body_samples",
        description="Inspect, delete, and repair the person body ID SQLite database.",
    )


def test_body_cli_paths_are_distinct_from_face_defaults() -> None:
    assert manage_body_database.DB_NAME != manage_body_database.manage_database.DB_NAME
    assert manage_body_database.SAMPLES_DIR != manage_body_database.manage_database.SAMPLES_DIR
    assert isinstance(manage_body_database.SAMPLES_DIR, Path)
