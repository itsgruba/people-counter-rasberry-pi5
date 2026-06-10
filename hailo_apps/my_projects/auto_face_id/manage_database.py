#!/usr/bin/env python3
"""Maintenance CLI for the auto face ID SQLite database and sample files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from hailo_apps.my_projects.auto_face_id.sqlite_db_handler import SQLiteDatabaseHandler
except ImportError:
    from sqlite_db_handler import SQLiteDatabaseHandler


PROJECT_DIR = Path(__file__).resolve().parent
DATABASE_DIR = PROJECT_DIR / "database"
SAMPLES_DIR = PROJECT_DIR / "samples"
DB_NAME = "persons.sqlite3"


def _create_db() -> SQLiteDatabaseHandler:
    return SQLiteDatabaseHandler(
        db_name=DB_NAME,
        threshold=0.55,
        database_dir=DATABASE_DIR,
        samples_dir=SAMPLES_DIR,
    )


def _print_summary(db: SQLiteDatabaseHandler) -> None:
    records = db.get_all_records()
    print(f"Database: {db.db_path}")
    print(f"Total people: {len(records)}")
    for record in records:
        print(
            f"- {record['label']}: global_id={record['global_id']} "
            f"samples={len(record['samples_json'])}"
        )


def _resolve_record(db: SQLiteDatabaseHandler, global_id: str | None, label: str | None):
    if global_id:
        record = db.get_record_by_id(global_id)
        if record is None:
            raise KeyError(f"Person not found: {global_id}")
        return record

    if label:
        record = db.get_record_by_label(label)
        if record is None:
            raise KeyError(f"Person not found: {label}")
        return record

    raise ValueError("Either global_id or label must be provided.")


def _sample_files_in_use(db: SQLiteDatabaseHandler) -> set[Path]:
    files: set[Path] = set()
    for record in db.get_all_records():
        for sample in record.get("samples_json") or []:
            sample_path = sample.get("sample_path")
            if sample_path:
                files.add(Path(sample_path).resolve())
    return files


def _delete_orphan_samples(db: SQLiteDatabaseHandler) -> int:
    referenced = _sample_files_in_use(db)
    deleted = 0
    if not db.samples_dir.exists():
        return deleted

    for path in db.samples_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            resolved = path.resolve()
        except FileNotFoundError:
            continue
        if resolved not in referenced:
            path.unlink()
            deleted += 1
    for path in sorted(
        db.samples_dir.rglob("*"),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass
    return deleted


def _repair_embeddings(db: SQLiteDatabaseHandler) -> int:
    updated = 0
    with db._lock, db._connection:  # noqa: SLF001 - maintenance utility uses the same storage layer
        rows = db._connection.execute("SELECT * FROM persons ORDER BY id").fetchall()
        for row in rows:
            samples = db._connection.execute(
                """
                SELECT embedding, received_at
                FROM face_samples
                WHERE person_id = ?
                ORDER BY rowid
                """,
                (row["id"],),
            ).fetchall()
            if not samples:
                continue

            embeddings = [
                np.frombuffer(sample["embedding"], dtype=np.float32) for sample in samples
            ]
            avg_embedding = np.mean(embeddings, axis=0).astype(np.float32)
            last_sample_received_time = samples[-1]["received_at"]
            db._connection.execute(
                """
                UPDATE persons
                SET avg_embedding = ?, last_sample_received_time = ?
                WHERE id = ?
                """,
                (avg_embedding.tobytes(), last_sample_received_time, row["id"]),
            )
            db._embedding_cache[row["global_id"]] = avg_embedding
            updated += 1
    return updated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect, delete, and repair the auto face ID SQLite database."
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("inspect", help="Print a summary of the database.").set_defaults(
        command="inspect"
    )

    delete_parser = subparsers.add_parser(
        "delete-person",
        help="Delete one person and all of their sample files.",
    )
    delete_parser.add_argument("--global-id", help="Delete the person with this global_id.")
    delete_parser.add_argument("--label", help="Delete the person with this label.")
    delete_parser.set_defaults(command="delete-person")

    subparsers.add_parser(
        "clear-all",
        help="Delete every person, sample row, and sample file.",
    ).set_defaults(command="clear-all")

    subparsers.add_parser(
        "prune-samples",
        help="Delete sample files on disk that are no longer referenced by SQLite.",
    ).set_defaults(command="prune-samples")

    subparsers.add_parser(
        "repair",
        help="Recalculate average embeddings from the sample rows currently stored in SQLite.",
    ).set_defaults(command="repair")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not getattr(args, "command", None):
        args.command = "inspect"

    db = _create_db()
    try:
        if args.command == "inspect":
            _print_summary(db)
            return

        if args.command == "delete-person":
            record = _resolve_record(db, args.global_id, args.label)
            db.delete_record(record["global_id"])
            print(
                f"Deleted person: {record['label']} "
                f"({record['global_id']}) and their sample files."
            )
            return

        if args.command == "clear-all":
            db.clear_table()
            print("Deleted every person and every sample file.")
            return

        if args.command == "prune-samples":
            deleted = _delete_orphan_samples(db)
            print(f"Deleted {deleted} orphan sample file(s).")
            return

        if args.command == "repair":
            updated = _repair_embeddings(db)
            deleted = _delete_orphan_samples(db)
            print(
                f"Repaired {updated} person record(s) and deleted {deleted} orphan sample file(s)."
            )
            return

        raise ValueError(f"Unknown command: {args.command}")
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
