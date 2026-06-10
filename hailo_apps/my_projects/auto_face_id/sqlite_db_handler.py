"""SQLite storage and in-memory vector search for the auto face ID project."""

from __future__ import annotations

import os
import shutil
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np


EMBEDDING_DIMENSION = 512


class SQLiteDatabaseHandler:
    """Store face identities in SQLite and search embeddings with NumPy."""

    def __init__(
        self,
        db_name: str,
        threshold: float,
        database_dir: str | Path,
        samples_dir: str | Path,
    ) -> None:
        self.database_dir = Path(database_dir)
        self.samples_dir = Path(samples_dir)
        self.database_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.database_dir / db_name
        self.classificaiton_confidence_threshold = threshold
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA synchronous = NORMAL")
        self._create_schema()
        self._ensure_person_columns()
        self._embedding_cache: dict[str, np.ndarray] = {}
        self._sample_embedding_cache: list[tuple[str, np.ndarray]] = []
        self._reload_embedding_cache()

    def _create_schema(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    global_id TEXT UNIQUE NOT NULL,
                    label TEXT UNIQUE NOT NULL,
                    avg_embedding BLOB NOT NULL,
                    last_sample_received_time INTEGER NOT NULL,
                    classification_confidence_threshold REAL NOT NULL,
                    value REAL NOT NULL DEFAULT 0.0,
                    visits_count INTEGER NOT NULL DEFAULT 0,
                    last_seen_at INTEGER,
                    last_seen_track_id INTEGER,
                    created_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS face_samples (
                    id TEXT PRIMARY KEY,
                    person_id INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    sample_path TEXT,
                    received_at INTEGER NOT NULL,
                    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_face_samples_person_id
                ON face_samples(person_id);

                CREATE TABLE IF NOT EXISTS visits (
                    id TEXT PRIMARY KEY,
                    person_id INTEGER NOT NULL,
                    visit_number INTEGER NOT NULL,
                    visited_at INTEGER NOT NULL,
                    photo_path TEXT NOT NULL,
                    track_id INTEGER,
                    created_at INTEGER NOT NULL,
                    UNIQUE(person_id, visit_number),
                    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_visits_person_id
                ON visits(person_id);
                """
            )

    def _ensure_person_columns(self) -> None:
        """Add new columns to older databases without requiring a manual migration."""
        columns = {
            row["name"]
            for row in self._connection.execute("PRAGMA table_info(persons)").fetchall()
        }
        migrations = [
            (
                "visits_count",
                "ALTER TABLE persons ADD COLUMN visits_count INTEGER NOT NULL DEFAULT 0",
            ),
            ("last_seen_at", "ALTER TABLE persons ADD COLUMN last_seen_at INTEGER"),
            (
                "last_seen_track_id",
                "ALTER TABLE persons ADD COLUMN last_seen_track_id INTEGER",
            ),
        ]
        with self._connection:
            for column_name, statement in migrations:
                if column_name not in columns:
                    self._connection.execute(statement)

    @staticmethod
    def _embedding_array(embedding: np.ndarray | list[float]) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.size != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Expected a {EMBEDDING_DIMENSION}-value embedding, got {vector.size}."
            )
        return vector

    @classmethod
    def _embedding_blob(cls, embedding: np.ndarray | list[float]) -> bytes:
        return cls._embedding_array(embedding).tobytes()

    @staticmethod
    def _blob_embedding(blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32).copy()

    @staticmethod
    def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator == 0.0:
            return 1.0
        similarity = float(np.dot(left, right) / denominator)
        return 1.0 - max(-1.0, min(1.0, similarity))

    def _reload_embedding_cache(self) -> None:
        with self._lock:
            rows = self._connection.execute(
                "SELECT global_id, avg_embedding FROM persons"
            ).fetchall()
            self._embedding_cache = {
                row["global_id"]: self._blob_embedding(row["avg_embedding"]) for row in rows
            }
            sample_rows = self._connection.execute(
                """
                SELECT persons.global_id, face_samples.embedding
                FROM face_samples
                JOIN persons ON persons.id = face_samples.person_id
                ORDER BY face_samples.rowid
                """
            ).fetchall()
            self._sample_embedding_cache = [
                (row["global_id"], self._blob_embedding(row["embedding"]))
                for row in sample_rows
            ]

    def _sample_rows(self, person_id: int) -> list[sqlite3.Row]:
        return self._connection.execute(
            """
            SELECT id, embedding, sample_path, received_at
            FROM face_samples
            WHERE person_id = ?
            ORDER BY rowid
            """,
            (person_id,),
        ).fetchall()

    def _visit_rows(self, person_id: int) -> list[sqlite3.Row]:
        return self._connection.execute(
            """
            SELECT id, visit_number, visited_at, photo_path, track_id
            FROM visits
            WHERE person_id = ?
            ORDER BY visit_number
            """,
            (person_id,),
        ).fetchall()

    def _row_to_record(self, row: sqlite3.Row, distance: float | None = None) -> dict[str, Any]:
        samples = [
            {
                "id": sample["id"],
                "embedding": self._blob_embedding(sample["embedding"]).tolist(),
                "sample_path": sample["sample_path"],
                "timestamp": sample["received_at"],
            }
            for sample in self._sample_rows(row["id"])
        ]
        record: dict[str, Any] = {
            "global_id": row["global_id"],
            "label": row["label"],
            "avg_embedding": self._blob_embedding(row["avg_embedding"]).tolist(),
            "last_sample_recieved_time": row["last_sample_received_time"],
            "samples_json": samples,
            "visits_json": [
                {
                    "id": visit["id"],
                    "visit_number": visit["visit_number"],
                    "timestamp": visit["visited_at"],
                    "photo_path": visit["photo_path"],
                    "track_id": visit["track_id"],
                }
                for visit in self._visit_rows(row["id"])
            ],
            "classificaiton_confidence_threshold": row[
                "classification_confidence_threshold"
            ],
            "visit_count": int(
                row["visits_count"] if "visits_count" in row.keys() else 0
            ),
            "last_seen_at": row["last_seen_at"] if "last_seen_at" in row.keys() else None,
            "last_seen_track_id": (
                row["last_seen_track_id"] if "last_seen_track_id" in row.keys() else None
            ),
        }
        if distance is not None:
            record["_distance"] = distance
        return record

    @staticmethod
    def _unknown_record() -> dict[str, Any]:
        return {
            "global_id": str(uuid.uuid4()),
            "label": "Unknown",
            "avg_embedding": None,
            "last_sample_recieved_time": None,
            "samples_json": None,
            "classificaiton_confidence_threshold": None,
            "_distance": 0.0,
        }

    def create_record(
        self,
        embedding: np.ndarray,
        sample: str | None,
        timestamp: int,
        label: str = "Unknown",
        global_id: str | None = None,
        threshold: float | None = None,
        visits_count: int = 1,
        last_seen_track_id: int | None = None,
    ) -> dict[str, Any]:
        vector = self._embedding_array(embedding)
        global_id = global_id or str(uuid.uuid4())
        threshold = (
            self.classificaiton_confidence_threshold if threshold is None else threshold
        )
        sample_id = str(uuid.uuid4())

        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO persons (
                    global_id,
                    label,
                    avg_embedding,
                    last_sample_received_time,
                    classification_confidence_threshold,
                    value,
                    visits_count,
                    created_at,
                    last_seen_at,
                    last_seen_track_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    global_id,
                    label,
                    vector.tobytes(),
                    timestamp,
                    threshold,
                    visits_count,
                    visits_count,
                    timestamp,
                    timestamp,
                    last_seen_track_id,
                ),
            )
            self._connection.execute(
                """
                INSERT INTO face_samples (id, person_id, embedding, sample_path, received_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (sample_id, cursor.lastrowid, vector.tobytes(), sample, timestamp),
            )
            self._embedding_cache[global_id] = vector.copy()
            self._sample_embedding_cache.append((global_id, vector.copy()))

        return self.get_record_by_id(global_id)

    def mark_person_seen(
        self,
        global_id: str,
        timestamp: int,
        track_id: int | None = None,
    ) -> dict[str, Any] | None:
        """Register a visible person and increment the visit counter for a new track."""
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT * FROM persons WHERE global_id = ?",
                (global_id,),
            ).fetchone()
            if row is None:
                return None

            last_track_id = (
                row["last_seen_track_id"] if "last_seen_track_id" in row.keys() else None
            )
            should_increment = track_id is not None and track_id != last_track_id

            if should_increment:
                self._connection.execute(
                    """
                    UPDATE persons
                    SET visits_count = visits_count + 1,
                        last_seen_at = ?,
                        last_seen_track_id = ?
                    WHERE global_id = ?
                    """,
                    (timestamp, track_id, global_id),
                )
            else:
                self._connection.execute(
                    """
                    UPDATE persons
                    SET last_seen_at = ?,
                        last_seen_track_id = ?
                    WHERE global_id = ?
                    """,
                    (timestamp, track_id, global_id),
                )

            updated_row = self._connection.execute(
                "SELECT * FROM persons WHERE global_id = ?",
                (global_id,),
            ).fetchone()
            if updated_row is None:
                return None

            record = self._row_to_record(updated_row)
            record["visit_incremented"] = should_increment
            return record

    def add_visit_record(
        self,
        global_id: str,
        visit_number: int,
        timestamp: int,
        photo_path: str,
        track_id: int | None = None,
        visit_id: str | None = None,
    ) -> dict[str, Any]:
        """Store the visual evidence for one visit_count increment."""
        visit_id = visit_id or str(uuid.uuid4())
        with self._lock, self._connection:
            person = self._connection.execute(
                "SELECT id FROM persons WHERE global_id = ?",
                (global_id,),
            ).fetchone()
            if person is None:
                raise KeyError(f"Person not found: {global_id}")

            self._connection.execute(
                """
                INSERT INTO visits (
                    id,
                    person_id,
                    visit_number,
                    visited_at,
                    photo_path,
                    track_id,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(person_id, visit_number) DO UPDATE SET
                    visited_at = excluded.visited_at,
                    photo_path = excluded.photo_path,
                    track_id = excluded.track_id
                """,
                (
                    visit_id,
                    person["id"],
                    visit_number,
                    timestamp,
                    photo_path,
                    track_id,
                    int(time.time()),
                ),
            )
            row = self._connection.execute(
                """
                SELECT id, visit_number, visited_at, photo_path, track_id
                FROM visits
                WHERE person_id = ? AND visit_number = ?
                """,
                (person["id"], visit_number),
            ).fetchone()
            if row is None:
                raise RuntimeError("Failed to store visit record.")
            return {
                "id": row["id"],
                "visit_number": row["visit_number"],
                "timestamp": row["visited_at"],
                "photo_path": row["photo_path"],
                "track_id": row["track_id"],
            }

    def insert_new_sample(
        self,
        record: dict[str, Any],
        embedding: np.ndarray,
        sample: str,
        timestamp: int,
        sample_id: str | None = None,
    ) -> None:
        vector = self._embedding_array(embedding)
        sample_id = sample_id or str(uuid.uuid4())
        global_id = record["global_id"]

        with self._lock, self._connection:
            person = self._connection.execute(
                "SELECT id FROM persons WHERE global_id = ?", (global_id,)
            ).fetchone()
            if person is None:
                raise KeyError(f"Person not found: {global_id}")

            self._connection.execute(
                """
                INSERT INTO face_samples (id, person_id, embedding, sample_path, received_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (sample_id, person["id"], vector.tobytes(), sample, timestamp),
            )
            sample_rows = self._sample_rows(person["id"])
            avg_embedding = np.mean(
                [self._blob_embedding(row["embedding"]) for row in sample_rows], axis=0
            ).astype(np.float32)
            self._connection.execute(
                """
                UPDATE persons
                SET avg_embedding = ?, last_sample_received_time = ?
                WHERE id = ?
                """,
                (avg_embedding.tobytes(), timestamp, person["id"]),
            )
            self._embedding_cache[global_id] = avg_embedding
            self._sample_embedding_cache.append((global_id, vector.copy()))

    def search_record(
        self,
        embedding: np.ndarray,
        top_k: int = 1,
        metric_type: str = "cosine",
    ) -> dict[str, Any]:
        if metric_type != "cosine":
            raise ValueError("SQLiteDatabaseHandler currently supports only cosine distance.")
        if top_k != 1:
            raise ValueError("SQLiteDatabaseHandler currently returns only the best match.")

        query = self._embedding_array(embedding)
        with self._lock:
            if not self._embedding_cache:
                return self._unknown_record()

            global_id, distance = min(
                (
                    (candidate_id, self._cosine_distance(query, candidate_embedding))
                    for candidate_id, candidate_embedding in self._embedding_cache.items()
                ),
                key=lambda item: item[1],
            )
            row = self._connection.execute(
                "SELECT * FROM persons WHERE global_id = ?", (global_id,)
            ).fetchone()
            if row is None:
                return self._unknown_record()

            confidence = 1.0 - distance
            if confidence > row["classification_confidence_threshold"]:
                return self._row_to_record(row, distance=distance)
            return self._unknown_record()

    def search_record_deep(
        self,
        embedding: np.ndarray,
        metric_type: str = "cosine",
    ) -> dict[str, Any]:
        """Search average person embeddings plus each stored face sample."""
        if metric_type != "cosine":
            raise ValueError("SQLiteDatabaseHandler currently supports only cosine distance.")

        query = self._embedding_array(embedding)
        with self._lock:
            candidates: list[tuple[str, float]] = [
                (candidate_id, self._cosine_distance(query, candidate_embedding))
                for candidate_id, candidate_embedding in self._embedding_cache.items()
            ]
            candidates.extend(
                (candidate_id, self._cosine_distance(query, sample_embedding))
                for candidate_id, sample_embedding in self._sample_embedding_cache
            )
            if not candidates:
                return self._unknown_record()

            global_id, distance = min(candidates, key=lambda item: item[1])
            row = self._connection.execute(
                "SELECT * FROM persons WHERE global_id = ?", (global_id,)
            ).fetchone()
            if row is None:
                return self._unknown_record()

            confidence = 1.0 - distance
            if confidence > row["classification_confidence_threshold"]:
                return self._row_to_record(row, distance=distance)
            return self._unknown_record()

    def get_all_records(self, only_unknowns: bool = False) -> list[dict[str, Any]]:
        with self._lock:
            if only_unknowns:
                rows = self._connection.execute(
                    "SELECT * FROM persons WHERE label = 'Unknown' ORDER BY id"
                ).fetchall()
            else:
                rows = self._connection.execute("SELECT * FROM persons ORDER BY id").fetchall()
            return [self._row_to_record(row) for row in rows]

    def get_people_cards(self) -> list[dict[str, Any]]:
        """Return lightweight person summaries for a frontend dashboard."""
        cards = []
        for record in self.get_all_records():
            thumbnail_path = None
            if record["samples_json"]:
                sample_path = record["samples_json"][0]["sample_path"]
                if sample_path:
                    thumbnail_path = self._sample_relative_path(sample_path)
            cards.append(
                {
                    "global_id": record["global_id"],
                    "label": record["label"],
                    "visit_count": int(record["visit_count"]),
                    "last_seen_at": record["last_seen_at"],
                    "thumbnail_name": thumbnail_path,
                    "sample_count": len(record["samples_json"] or []),
                }
            )
        return cards

    def get_record_by_id(self, global_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM persons WHERE global_id = ?", (global_id,)
            ).fetchone()
            return self._row_to_record(row) if row is not None else None

    def get_record_by_label(self, label: str = "Unknown") -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM persons WHERE label = ?", (label,)
            ).fetchone()
            return self._row_to_record(row) if row is not None else None

    def update_record_label(self, global_id: str, label: str = "Unknown") -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE persons SET label = ? WHERE global_id = ?", (label, global_id)
            )

    def delete_record(self, global_id: str) -> None:
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT id FROM persons WHERE global_id = ?", (global_id,)
            ).fetchone()
            if row is None:
                return
            sample_paths = [
                sample["sample_path"] for sample in self._sample_rows(row["id"])
            ]
            visit_paths = [
                visit["photo_path"] for visit in self._visit_rows(row["id"])
            ]
            self._connection.execute("DELETE FROM persons WHERE id = ?", (row["id"],))
            self._embedding_cache.pop(global_id, None)
            self._sample_embedding_cache = [
                (candidate_id, embedding)
                for candidate_id, embedding in self._sample_embedding_cache
                if candidate_id != global_id
            ]
        for sample_path in [*sample_paths, *visit_paths]:
            self._delete_sample_file(sample_path)

    def clear_table(self) -> None:
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM persons")
            self._connection.execute("DELETE FROM sqlite_sequence WHERE name = 'persons'")
            self._embedding_cache.clear()
            self._sample_embedding_cache.clear()
        for sample_path in self.samples_dir.iterdir():
            if sample_path.is_dir():
                shutil.rmtree(sample_path)
            elif sample_path.is_file():
                os.remove(sample_path)
        print("All records deleted from the SQLite database")

    def _sample_relative_path(self, sample_path: str | None) -> str | None:
        if not sample_path:
            return None
        path = Path(sample_path)
        try:
            return path.resolve().relative_to(self.samples_dir.resolve()).as_posix()
        except (OSError, ValueError):
            return path.name

    def _delete_sample_file(self, sample_path: str | None) -> None:
        if not sample_path:
            return
        path = Path(sample_path)
        try:
            path.resolve().relative_to(self.samples_dir.resolve())
        except ValueError:
            return
        if path.exists() and path.is_file():
            os.remove(path)
            self._delete_empty_sample_dirs(path.parent)

    def _delete_empty_sample_dirs(self, start_dir: Path) -> None:
        try:
            current = start_dir.resolve()
            samples_root = self.samples_dir.resolve()
            current.relative_to(samples_root)
        except (OSError, ValueError):
            return

        while current != samples_root:
            try:
                current.rmdir()
            except OSError:
                return
            current = current.parent

    def close(self) -> None:
        with self._lock:
            self._connection.close()
