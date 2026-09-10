"""Hardware-independent regressions for field embedding collection."""

import ast
from dataclasses import dataclass, field
import logging
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import Mock

import numpy as np

from hailo_apps.my_projects.auto_face_id.sqlite_db_handler import SQLiteDatabaseHandler


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "hailo_apps/my_projects/auto_face_id/person_face_id.py"
)
TREE = ast.parse(SOURCE.read_text())
NAMES = {
    "_normalized_embedding",
    "_aggregate_embeddings",
    "_handle_unknown_person",
    "_record_pending_vote",
    "_stable_pending_vote",
    "_reset_runtime_person_track",
    "_cleanup_stale_person_tracks",
}
nodes = []
for node in TREE.body:
    if isinstance(node, ast.ClassDef) and node.name in {
        "PendingSample",
        "PendingVote",
        "PendingIdentity",
    }:
        nodes.append(node)
    elif isinstance(node, ast.ClassDef) and node.name == "PersonFaceIdApp":
        node.bases = []
        node.body = [
            method
            for method in node.body
            if isinstance(method, ast.FunctionDef) and method.name in NAMES
        ]
        nodes.append(node)

namespace = {
    "dataclass": dataclass,
    "field": field,
    "np": np,
    "Path": Path,
    "time": time,
    "uuid": uuid,
    "logger": logging.getLogger(__name__),
}
module = ast.Module(
    body=[
        ast.ImportFrom(
            module="__future__",
            names=[ast.alias(name="annotations")],
            level=0,
        ),
        *nodes,
    ],
    type_ignores=[],
)
exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
App = namespace["PersonFaceIdApp"]
PendingIdentity = namespace["PendingIdentity"]
PendingSample = namespace["PendingSample"]


class EmbeddingCollectionTests(unittest.TestCase):
    def test_low_quality_photo_still_keeps_embedding(self):
        app = App()
        app.pending_unknowns = {7: PendingIdentity()}
        app.unknown_sample_interval = 1
        app.max_pending_embeddings = 10
        app.min_enroll_confidence = 0.55
        app.recognition_stats = {
            "outside_enroll_zone": 0,
            "invalid_embeddings": 0,
            "candidate_embeddings": 0,
            "candidate_photos": 0,
            "candidate_without_photo": 0,
        }
        app._enroll_if_ready = Mock()
        app._is_inside_enroll_zone = Mock(return_value=True)
        app._is_good_enrollment_sample = Mock(return_value=False)
        app._save_face_sample = Mock(side_effect=AssertionError("bad JPEG must not be saved"))
        app._cleanup_empty_sample_dirs = Mock()
        face = SimpleNamespace(get_confidence=lambda: 0.20)

        app._handle_unknown_person(
            7,
            10,
            np.zeros((8, 8, 3), dtype=np.uint8),
            face,
            object(),
            np.ones(512, dtype=np.float32),
            8,
            8,
        )

        sample = app.pending_unknowns[7].samples[0]
        self.assertIsNone(sample.image_path)
        self.assertAlmostEqual(float(np.linalg.norm(sample.embedding)), 1.0, places=5)
        self.assertEqual(app.recognition_stats["candidate_without_photo"], 1)

    def test_embedding_validation_and_normalized_average(self):
        self.assertIsNone(App._normalized_embedding(np.zeros(512, dtype=np.float32)))
        self.assertIsNone(App._normalized_embedding(np.ones(511, dtype=np.float32)))
        samples = [
            PendingSample(np.ones(512), None, 0.2, 1),
            PendingSample(np.full(512, 2.0), None, 0.3, 2),
        ]
        average = App._aggregate_embeddings(samples)
        self.assertAlmostEqual(float(np.linalg.norm(average)), 1.0, places=5)

    def test_exit_identity_accepts_stable_low_confidence_votes(self):
        app = App()
        app.pending_unknowns = {}
        app.recognition_vote_window = 3
        app.recognition_vote_threshold = 2
        person = {"global_id": "person-id", "label": "person_1"}

        self.assertIsNone(app._record_pending_vote(7, person, 0.05))
        vote = app._record_pending_vote(7, person, 0.10)

        self.assertEqual(vote.global_id, "person-id")
        self.assertEqual(vote.confidence, 0.10)


class PlaceholderBackfillTests(unittest.TestCase):
    def test_embedding_without_jpeg_makes_placeholder_searchable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            database = SQLiteDatabaseHandler(
                db_name="test.sqlite3",
                threshold=0.55,
                database_dir=root,
                samples_dir=root / "samples",
            )
            person = database.create_placeholder_record(1, "person_1")
            database.set_person_inside(person["global_id"], 1, 7)
            embedding = np.arange(1, 513, dtype=np.float32)
            embedding /= np.linalg.norm(embedding)

            database.insert_new_sample(person, embedding, None, 2)
            match = database.search_entered_record_best(embedding)

            self.assertEqual(match["global_id"], person["global_id"])
            self.assertIsNone(match["samples_json"][0]["sample_path"])
            database.close()


class TrackReuseTests(unittest.TestCase):
    def test_recycled_track_id_does_not_reuse_previous_person(self):
        app = App()
        app.identity_track_max_gap_frames = 8
        app.person_track_last_seen_frame = {43: 1}
        app.track_to_global_id = {43: "old-person"}
        app.track_to_label = {43: "person_2"}
        app.last_printed_identity = {43: "old-person:person_2:recognized"}
        app.known_track_last_sample_frame = {43: 1}
        app.pending_entry_contexts = {}
        app.entry_detector = SimpleNamespace(
            tracks={43: object()},
            has_uncounted_crossing=lambda track_id: False,
        )
        app.exit_detector = SimpleNamespace(tracks={43: object()})
        app._discard_pending_identity = Mock()

        app._cleanup_stale_person_tracks({43}, frame_number=10)

        self.assertNotIn(43, app.track_to_global_id)
        self.assertNotIn(43, app.track_to_label)
        self.assertNotIn(43, app.entry_detector.tracks)
        self.assertNotIn(43, app.exit_detector.tracks)
        self.assertEqual(app.person_track_last_seen_frame[43], 10)


if __name__ == "__main__":
    unittest.main()
