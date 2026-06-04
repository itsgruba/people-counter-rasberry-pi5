#!/usr/bin/env python3
"""Automatic face ID enrollment and recognition from a live camera stream."""

from __future__ import annotations

import argparse
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import gi
import hailo
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from hailo_apps.python.core.common.buffer_utils import (
    get_caps_from_pad,
    get_numpy_from_buffer_efficient,
)
from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.core.common.parser import get_pipeline_parser
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.python.pipeline_apps.face_recognition.face_recognition_pipeline import (
    GStreamerFaceRecognitionApp,
)
from hailo_apps.my_projects.auto_face_id.sqlite_db_handler import SQLiteDatabaseHandler

logger = get_logger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
DATABASE_DIR = PROJECT_DIR / "database"
SAMPLES_DIR = PROJECT_DIR / "samples"
DB_NAME = "persons.sqlite3"


@dataclass
class PendingSample:
    """One candidate sample for a newly observed unknown face."""

    embedding: np.ndarray
    image_path: str
    confidence: float
    timestamp: int


@dataclass
class PendingIdentity:
    """Samples accumulated for a tracker ID before creating a permanent identity."""

    samples: list[PendingSample] = field(default_factory=list)
    last_sample_frame: int = -1


class AutoFaceIdData(app_callback_class):
    """Application state shared with the GStreamer callback."""

    def __init__(self) -> None:
        super().__init__()
        self.latest_track_id = -1


class AutoFaceIdApp(GStreamerFaceRecognitionApp):
    """Face recognition pipeline with automatic online enrollment."""

    def __init__(self, app_callback, user_data, parser: argparse.ArgumentParser | None = None):
        parser = parser or self._build_parser()
        super().__init__(app_callback, user_data, parser)

        self.samples_per_person = self.options_menu.samples_per_person
        self.unknown_sample_interval = self.options_menu.unknown_sample_interval
        self.min_enroll_confidence = self.options_menu.min_enroll_confidence
        self.print_every_frame = self.options_menu.print_every_frame

        DATABASE_DIR.mkdir(parents=True, exist_ok=True)
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        self.database_dir = DATABASE_DIR
        self.samples_dir = SAMPLES_DIR
        self.db_handler = SQLiteDatabaseHandler(
            db_name=DB_NAME,
            threshold=self.lance_db_vector_search_classificaiton_confidence_threshold,
            database_dir=str(self.database_dir),
            samples_dir=str(self.samples_dir),
        )

        self.pending_unknowns: dict[int, PendingIdentity] = {}
        self.track_to_global_id: dict[int, str] = {}
        self.track_to_label: dict[int, str] = {}
        self.last_printed_identity: dict[int, str] = {}
        self.next_person_index = self._load_next_person_index()

        logger.info("Auto Face ID database: %s", self.database_dir / DB_NAME)
        logger.info("Auto Face ID samples: %s", self.samples_dir)

    @staticmethod
    def _build_parser() -> argparse.ArgumentParser:
        parser = get_pipeline_parser()
        parser.description = "Automatic face ID enrollment and recognition."
        parser.add_argument(
            "--samples-per-person",
            type=int,
            default=5,
            help="Number of unknown-face samples required before creating a new person.",
        )
        parser.add_argument(
            "--unknown-sample-interval",
            type=int,
            default=5,
            help="Minimum frames between samples collected for the same unknown track.",
        )
        parser.add_argument(
            "--min-enroll-confidence",
            type=float,
            default=0.55,
            help="Minimum face detection confidence for automatic enrollment samples.",
        )
        parser.add_argument(
            "--print-every-frame",
            action="store_true",
            help="Print known identities every processed callback instead of only on changes.",
        )
        return parser

    def _load_next_person_index(self) -> int:
        max_index = 0
        for record in self.db_handler.get_all_records():
            label = record.get("label", "")
            if label.startswith("person_"):
                suffix = label.removeprefix("person_")
                if suffix.isdigit():
                    max_index = max(max_index, int(suffix))
        return max_index + 1

    def _make_person_label(self) -> str:
        label = f"person_{self.next_person_index}"
        self.next_person_index += 1
        return label

    def _add_identity_classification(
        self,
        detection,
        tracker_name: str,
        track_id: int,
        label: str,
        confidence: float,
    ) -> None:
        classifications = detection.get_objects_typed(hailo.HAILO_CLASSIFICATION)
        for classification in classifications:
            detection.remove_object(classification)

        new_classification = hailo.HailoClassification(
            type="face_recon",
            label=label,
            confidence=confidence,
        )
        detection.add_object(new_classification)
        self.tracker.remove_classifications_from_track(tracker_name, track_id, "face_recon")
        self.tracker.add_object_to_track(tracker_name, track_id, new_classification)

    def _print_identity(
        self,
        track_id: int,
        global_id: str,
        label: str,
        confidence: float,
        state: str,
    ) -> None:
        identity_key = f"{global_id}:{label}:{state}"
        if not self.print_every_frame and self.last_printed_identity.get(track_id) == identity_key:
            return
        self.last_printed_identity[track_id] = identity_key
        print(
            f"{state}: track_id={track_id} global_id={global_id} "
            f"label={label} confidence={confidence:.2f}"
        )

    def _save_face_sample(self, frame: np.ndarray, detection, width: int, height: int) -> str:
        image_path = self.samples_dir / f"{uuid.uuid4()}.jpeg"
        cropped = self.crop_frame(frame, detection.get_bbox(), width, height)
        self.save_image_file(cropped, str(image_path))
        return str(image_path)

    def _recognize_embedding(self, embedding_vector: np.ndarray) -> tuple[dict, float]:
        person = self.db_handler.search_record(embedding=embedding_vector)
        confidence = 1 - person["_distance"]
        return person, confidence

    def _enroll_if_ready(
        self,
        track_id: int,
        detection,
        tracker_name: str,
    ) -> None:
        pending = self.pending_unknowns.get(track_id)
        if pending is None or len(pending.samples) < self.samples_per_person:
            return

        avg_embedding = np.mean([sample.embedding for sample in pending.samples], axis=0)
        person, confidence = self._recognize_embedding(avg_embedding)
        if person["label"] != "Unknown":
            self.track_to_global_id[track_id] = person["global_id"]
            self.track_to_label[track_id] = person["label"]
            self._add_identity_classification(
                detection,
                tracker_name,
                track_id,
                person["label"],
                confidence,
            )
            self._print_identity(
                track_id,
                person["global_id"],
                person["label"],
                confidence,
                "recognized-after-samples",
            )
            self.pending_unknowns.pop(track_id, None)
            return

        label = self._make_person_label()
        best_sample = max(pending.samples, key=lambda sample: sample.confidence)
        new_person = self.db_handler.create_record(
            embedding=avg_embedding,
            sample=best_sample.image_path,
            timestamp=int(time.time()),
            label=label,
        )

        for sample in pending.samples:
            if sample.image_path == best_sample.image_path:
                continue
            self.db_handler.insert_new_sample(
                record=self.db_handler.get_record_by_id(new_person["global_id"]),
                embedding=sample.embedding,
                sample=sample.image_path,
                timestamp=sample.timestamp,
            )

        self.track_to_global_id[track_id] = new_person["global_id"]
        self.track_to_label[track_id] = label
        self._add_identity_classification(detection, tracker_name, track_id, label, 1.0)
        self._print_identity(track_id, new_person["global_id"], label, 1.0, "enrolled")
        self.pending_unknowns.pop(track_id, None)

    def _handle_unknown_face(
        self,
        track_id: int,
        frame_number: int,
        frame: np.ndarray,
        detection,
        embedding_vector: np.ndarray,
        width: int,
        height: int,
        tracker_name: str,
    ) -> None:
        detection_confidence = detection.get_confidence()
        if detection_confidence < self.min_enroll_confidence:
            return

        pending = self.pending_unknowns.setdefault(track_id, PendingIdentity())
        if (
            pending.last_sample_frame >= 0
            and frame_number - pending.last_sample_frame < self.unknown_sample_interval
        ):
            return

        image_path = self._save_face_sample(frame, detection, width, height)
        pending.samples.append(
            PendingSample(
                embedding=embedding_vector,
                image_path=image_path,
                confidence=detection_confidence,
                timestamp=int(time.time()),
            )
        )
        pending.last_sample_frame = frame_number
        print(
            f"collecting: track_id={track_id} samples="
            f"{len(pending.samples)}/{self.samples_per_person}"
        )
        self._enroll_if_ready(track_id, detection, tracker_name)

    def vector_db_callback(self, pad, info, user_data):
        tracker_name = self.tracker.get_trackers_list()[0]
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK

        frame_number = user_data.get_count()
        fmt, width, height = get_caps_from_pad(pad)
        roi = hailo.get_roi_from_buffer(buffer)

        frame = None
        for detection in (
            d for d in roi.get_objects_typed(hailo.HAILO_DETECTION) if d.get_label() == "face"
        ):
            track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if not track:
                continue
            track_id = track[0].get_id()

            embedding = detection.get_objects_typed(hailo.HAILO_MATRIX)
            if len(embedding) != 1:
                continue
            embedding_vector = np.array(embedding[0].get_data())
            detection.remove_object(embedding[0])

            person, confidence = self._recognize_embedding(embedding_vector)
            if person["label"] != "Unknown":
                self.track_to_global_id[track_id] = person["global_id"]
                self.track_to_label[track_id] = person["label"]
                self.pending_unknowns.pop(track_id, None)
                self._add_identity_classification(
                    detection,
                    tracker_name,
                    track_id,
                    person["label"],
                    confidence,
                )
                self._print_identity(
                    track_id,
                    person["global_id"],
                    person["label"],
                    confidence,
                    "recognized",
                )
                continue

            if frame is None:
                frame = get_numpy_from_buffer_efficient(buffer, fmt, width, height)
            self._handle_unknown_face(
                track_id,
                frame_number,
                frame,
                detection,
                embedding_vector,
                width,
                height,
                tracker_name,
            )

        return Gst.PadProbeReturn.OK


def app_callback(element, buffer, user_data):
    """Final display callback. Identity work happens in vector_db_callback."""
    return Gst.FlowReturn.OK


def main() -> None:
    user_data = AutoFaceIdData()
    app = AutoFaceIdApp(app_callback, user_data)
    app.run()


if __name__ == "__main__":
    main()
