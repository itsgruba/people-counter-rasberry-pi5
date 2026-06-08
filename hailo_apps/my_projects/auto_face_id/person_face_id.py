#!/usr/bin/env python3
"""Person tracking with face detection/recognition merged from parallel Hailo branches."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import gi
import hailo
import numpy as np
from hailo import HailoTracker

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from PIL import Image
import cv2

from hailo_apps.python.core.common.buffer_utils import (
    get_caps_from_pad,
    get_numpy_from_buffer_efficient,
)
from hailo_apps.python.core.common.hef_utils import get_hef_labels_json
from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.core.common.core import (
    resolve_hef_path,
    resolve_hef_paths,
    get_resource_path,
)
from hailo_apps.python.core.common.defines import (
    ARCFACE_MOBILEFACENET_POSTPROCESS_FUNCTION,
    DETECTION_PIPELINE,
    DETECTION_POSTPROCESS_FUNCTION,
    DETECTION_POSTPROCESS_SO_FILENAME,
    FACE_ALIGN_POSTPROCESS_SO_FILENAME,
    FACE_CROP_POSTPROCESS_SO_FILENAME,
    FACE_DETECTION_JSON_NAME,
    FACE_DETECTION_POSTPROCESS_SO_FILENAME,
    FACE_RECOGNITION_PIPELINE,
    FACE_RECOGNITION_POSTPROCESS_SO_FILENAME,
    HAILO10H_ARCH,
    HAILO8L_ARCH,
    HAILO8_ARCH,
    CLIP_CROPPER_OBJECT_POSTPROCESS_FUNCTION_NAME,
    CLIP_CROPPER_POSTPROCESS_SO_FILENAME,
    SCRFD_10G_POSTPROCESS_FUNCTION,
    SCRFD_2_5G_POSTPROCESS_FUNCTION,
    RESOURCES_JSON_DIR_NAME,
    RESOURCES_SO_DIR_NAME,
)
from hailo_apps.python.core.common.parser import get_pipeline_parser
from hailo_apps.python.core.common.hailo_logger import init_logging, level_from_args
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class, GStreamerApp
from hailo_apps.python.core.gstreamer.gstreamer_helper_pipelines import (
    CROPPER_PIPELINE,
    DISPLAY_PIPELINE,
    INFERENCE_PIPELINE,
    INFERENCE_PIPELINE_WRAPPER,
    QUEUE,
    TRACKER_PIPELINE,
    USER_CALLBACK_PIPELINE,
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


class PersonFaceIdData(app_callback_class):
    """Shared state for the GStreamer callback."""

    def __init__(self) -> None:
        super().__init__()
        self.latest_track_id = -1


class PersonFaceIdApp(GStreamerApp):
    """Detect persons, then use faces inside person ROIs for recognition."""

    def __init__(self, user_data, parser: argparse.ArgumentParser | None = None):
        parser = parser or self._build_parser()
        super().__init__(parser, user_data)

        if hasattr(self.options_menu, "log_level") or hasattr(self.options_menu, "debug"):
            init_logging(
                level=level_from_args(self.options_menu),
                log_file=getattr(self.options_menu, "log_file", None),
            )

        self.samples_per_person = self.options_menu.samples_per_person
        self.unknown_sample_interval = self.options_menu.unknown_sample_interval
        self.min_enroll_confidence = self.options_menu.min_enroll_confidence
        self.print_every_frame = self.options_menu.print_every_frame
        self.notify_url = self.options_menu.notify_url
        self.person_class_id = self.options_menu.person_class_id
        self.debug_face_overlay = self.options_menu.use_frame
        self._shutdown_started = False
        self.face_tracker_name = "hailo_face_tracker"
        self.person_tracker_name = "person_tracker"
        self.tracker = HailoTracker.get_instance()

        DATABASE_DIR.mkdir(parents=True, exist_ok=True)
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        self.database_dir = DATABASE_DIR
        self.samples_dir = SAMPLES_DIR

        self.person_hef_path = resolve_hef_path(
            self.options_menu.person_hef_path,
            app_name=DETECTION_PIPELINE,
            arch=self.arch,
        )
        if self.person_hef_path is None:
            raise RuntimeError("Failed to resolve person detection HEF.")

        face_models = resolve_hef_paths(
            self.options_menu.face_hef_path,
            app_name=FACE_RECOGNITION_PIPELINE,
            arch=self.arch,
        )
        self.face_detection_hef_path = face_models[0].path
        self.face_recognition_hef_path = face_models[1].path

        self.person_labels_json = get_hef_labels_json(self.person_hef_path)
        self.face_labels_json = get_resource_path(
            pipeline_name=None,
            resource_type=RESOURCES_JSON_DIR_NAME,
            arch=self.arch,
            model=FACE_DETECTION_JSON_NAME,
        )

        self.person_post_process_so = get_resource_path(
            pipeline_name=None,
            resource_type=RESOURCES_SO_DIR_NAME,
            arch=self.arch,
            model=DETECTION_POSTPROCESS_SO_FILENAME,
        )
        self.face_post_process_so = get_resource_path(
            pipeline_name=None,
            resource_type=RESOURCES_SO_DIR_NAME,
            arch=self.arch,
            model=FACE_DETECTION_POSTPROCESS_SO_FILENAME,
        )
        self.face_recognition_post_process_so = get_resource_path(
            pipeline_name=None,
            resource_type=RESOURCES_SO_DIR_NAME,
            arch=self.arch,
            model=FACE_RECOGNITION_POSTPROCESS_SO_FILENAME,
        )
        self.face_align_post_process_so = get_resource_path(
            pipeline_name=None,
            resource_type=RESOURCES_SO_DIR_NAME,
            arch=self.arch,
            model=FACE_ALIGN_POSTPROCESS_SO_FILENAME,
        )
        self.face_cropper_post_process_so = get_resource_path(
            pipeline_name=None,
            resource_type=RESOURCES_SO_DIR_NAME,
            arch=self.arch,
            model=FACE_CROP_POSTPROCESS_SO_FILENAME,
        )
        self.person_cropper_post_process_so = get_resource_path(
            pipeline_name=None,
            resource_type=RESOURCES_SO_DIR_NAME,
            arch=self.arch,
            model=CLIP_CROPPER_POSTPROCESS_SO_FILENAME,
        )

        if self.arch in (HAILO8_ARCH, HAILO10H_ARCH):
            self.face_detection_func = SCRFD_10G_POSTPROCESS_FUNCTION
        elif self.arch == HAILO8L_ARCH:
            self.face_detection_func = SCRFD_2_5G_POSTPROCESS_FUNCTION
        else:
            raise RuntimeError(f"Unsupported Hailo architecture: {self.arch}")

        self.person_post_function_name = DETECTION_POSTPROCESS_FUNCTION
        self.face_recognition_post_function_name = ARCFACE_MOBILEFACENET_POSTPROCESS_FUNCTION
        self.person_cropper_function_name = CLIP_CROPPER_OBJECT_POSTPROCESS_FUNCTION_NAME
        self.face_cropper_function_name = "face_recognition"

        self.person_thresholds_str = (
            "nms-score-threshold=0.3 "
            "nms-iou-threshold=0.45 "
            "output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        )

        self.db_handler = SQLiteDatabaseHandler(
            db_name=DB_NAME,
            threshold=0.55,
            database_dir=str(self.database_dir),
            samples_dir=str(self.samples_dir),
        )

        self.pending_unknowns: dict[int, PendingIdentity] = {}
        self.face_track_embeddings: dict[int, np.ndarray] = {}
        self.track_to_global_id: dict[int, str] = {}
        self.track_to_label: dict[int, str] = {}
        self.last_printed_identity: dict[int, str] = {}
        self.recognition_stats = self._new_recognition_stats()
        self.next_person_index = self._load_next_person_index()

        self.app_callback = self.pipeline_callback

        logger.info("Person-face database: %s", self.database_dir / DB_NAME)
        logger.info("Person-face samples: %s", self.samples_dir)
        logger.info("Person-face database records: %d", len(self.db_handler.get_all_records()))

        self.create_pipeline()
        self._connect_face_embedding_callback()

    @staticmethod
    def _build_parser() -> argparse.ArgumentParser:
        parser = get_pipeline_parser()
        parser.description = "Track person detections and run face detection/recognition inside each person ROI."
        parser.add_argument(
            "--person-hef-path",
            default=None,
            help="Path or model name for the person detection HEF.",
        )
        parser.add_argument(
            "--face-hef-path",
            action="append",
            default=None,
            help=(
                "Path or model name for the face detection and face recognition HEFs. "
                "Provide two values if overriding the defaults."
            ),
        )
        parser.add_argument(
            "--person-class-id",
            type=int,
            default=1,
            help="Class ID to track as person. Default is 1 for the Hailo detection post-process.",
        )
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
        parser.add_argument(
            "--notify-url",
            default=None,
            help=(
                "Optional HTTP endpoint that receives JSON notifications when a person is "
                "recognized or enrolled."
            ),
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

    @staticmethod
    def _new_recognition_stats() -> dict[str, int]:
        return {
            "frames": 0,
            "persons": 0,
            "faces": 0,
            "matched": 0,
            "embeddings": 0,
            "known": 0,
            "unknown": 0,
            "no_face_track": 0,
            "no_person_match": 0,
            "no_person_track": 0,
            "no_embedding": 0,
            "multiple_embeddings": 0,
            "branch_embeddings": 0,
            "branch_no_embedding": 0,
        }

    @staticmethod
    def _get_track_id(detection) -> int | None:
        if detection is None:
            return None
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if not track:
            return None
        return track[0].get_id()

    def _add_identity_classification(
        self,
        detection,
        label: str,
        confidence: float,
        tracker_name: str | None = None,
        track_id: int | None = None,
    ) -> None:
        classifications = detection.get_objects_typed(hailo.HAILO_CLASSIFICATION)
        for classification in classifications:
            if classification.get_classification_type() == "face_recon":
                detection.remove_object(classification)

        new_classification = hailo.HailoClassification(
            type="face_recon",
            label=label,
            confidence=confidence,
        )
        detection.add_object(new_classification)

        if tracker_name is None or track_id is None:
            return

        self.tracker.remove_classifications_from_track(tracker_name, track_id, "face_recon")
        self.tracker.add_object_to_track(tracker_name, track_id, new_classification)

    def _log_recognition_stats(self, frame_number: int) -> None:
        stats = self.recognition_stats
        if stats["frames"] < 30:
            return

        logger.info(
            "Recognition stats through frame %d: frames=%d persons=%d faces=%d "
            "matched=%d embeddings=%d known=%d unknown=%d no_face_track=%d "
            "no_person_match=%d no_person_track=%d no_embedding=%d multiple_embeddings=%d",
            frame_number,
            stats["frames"],
            stats["persons"],
            stats["faces"],
            stats["matched"],
            stats["embeddings"],
            stats["known"],
            stats["unknown"],
            stats["no_face_track"],
            stats["no_person_match"],
            stats["no_person_track"],
            stats["no_embedding"],
            stats["multiple_embeddings"],
        )
        logger.info(
            "Face embedding callback stats: embeddings=%d no_embedding=%d",
            stats["branch_embeddings"],
            stats["branch_no_embedding"],
        )
        self.recognition_stats = self._new_recognition_stats()

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

    def _mark_person_seen(self, global_id: str, track_id: int) -> dict | None:
        """Increment the persisted visit counter when a known person is seen again."""
        return self.db_handler.mark_person_seen(
            global_id=global_id,
            timestamp=int(time.time()),
            track_id=track_id,
        )

    def _notify_frontend(
        self,
        event: str,
        global_id: str,
        label: str,
        track_id: int,
        confidence: float,
        visit_count: int | None = None,
    ) -> None:
        """Send a best-effort JSON event to an optional frontend/backend endpoint."""
        if not self.notify_url:
            return

        payload = {
            "event": event,
            "global_id": global_id,
            "label": label,
            "track_id": track_id,
            "confidence": confidence,
            "visit_count": visit_count,
            "timestamp": int(time.time()),
        }
        request = urllib.request.Request(
            self.notify_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=1.0):
                pass
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.debug("Failed to notify %s: %s", self.notify_url, exc)

    def shutdown(self, signum=None, frame=None):
        """Prevent double shutdown and noisy HailoRT transfer errors on Ctrl-C."""
        if self._shutdown_started:
            return
        self._shutdown_started = True
        super().shutdown(signum, frame)

    def _connect_face_embedding_callback(self) -> None:
        identity = self.pipeline.get_by_name("face_embedding_callback")
        if identity is None:
            logger.warning("face_embedding_callback not found in pipeline")
            return

        identity_pad = identity.get_static_pad("src")
        identity_pad.add_probe(Gst.PadProbeType.BUFFER, self.face_embedding_callback)

    def _on_pipeline_rebuilt(self) -> None:
        self.face_track_embeddings.clear()
        self._connect_face_embedding_callback()

    def face_embedding_callback(self, pad, info):
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK

        roi = hailo.get_roi_from_buffer(buffer)
        for detection in (
            detection
            for detection in roi.get_objects_typed(hailo.HAILO_DETECTION)
            if detection.get_label() == "face"
        ):
            face_track_id = self._get_track_id(detection)
            if face_track_id is None:
                continue

            embeddings = detection.get_objects_typed(hailo.HAILO_MATRIX)
            if not embeddings:
                self.recognition_stats["branch_no_embedding"] += 1
                continue

            self.face_track_embeddings[face_track_id] = np.array(embeddings[0].get_data())
            self.recognition_stats["branch_embeddings"] += 1
            for embedding in embeddings:
                detection.remove_object(embedding)

        return Gst.PadProbeReturn.OK

    @staticmethod
    def _draw_detection(frame: np.ndarray, detection, width: int, height: int, label: str) -> None:
        bbox = detection.get_bbox()
        x_min = max(0, min(bbox.xmin(), 1))
        y_min = max(0, min(bbox.ymin(), 1))
        x_max = max(0, min(bbox.xmax(), 1))
        y_max = max(0, min(bbox.ymax(), 1))

        x1 = int(x_min * width)
        y1 = int(y_min * height)
        x2 = int(x_max * width)
        y2 = int(y_max * height)

        if x2 <= x1 or y2 <= y1:
            return

        color = (0, 255, 0) if label == "face" else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _bbox_to_pixels(detection, width: int, height: int) -> tuple[int, int, int, int]:
        bbox = detection.get_bbox()
        x1 = int(max(0, min(bbox.xmin(), 1)) * width)
        y1 = int(max(0, min(bbox.ymin(), 1)) * height)
        x2 = int(max(0, min(bbox.xmax(), 1)) * width)
        y2 = int(max(0, min(bbox.ymax(), 1)) * height)
        return x1, y1, x2, y2

    @staticmethod
    def _box_contains_point(box: tuple[int, int, int, int], point: tuple[int, int]) -> bool:
        x1, y1, x2, y2 = box
        px, py = point
        return x1 <= px <= x2 and y1 <= py <= y2

    @staticmethod
    def _bbox_area(box: tuple[int, int, int, int]) -> int:
        x1, y1, x2, y2 = box
        return max(0, x2 - x1) * max(0, y2 - y1)

    @staticmethod
    def _intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        x1 = max(ax1, bx1)
        y1 = max(ay1, by1)
        x2 = min(ax2, bx2)
        y2 = min(ay2, by2)
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _find_person_for_face(self, face_detection, person_detections, width: int, height: int):
        face_x1, face_y1, face_x2, face_y2 = self._bbox_to_pixels(face_detection, width, height)
        face_center = ((face_x1 + face_x2) // 2, (face_y1 + face_y2) // 2)
        face_box = (face_x1, face_y1, face_x2, face_y2)

        best_person = None
        best_score = None
        for person in person_detections:
            if person.get_label() != "person":
                continue
            person_box = self._bbox_to_pixels(person, width, height)
            if not self._box_contains_point(person_box, face_center):
                continue
            overlap_ratio = self._intersection_area(face_box, person_box) / max(1, self._bbox_area(face_box))
            if overlap_ratio < 0.25:
                continue
            area = self._bbox_area(person_box)
            score = (overlap_ratio, -area)
            if best_person is None or score > best_score:
                best_person = person
                best_score = score

        return best_person

    @staticmethod
    def _prune_non_person_detections(roi) -> int:
        removed = 0
        for detection in list(roi.get_objects_typed(hailo.HAILO_DETECTION)):
            label = detection.get_label()
            if label not in {"person", "face"}:
                roi.remove_object(detection)
                removed += 1
        return removed

    def save_image_file(self, frame, image_path):
        image = Image.fromarray(frame)
        image.save(image_path, format="JPEG", quality=85)

    def crop_frame(self, frame, bbox, width, height):
        x_min = max(0, min(bbox.xmin() - 0.15, 1))
        y_min = max(0, min(bbox.ymin() - 0.15, 1))
        x_max = max(0, min(bbox.xmax() + 0.15, 1))
        y_max = max(0, min(bbox.ymax() + 0.15, 1))

        x_min = int(x_min * width)
        y_min = int(y_min * height)
        x_max = int(x_max * width)
        y_max = int(y_max * height)

        return frame[y_min:y_max, x_min:x_max]

    def _save_face_sample(self, frame: np.ndarray, detection, width: int, height: int) -> str:
        image_path = self.samples_dir / f"{uuid.uuid4()}.jpeg"
        cropped = self.crop_frame(frame, detection.get_bbox(), width, height)
        self.save_image_file(cropped, str(image_path))
        return str(image_path)

    def _recognize_embedding(self, embedding_vector: np.ndarray) -> tuple[dict, float]:
        person = self.db_handler.search_record(embedding=embedding_vector)
        confidence = 0.0 if person["label"] == "Unknown" else 1 - person["_distance"]
        return person, confidence

    def _enroll_if_ready(
        self,
        track_id: int,
        person_detection,
    ) -> None:
        pending = self.pending_unknowns.get(track_id)
        if pending is None or len(pending.samples) < self.samples_per_person:
            return

        avg_embedding = np.mean([sample.embedding for sample in pending.samples], axis=0)
        person, confidence = self._recognize_embedding(avg_embedding)
        if person["label"] != "Unknown":
            self.track_to_global_id[track_id] = person["global_id"]
            self.track_to_label[track_id] = person["label"]
            updated_record = self._mark_person_seen(person["global_id"], track_id)
            self._add_identity_classification(
                person_detection,
                person["label"],
                confidence,
                self.person_tracker_name,
                track_id,
            )
            self._print_identity(
                track_id,
                person["global_id"],
                person["label"],
                confidence,
                "recognized-after-samples",
            )
            if updated_record and updated_record.get("visit_incremented"):
                self._notify_frontend(
                    "recognized",
                    person["global_id"],
                    person["label"],
                    track_id,
                    confidence,
                    visit_count=updated_record["visit_count"],
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
            last_seen_track_id=track_id,
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
        self._add_identity_classification(
            person_detection,
            label,
            1.0,
            self.person_tracker_name,
            track_id,
        )
        self._print_identity(track_id, new_person["global_id"], label, 1.0, "enrolled")
        self._notify_frontend(
            "enrolled",
            new_person["global_id"],
            label,
            track_id,
            1.0,
            visit_count=new_person["visit_count"],
        )
        self.pending_unknowns.pop(track_id, None)

    def _handle_unknown_person(
        self,
        track_id: int,
        frame_number: int,
        frame: np.ndarray,
        face_detection,
        person_detection,
        embedding_vector: np.ndarray,
        width: int,
        height: int,
    ) -> None:
        detection_confidence = face_detection.get_confidence()
        if detection_confidence < self.min_enroll_confidence:
            return

        pending = self.pending_unknowns.setdefault(track_id, PendingIdentity())
        if (
            pending.last_sample_frame >= 0
            and frame_number - pending.last_sample_frame < self.unknown_sample_interval
        ):
            return

        image_path = self._save_face_sample(frame, face_detection, width, height)
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
        self._enroll_if_ready(track_id, person_detection)

    def get_pipeline_string(self):
        source_pipeline = self.get_source_pipeline()

        person_detection_pipeline = INFERENCE_PIPELINE(
            hef_path=self.person_hef_path,
            post_process_so=self.person_post_process_so,
            post_function_name=self.person_post_function_name,
            batch_size=self.batch_size,
            config_json=self.person_labels_json,
            additional_params=self.person_thresholds_str,
            name="person_detection",
        )
        person_detection_wrapper = INFERENCE_PIPELINE_WRAPPER(
            person_detection_pipeline,
            name="person_detection_wrapper",
        )
        person_tracker_pipeline = TRACKER_PIPELINE(
            class_id=self.person_class_id,
            kalman_dist_thr=0.7,
            iou_thr=0.8,
            init_iou_thr=0.9,
            keep_new_frames=2,
            keep_tracked_frames=6,
            keep_lost_frames=8,
            keep_past_metadata=True,
            name="person_tracker",
        )

        face_detection_pipeline = INFERENCE_PIPELINE(
            hef_path=self.face_detection_hef_path,
            post_process_so=self.face_post_process_so,
            post_function_name=self.face_detection_func,
            batch_size=self.batch_size,
            config_json=self.face_labels_json,
        )
        face_detection_wrapper = INFERENCE_PIPELINE_WRAPPER(face_detection_pipeline)
        face_tracker_pipeline = TRACKER_PIPELINE(
            class_id=-1,
            kalman_dist_thr=0.7,
            iou_thr=0.8,
            init_iou_thr=0.9,
            keep_new_frames=2,
            keep_tracked_frames=6,
            keep_lost_frames=8,
            keep_past_metadata=True,
            name=self.face_tracker_name,
        )
        face_id_pipeline = INFERENCE_PIPELINE(
            hef_path=self.face_recognition_hef_path,
            post_process_so=self.face_recognition_post_process_so,
            post_function_name=self.face_recognition_post_function_name,
            batch_size=self.batch_size,
            config_json=None,
            name="face_recognition_inference",
        )
        face_cropper_pipeline = CROPPER_PIPELINE(
            inner_pipeline=(
                f'hailofilter so-path={self.face_align_post_process_so} '
                f'name=face_align_hailofilter use-gst-buffer=true qos=false ! '
                f'{QUEUE(name="detector_pos_face_align_q")} ! '
                f'{face_id_pipeline}'
            ),
            so_path=self.face_cropper_post_process_so,
            function_name=self.face_cropper_function_name,
            internal_offset=True,
        )

        display_pipeline = DISPLAY_PIPELINE(
            video_sink=self.video_sink,
            sync=self.sync,
            show_fps=self.show_fps,
        )

        return (
            f"{source_pipeline} ! "
            f"{face_detection_wrapper} ! "
            f"{face_tracker_pipeline} ! "
            f"{face_cropper_pipeline} ! "
            f"{USER_CALLBACK_PIPELINE(name='face_embedding_callback')} ! "
            f"{person_detection_wrapper} ! "
            f"{person_tracker_pipeline} ! "
            f"{USER_CALLBACK_PIPELINE(name='identity_callback')} ! "
            f"{display_pipeline}"
        )

    def pipeline_callback(self, element, buffer, user_data):
        if buffer is None:
            logger.warning("Received None buffer.")
            return

        roi = hailo.get_roi_from_buffer(buffer)
        removed_detections = self._prune_non_person_detections(roi)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
        face_detections = [d for d in detections if d.get_label() == "face"]
        person_detections = [d for d in detections if d.get_label() == "person"]

        for person_detection in person_detections:
            person_track_id = self._get_track_id(person_detection)
            if person_track_id is None:
                continue
            display_id = self.track_to_label.get(person_track_id, "Unknown")
            self._add_identity_classification(
                person_detection,
                display_id,
                1.0,
                self.person_tracker_name,
                person_track_id,
            )

        pad = element.get_static_pad("src")
        fmt, width, height = get_caps_from_pad(pad)
        frame_number = user_data.get_count()
        frame = get_numpy_from_buffer_efficient(buffer, fmt, width, height) if self.debug_face_overlay else None

        stats = self.recognition_stats
        stats["frames"] += 1
        stats["persons"] += len(person_detections)
        stats["faces"] += len(face_detections)
        if removed_detections:
            logger.debug("Removed %d non-person/non-face detections on frame %d", removed_detections, frame_number)

        for detection in face_detections:
            matched_person = self._find_person_for_face(detection, person_detections, width, height)
            face_track_id = self._get_track_id(detection)
            if face_track_id is None:
                stats["no_face_track"] += 1
                continue
            embedding_vector = self.face_track_embeddings.pop(face_track_id, None)

            if matched_person is None:
                stats["no_person_match"] += 1
                continue

            matched_person_track_id = self._get_track_id(matched_person)
            if matched_person_track_id is None:
                stats["no_person_track"] += 1
                continue
            stats["matched"] += 1

            if embedding_vector is None:
                embeddings = detection.get_objects_typed(hailo.HAILO_MATRIX)
                if not embeddings:
                    stats["no_embedding"] += 1
                    continue
                if len(embeddings) > 1:
                    stats["multiple_embeddings"] += 1
                embedding_vector = np.array(embeddings[0].get_data())
                for embedding in embeddings:
                    detection.remove_object(embedding)
            stats["embeddings"] += 1

            known_global_id = self.track_to_global_id.get(matched_person_track_id)
            if known_global_id is not None:
                known_label = self.track_to_label.get(matched_person_track_id, known_global_id)
                self._add_identity_classification(
                    matched_person,
                    known_label,
                    1.0,
                    self.person_tracker_name,
                    matched_person_track_id,
                )
                stats["known"] += 1
                if frame is not None and self.debug_face_overlay:
                    self._draw_detection(frame, detection, width, height, "face")
                    self._draw_detection(frame, matched_person, width, height, known_label)
                    user_data.set_frame(frame)
                self._print_identity(
                    matched_person_track_id,
                    known_global_id,
                    known_label,
                    1.0,
                    "recognized",
                )
                continue

            bbox = detection.get_bbox()
            logger.debug(
                "Face candidate: frame=%d track_id=%d conf=%.2f bbox=(%.3f, %.3f, %.3f, %.3f)",
                frame_number,
                face_track_id,
                detection.get_confidence(),
                bbox.xmin(),
                bbox.ymin(),
                bbox.xmax(),
                bbox.ymax(),
            )

            person, confidence = self._recognize_embedding(embedding_vector)
            if person["label"] != "Unknown":
                self.track_to_global_id[matched_person_track_id] = person["global_id"]
                self.track_to_label[matched_person_track_id] = person["label"]
                updated_record = self._mark_person_seen(
                    person["global_id"],
                    matched_person_track_id,
                )
                self.pending_unknowns.pop(matched_person_track_id, None)
                self._add_identity_classification(
                    matched_person,
                    person["label"],
                    confidence,
                    self.person_tracker_name,
                    matched_person_track_id,
                )
                stats["known"] += 1
                if frame is not None and self.debug_face_overlay:
                    self._draw_detection(frame, detection, width, height, "face")
                    self._draw_detection(frame, matched_person, width, height, person["label"])
                    user_data.set_frame(frame)
                self._print_identity(
                    matched_person_track_id,
                    person["global_id"],
                    person["label"],
                    confidence,
                    "recognized",
                )
                if updated_record and updated_record.get("visit_incremented"):
                    self._notify_frontend(
                        "recognized",
                        person["global_id"],
                        person["label"],
                        matched_person_track_id,
                        confidence,
                        visit_count=updated_record["visit_count"],
                    )
                continue

            stats["unknown"] += 1
            if frame is None:
                frame = get_numpy_from_buffer_efficient(buffer, fmt, width, height)

            if self.debug_face_overlay:
                self._draw_detection(frame, detection, width, height, "face")
                self._draw_detection(frame, matched_person, width, height, "Unknown")
                user_data.set_frame(frame)

            self._handle_unknown_person(
                matched_person_track_id,
                frame_number,
                frame,
                detection,
                matched_person,
                embedding_vector,
                width,
                height,
            )

        self._log_recognition_stats(frame_number)
        return Gst.FlowReturn.OK


def main() -> None:
    logger.info("Starting person-face ID app.")
    user_data = PersonFaceIdData()
    app = PersonFaceIdApp(user_data)
    app.run()


if __name__ == "__main__":
    main()
