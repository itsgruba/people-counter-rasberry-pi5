#!/usr/bin/env python3
"""
Person + face identification application.

High-level flow:
1. GStreamer/Hailo detects and tracks faces.
2. Face embeddings are cached by face track ID.
3. GStreamer/Hailo detects and tracks persons.
4. Every face is matched back to the containing person box.
5. The person track is recognized, enrolled as a new person, or left pending.
6. ENTRY adds a person after A -> B; EXIT removes the recognized person after A -> B.

The file is intentionally split into sections below. When adding a second camera,
keep camera-specific behavior behind CameraMode first, then move shared recognition
logic into smaller modules only when it becomes painful to keep here.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from enum import Enum
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
DEFAULT_PIPELINE_LATENCY_MS = 50
DEFAULT_LOW_LATENCY_QUEUE_SIZE = 1

QUEUE_PATTERN = re.compile(
    r"queue name=(?P<name>\S+) "
    r"leaky=(?P<leaky>\S+) "
    r"max-size-buffers=(?P<buffers>\d+) "
    r"max-size-bytes=(?P<bytes>\d+) "
    r"max-size-time=(?P<time>\d+)"
)


# ---------------------------------------------------------------------------
# Camera roles
# ---------------------------------------------------------------------------


class CameraMode(str, Enum):
    """Role of the current camera instance.

    ENTRY enrolls/recognizes people and records entry crossings. EXIT recognizes
    only active entered people and removes them after an exit crossing.
    """

    ENTRY = "entry"
    EXIT = "exit"


# ---------------------------------------------------------------------------
# Small state containers
# ---------------------------------------------------------------------------


@dataclass
class PendingSample:
    """One candidate sample for a newly observed unknown face."""

    embedding: np.ndarray
    image_path: str
    confidence: float
    timestamp: int


@dataclass
class PendingVote:
    """One recognition vote for an unresolved person track."""

    global_id: str | None
    label: str | None
    confidence: float
    timestamp: int


@dataclass
class PendingIdentity:
    """Samples accumulated for a tracker ID before creating a permanent identity."""

    samples: list[PendingSample] = field(default_factory=list)
    recognition_votes: list[PendingVote] = field(default_factory=list)
    last_sample_frame: int = -1
    first_seen_time: float = field(default_factory=time.time)


@dataclass
class EntryEvent:
    """One completed A -> B line crossing for a person track."""

    entry_event_id: str
    track_id: int
    entered_at: int
    frame_number: int
    point: tuple[float, float]


@dataclass
class EntryTrackState:
    """Line-crossing state keyed by the Hailo person tracker ID."""

    last_side_a: int | None = None
    last_side_b: int | None = None
    stage: str = "outside"
    crossed_a_frame: int | None = None
    crossed_a_at: int | None = None
    entry_event_id: str | None = None
    entered_frame: int | None = None
    entered_at: int | None = None
    entered_global_id: str | None = None
    last_seen_at: float = field(default_factory=time.time)
    last_frame: int = 0
    last_point: tuple[float, float] | None = None


@dataclass
class PendingEntryContext:
    """Runtime context for an A -> B entry waiting for identity resolution."""

    event: EntryEvent
    entry_photo_path: str | None
    created_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Entry line crossing
# ---------------------------------------------------------------------------


class EntryDetector:
    """Detect a completed entry when a track crosses line A and then line B."""

    def __init__(
        self,
        line_a_y: float,
        line_b_y: float,
        margin: float,
        track_ttl_seconds: float,
        min_frames_between_lines: int,
    ) -> None:
        self.line_a_y = line_a_y
        self.line_b_y = line_b_y
        self.margin = max(0.0, margin)
        self.track_ttl_seconds = max(0.0, track_ttl_seconds)
        self.min_frames_between_lines = max(0, min_frames_between_lines)
        self.tracks: dict[int, EntryTrackState] = {}

    @staticmethod
    def anchor_point(detection) -> tuple[float, float]:
        bbox = detection.get_bbox()
        return (bbox.xmin() + bbox.xmax()) / 2.0, bbox.ymax()

    def _side(self, value: float, line_y: float) -> int:
        if value < line_y - self.margin:
            return -1
        if value > line_y + self.margin:
            return 1
        return 0

    def _update_side(
        self,
        previous_side: int | None,
        value: float,
        line_y: float,
    ) -> tuple[int | None, bool]:
        current_side = self._side(value, line_y)
        if current_side == 0:
            return previous_side, False
        if previous_side is None:
            return current_side, False
        if previous_side != current_side:
            return current_side, True
        return previous_side, False

    @staticmethod
    def _line_cross_order(
        previous_y: float,
        current_y: float,
        first_line_y: float,
        second_line_y: float,
    ) -> bool:
        delta_y = current_y - previous_y
        if abs(delta_y) <= 1e-12:
            return True
        first_t = (first_line_y - previous_y) / delta_y
        second_t = (second_line_y - previous_y) / delta_y
        return first_t <= second_t

    def update(
        self,
        track_id: int,
        detection,
        frame_number: int,
        timestamp: int,
    ) -> EntryEvent | None:
        # The bottom-center of the person box is used as the crossing point.
        # This is more stable for doorway logic than the bbox center because it
        # follows the person's feet/ground position.
        point = self.anchor_point(detection)
        state = self.tracks.setdefault(track_id, EntryTrackState())
        previous_point = state.last_point
        state.last_seen_at = time.time()
        state.last_frame = frame_number

        # Each line has a small dead zone (margin). The side only changes when
        # the anchor point moves clearly past the line, which prevents jitter
        # around the line from creating duplicate events.
        new_side_a, crossed_a = self._update_side(state.last_side_a, point[1], self.line_a_y)
        new_side_b, crossed_b = self._update_side(state.last_side_b, point[1], self.line_b_y)
        state.last_side_a = new_side_a
        state.last_side_b = new_side_b
        crossed_a_before_b = True
        if crossed_a and crossed_b and previous_point is not None:
            crossed_a_before_b = self._line_cross_order(
                previous_point[1],
                point[1],
                self.line_a_y,
                self.line_b_y,
            )

        if crossed_a and state.stage != "entered":
            state.stage = "crossed_a"
            state.crossed_a_frame = frame_number
            state.crossed_a_at = timestamp

        # A valid entry is only A -> B. If both lines are crossed in one frame,
        # _line_cross_order uses the previous and current anchor positions to
        # keep the order deterministic.
        enough_frames = (
            state.crossed_a_frame is not None
            and frame_number - state.crossed_a_frame >= self.min_frames_between_lines
        )
        if crossed_b and state.stage == "crossed_a" and enough_frames and crossed_a_before_b:
            state.stage = "entered"
            state.entry_event_id = state.entry_event_id or str(uuid.uuid4())
            state.entered_frame = frame_number
            state.entered_at = timestamp
            state.last_point = point
            return EntryEvent(
                entry_event_id=state.entry_event_id,
                track_id=track_id,
                entered_at=timestamp,
                frame_number=frame_number,
                point=point,
            )

        state.last_point = point
        return None

    def mark_entry_counted(self, track_id: int, global_id: str) -> None:
        state = self.tracks.get(track_id)
        if state is not None and state.entered_at is not None:
            state.entered_global_id = global_id

    def has_uncounted_entry(self, track_id: int) -> bool:
        state = self.tracks.get(track_id)
        return (
            state is not None
            and state.entered_at is not None
            and state.entry_event_id is not None
            and state.entered_global_id is None
        )

    def uncounted_entry_event(self, track_id: int) -> EntryEvent | None:
        state = self.tracks.get(track_id)
        if (
            state is None
            or state.entry_event_id is None
            or state.entered_at is None
            or state.entered_frame is None
            or state.entered_global_id is not None
        ):
            return None
        return EntryEvent(
            entry_event_id=state.entry_event_id,
            track_id=track_id,
            entered_at=state.entered_at,
            frame_number=state.entered_frame,
            point=state.last_point or (0.0, 0.0),
        )

    def cleanup(self, active_track_ids: set[int]) -> None:
        if self.track_ttl_seconds <= 0:
            return
        now = time.time()
        expired_track_ids = [
            track_id
            for track_id, state in self.tracks.items()
            if (
                track_id not in active_track_ids
                and now - state.last_seen_at >= self.track_ttl_seconds
            )
        ]
        for track_id in expired_track_ids:
            self.tracks.pop(track_id, None)


# ---------------------------------------------------------------------------
# Data shared between GStreamer callbacks and optional debug HTTP stream
# ---------------------------------------------------------------------------


class PersonFaceIdData(app_callback_class):
    """Small thread-safe state object passed into the GStreamer callback."""

    def __init__(self) -> None:
        super().__init__()
        self.latest_track_id = -1
        self.latest_debug_jpeg: bytes | None = None
        self.latest_debug_frame_number = 0
        self.latest_debug_frame_timestamp = 0.0
        self._debug_frame_condition = threading.Condition()

    def set_debug_frame(self, frame: np.ndarray, frame_number: int, jpeg_quality: int) -> None:
        """Store the latest processed frame as JPEG for the MJPEG debug endpoint."""
        jpeg_quality = max(1, min(jpeg_quality, 100))
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        success, encoded = cv2.imencode(".jpg", bgr_frame, encode_params)
        if not success:
            logger.warning("Failed to encode debug frame %d as JPEG", frame_number)
            return

        with self._debug_frame_condition:
            self.latest_debug_jpeg = encoded.tobytes()
            self.latest_debug_frame_number = frame_number
            self.latest_debug_frame_timestamp = time.time()
            self._debug_frame_condition.notify_all()

    def wait_for_debug_frame(
        self,
        last_frame_number: int | None,
        timeout: float = 1.0,
    ) -> tuple[bytes | None, int]:
        """Wait until a newer debug frame is available, then return its JPEG bytes."""
        with self._debug_frame_condition:
            self._debug_frame_condition.wait_for(
                lambda: (
                    not self.running
                    or (
                        self.latest_debug_jpeg is not None
                        and self.latest_debug_frame_number != last_frame_number
                    )
                ),
                timeout=timeout,
            )
            return self.latest_debug_jpeg, self.latest_debug_frame_number


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------


class PersonFaceIdApp(GStreamerApp):
    """Build the Hailo pipeline and own all recognition/enrollment state."""

    def __init__(self, user_data, parser: argparse.ArgumentParser | None = None):
        parser = parser or self._build_parser()
        super().__init__(parser, user_data)

        # Logging must be initialized after GStreamerApp parses CLI options.
        if hasattr(self.options_menu, "log_level") or hasattr(self.options_menu, "debug"):
            init_logging(
                level=level_from_args(self.options_menu),
                log_file=getattr(self.options_menu, "log_file", None),
            )

        # Camera role. Keep mode decisions here or in small helper methods so a
        # future exit camera does not fork the whole pipeline callback.
        self.camera_mode = CameraMode(self.options_menu.camera_mode)

        # Enrollment and recognition tuning. These values decide when an
        # unknown face becomes a stable existing match or a brand-new person.
        self.samples_per_person = self.options_menu.samples_per_person
        self.unknown_sample_interval = self.options_menu.unknown_sample_interval
        self.min_enroll_confidence = self.options_menu.min_enroll_confidence
        self.min_unknown_age_seconds = self.options_menu.min_unknown_age_seconds
        self.pending_identity_ttl_seconds = self.options_menu.pending_identity_ttl_seconds
        self.recognition_vote_window = max(1, self.options_menu.recognition_vote_window)
        self.recognition_vote_threshold = max(1, self.options_menu.recognition_vote_threshold)
        self.recognition_vote_threshold = min(
            self.recognition_vote_threshold,
            self.recognition_vote_window,
        )

        # Live-configured enrollment zone. Recognition still runs everywhere;
        # this zone only limits where new people are automatically enrolled.
        self.enroll_zone = self._parse_enroll_zone(self.options_menu.enroll_zone)
        self.enroll_zone_file = (
            self._resolve_live_config_path(self.options_menu.enroll_zone_file)
            if self.options_menu.enroll_zone_file
            else None
        )
        self._enroll_zone_file_mtime: float | None = None
        self._reload_enroll_zone_file(force=True)
        self.enroll_zone_anchor = self.options_menu.enroll_zone_anchor
        self.exit_recognition_zone = self._parse_normalized_zone(
            self.options_menu.exit_recognition_zone,
            "--exit-recognition-zone",
        )
        self.exit_recognition_zone_file = (
            self._resolve_live_config_path(self.options_menu.exit_recognition_zone_file)
            if self.options_menu.exit_recognition_zone_file
            else None
        )
        self._exit_recognition_zone_file_mtime: float | None = None
        self._reload_exit_recognition_zone_file(force=True)
        self.exit_recognition_zone_anchor = self.options_menu.exit_recognition_zone_anchor
        self.min_enroll_face_width_ratio = self.options_menu.min_enroll_face_width_ratio
        self.min_enroll_face_height_ratio = self.options_menu.min_enroll_face_height_ratio
        self.max_enroll_edge_margin = self.options_menu.max_enroll_edge_margin
        self.min_enroll_blur_score = self.options_menu.min_enroll_blur_score
        self.max_enroll_nose_offset = self.options_menu.max_enroll_nose_offset
        self.min_enroll_eye_balance = self.options_menu.min_enroll_eye_balance
        self.require_enroll_landmarks = self.options_menu.require_enroll_landmarks

        # External output and debug controls.
        self.print_every_frame = self.options_menu.print_every_frame
        self.notify_url = self.options_menu.notify_url
        self.person_class_id = self.options_menu.person_class_id
        self.debug_face_overlay = self.options_menu.use_frame
        self.debug_stream_enabled = not self.options_menu.disable_debug_stream
        self.debug_stream_host = self.options_menu.debug_stream_host
        self.debug_stream_port = self.options_menu.debug_stream_port
        self.debug_jpeg_quality = self.options_menu.debug_jpeg_quality
        self.debug_show_stats = not self.options_menu.debug_stream_no_stats
        self.low_latency_enabled = not self.options_menu.disable_low_latency
        self.low_latency_queue_size = max(1, self.options_menu.low_latency_queue_size)
        self.pipeline_latency = max(0, self.options_menu.pipeline_latency_ms)

        # Entry counting is intentionally tied to camera_mode. EXIT uses its own
        # recognition zone and crossing detector below.
        self._validate_entry_options()
        self.entry_counter_enabled = (
            self.camera_mode == CameraMode.ENTRY
            and not self.options_menu.disable_entry_counter
        )
        if self.camera_mode == CameraMode.EXIT and not self.options_menu.disable_entry_counter:
            logger.warning(
                "camera_mode=exit runs entered-person recognition only; "
                "entry counting is disabled for this instance."
            )
        self.entry_detector = EntryDetector(
            line_a_y=self.options_menu.entry_line_a_y,
            line_b_y=self.options_menu.entry_line_b_y,
            margin=self.options_menu.entry_line_margin,
            track_ttl_seconds=self.options_menu.entry_track_ttl_seconds,
            min_frames_between_lines=self.options_menu.entry_min_frames_between_lines,
        )
        self.entry_pending_resolution_seconds = max(
            0.0,
            self.options_menu.entry_pending_resolution_seconds,
        )
        self.entry_lines_file = (
            self._resolve_live_config_path(self.options_menu.entry_lines_file)
            if self.options_menu.entry_lines_file
            else self.enroll_zone_file
        )
        self.entry_lines_file_explicit = self.options_menu.entry_lines_file is not None
        self._entry_lines_file_mtime: float | None = None
        self._reload_entry_lines_file(force=True)

        # Exit uses the same robust A -> B crossing mechanics as entry, but has
        # independent lines because the second camera has its own perspective.
        self._validate_exit_options()
        self.exit_counter_enabled = (
            self.camera_mode == CameraMode.EXIT
            and not self.options_menu.disable_exit_counter
        )
        self.exit_detector = EntryDetector(
            line_a_y=self.options_menu.exit_line_a_y,
            line_b_y=self.options_menu.exit_line_b_y,
            margin=self.options_menu.exit_line_margin,
            track_ttl_seconds=self.options_menu.exit_track_ttl_seconds,
            min_frames_between_lines=self.options_menu.exit_min_frames_between_lines,
        )
        self.exit_lines_file = (
            self._resolve_live_config_path(self.options_menu.exit_lines_file)
            if self.options_menu.exit_lines_file
            else self.exit_recognition_zone_file
        )
        self.exit_lines_file_explicit = self.options_menu.exit_lines_file is not None
        self._exit_lines_file_mtime: float | None = None
        self._reload_exit_lines_file(force=True)
        self._shutdown_started = False
        self._debug_fps = 0.0
        self._debug_fps_frames = 0
        self._debug_fps_updated_at = time.time()
        self._debug_server_thread: threading.Thread | None = None

        # Hailo tracker names. These must match the names used in
        # get_pipeline_string(), because classifications are attached back to
        # tracker metadata by name and track_id.
        self.face_tracker_name = "hailo_face_tracker"
        self.person_tracker_name = "person_tracker"
        self.tracker = HailoTracker.get_instance()

        # Persistent storage layout for the SQLite database and saved images.
        DATABASE_DIR.mkdir(parents=True, exist_ok=True)
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        self.database_dir = DATABASE_DIR
        self.samples_dir = SAMPLES_DIR

        # Resolve model files and native post-processing libraries once during
        # startup; the pipeline builder only consumes the resolved paths.
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

        # Detection thresholds are passed as a single string because the shared
        # Hailo helper pipeline expects additional hailofilter params in that form.
        self.person_thresholds_str = (
            "nms-score-threshold=0.3 "
            "nms-iou-threshold=0.45 "
            "output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        )

        # Database handler owns persistent people, samples, visits, and entry events.
        self.db_handler = SQLiteDatabaseHandler(
            db_name=DB_NAME,
            threshold=0.55,
            database_dir=str(self.database_dir),
            samples_dir=str(self.samples_dir),
        )

        # Runtime identity maps are keyed by Hailo person track_id. Track IDs are
        # short-lived camera-session IDs; global_id is the persistent DB identity.
        self.pending_unknowns: dict[int, PendingIdentity] = {}
        self.face_track_embeddings: dict[int, np.ndarray] = {}
        self.track_to_global_id: dict[int, str] = {}
        self.track_to_label: dict[int, str] = {}
        self.last_printed_identity: dict[int, str] = {}
        self.recognition_stats = self._new_recognition_stats()
        self.entered_people: list[dict] = []
        self.entered_people_by_entry_event_id: dict[str, dict] = {}
        self.pending_entry_contexts: dict[int, PendingEntryContext] = {}
        self._load_entered_people()
        self.next_person_index = self._load_next_person_index()

        # GStreamerApp calls this for the identity_callback element.
        self.app_callback = self.pipeline_callback

        logger.info("Person-face database: %s", self.database_dir / DB_NAME)
        logger.info("Person-face samples: %s", self.samples_dir)
        logger.info("Camera mode: %s", self.camera_mode.value)
        logger.info("Person-face database records: %d", len(self.db_handler.get_all_records()))
        logger.info("Loaded entered_people records: %d", len(self.entered_people))
        if self.entry_counter_enabled:
            logger.info(
                "Entry counter enabled: line_a_y=%.3f line_b_y=%.3f margin=%.3f",
                self.entry_detector.line_a_y,
                self.entry_detector.line_b_y,
                self.entry_detector.margin,
            )
        if self.exit_counter_enabled:
            logger.info(
                "Exit counter enabled: line_a_y=%.3f line_b_y=%.3f margin=%.3f",
                self.exit_detector.line_a_y,
                self.exit_detector.line_b_y,
                self.exit_detector.margin,
            )

        if self.options_menu.disable_local_display:
            self.video_sink = "fakesink"

        if self.debug_stream_enabled:
            self._start_debug_stream_server()

        self.create_pipeline()
        self._connect_face_embedding_callback()

    # ------------------------------------------------------------------
    # CLI and live configuration
    # ------------------------------------------------------------------

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
            "--camera-mode",
            choices=tuple(mode.value for mode in CameraMode),
            default=CameraMode.ENTRY.value,
            help=(
                "Role of this camera instance. 'entry' enables A -> B entry counting. "
                "'exit' recognizes entered people and removes them after exit A -> B."
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
            "--min-unknown-age-seconds",
            type=float,
            default=2.0,
            help="Minimum time to observe an unknown track before creating a new person.",
        )
        parser.add_argument(
            "--pending-identity-ttl-seconds",
            type=float,
            default=15.0,
            help=(
                "Maximum age for unresolved pending unknown identities. "
                "Expired pending samples are deleted. Use 0 to disable cleanup."
            ),
        )
        parser.add_argument(
            "--recognition-vote-window",
            type=int,
            default=5,
            help="Number of recent embeddings used for pending recognition voting.",
        )
        parser.add_argument(
            "--recognition-vote-threshold",
            type=int,
            default=3,
            help="Votes for one existing person required before binding an unknown track.",
        )
        parser.add_argument(
            "--min-enroll-face-width-ratio",
            type=float,
            default=0.05,
            help="Minimum face bbox width, as a fraction of frame width, for enrollment samples.",
        )
        parser.add_argument(
            "--min-enroll-face-height-ratio",
            type=float,
            default=0.06,
            help="Minimum face bbox height, as a fraction of frame height, for enrollment samples.",
        )
        parser.add_argument(
            "--enroll-zone",
            default=None,
            help=(
                "Optional normalized enrollment polygon as x1,y1,x2,y2,... . "
                "A four-number value is treated as rectangle xmin,ymin,xmax,ymax. "
                "Recognition still runs everywhere; only new-person enrollment is limited."
            ),
        )
        parser.add_argument(
            "--enroll-zone-file",
            default=None,
            help=(
                "Optional text file containing the same normalized polygon as --enroll-zone. "
                "The file is reloaded while the app is running when it changes."
            ),
        )
        parser.add_argument(
            "--enroll-zone-anchor",
            choices=("person-feet", "person-center", "face-center"),
            default="person-feet",
            help="Point tested against --enroll-zone. Default is bottom-center of the person box.",
        )
        parser.add_argument(
            "--exit-recognition-zone",
            default=None,
            help=(
                "Optional normalized polygon where exit-camera identity recognition is allowed, "
                "as x1,y1,x2,y2,... . Four numbers define xmin,ymin,xmax,ymax. "
                "Outside this zone an unbound exit-camera track remains Unknown."
            ),
        )
        parser.add_argument(
            "--exit-recognition-zone-file",
            default=None,
            help=(
                "Optional text file containing the same normalized polygon as "
                "--exit-recognition-zone. The file is reloaded while the app is running."
            ),
        )
        parser.add_argument(
            "--exit-recognition-zone-anchor",
            choices=("person-feet", "person-center", "face-center"),
            default="person-feet",
            help=(
                "Point tested against --exit-recognition-zone. "
                "Default is bottom-center of the person box."
            ),
        )
        parser.add_argument(
            "--max-enroll-edge-margin",
            type=float,
            default=0.02,
            help="Reject enrollment samples when the face bbox is this close to a frame edge.",
        )
        parser.add_argument(
            "--min-enroll-blur-score",
            type=float,
            default=150.0,
            help=(
                "Minimum Laplacian variance for enrollment crops. "
                "Use 0 to disable blur filtering."
            ),
        )
        parser.add_argument(
            "--max-enroll-nose-offset",
            type=float,
            default=0.35,
            help=(
                "Reject enrollment samples when the nose is too far from the midpoint "
                "between the eyes, normalized by eye distance. Use 0 to disable."
            ),
        )
        parser.add_argument(
            "--min-enroll-eye-balance",
            type=float,
            default=0.45,
            help=(
                "Reject profile-like enrollment samples when nose-to-eye distances are "
                "too uneven. Use 0 to disable."
            ),
        )
        parser.add_argument(
            "--require-enroll-landmarks",
            action="store_true",
            help="Reject enrollment samples that do not include face landmarks.",
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
        parser.add_argument(
            "--disable-entry-counter",
            action="store_true",
            help="Disable A -> B entry counting for tracked people.",
        )
        parser.add_argument(
            "--entry-line-a-y",
            type=float,
            default=0.55,
            help="Normalized Y coordinate for entry line A. Default: 0.55.",
        )
        parser.add_argument(
            "--entry-line-b-y",
            type=float,
            default=0.75,
            help="Normalized Y coordinate for entry line B. Default: 0.75.",
        )
        parser.add_argument(
            "--entry-line-margin",
            type=float,
            default=0.02,
            help="Normalized hysteresis margin around each entry line. Default: 0.02.",
        )
        parser.add_argument(
            "--entry-track-ttl-seconds",
            type=float,
            default=60.0,
            help="Seconds to keep entry state for disappeared tracks. Default: 60.",
        )
        parser.add_argument(
            "--entry-min-frames-between-lines",
            type=int,
            default=0,
            help="Minimum frame gap between crossing line A and line B. Default: 0.",
        )
        parser.add_argument(
            "--entry-pending-resolution-seconds",
            type=float,
            default=2.0,
            help=(
                "Seconds to wait for recognition after A -> B before creating a new "
                "fallback person. Use 0 to create immediately. Default: 2."
            ),
        )
        parser.add_argument(
            "--entry-lines-file",
            default=None,
            help=(
                "Optional text file reloaded while the app is running. Supports "
                "entry_line_a_y=0.55, entry_line_b_y=0.75, entry_line_margin=0.02. "
                "If omitted, the app also looks for those keys in --enroll-zone-file."
            ),
        )
        parser.add_argument(
            "--disable-exit-counter",
            action="store_true",
            help="Disable A -> B exit counting for tracked people.",
        )
        parser.add_argument(
            "--exit-line-a-y",
            type=float,
            default=0.55,
            help="Normalized Y coordinate for exit line A. Default: 0.55.",
        )
        parser.add_argument(
            "--exit-line-b-y",
            type=float,
            default=0.75,
            help="Normalized Y coordinate for exit line B. Default: 0.75.",
        )
        parser.add_argument(
            "--exit-line-margin",
            type=float,
            default=0.02,
            help="Normalized hysteresis margin around each exit line. Default: 0.02.",
        )
        parser.add_argument(
            "--exit-track-ttl-seconds",
            type=float,
            default=60.0,
            help="Seconds to keep exit state for disappeared tracks. Default: 60.",
        )
        parser.add_argument(
            "--exit-min-frames-between-lines",
            type=int,
            default=0,
            help="Minimum frame gap between crossing exit line A and B. Default: 0.",
        )
        parser.add_argument(
            "--exit-lines-file",
            default=None,
            help=(
                "Optional text file reloaded while running. Supports exit_line_a_y, "
                "exit_line_b_y and exit_line_margin. If omitted, the app also looks "
                "for these keys in --exit-recognition-zone-file."
            ),
        )
        parser.add_argument(
            "--disable-debug-stream",
            action="store_true",
            help="Disable the built-in Flask MJPEG debug stream.",
        )
        parser.add_argument(
            "--debug-stream-host",
            default="0.0.0.0",
            help="Host/interface for the Flask MJPEG debug server. Default: 0.0.0.0.",
        )
        parser.add_argument(
            "--debug-stream-port",
            type=int,
            default=8090,
            help="Port for the Flask MJPEG debug server. Default: 8090.",
        )
        parser.add_argument(
            "--debug-jpeg-quality",
            type=int,
            default=80,
            help="JPEG quality for the MJPEG debug stream, from 1 to 100. Default: 80.",
        )
        parser.add_argument(
            "--debug-stream-no-stats",
            action="store_true",
            help="Do not draw frame/FPS/statistics text on the MJPEG debug stream.",
        )
        parser.add_argument(
            "--disable-local-display",
            action="store_true",
            help=(
                "Use a fakesink for the GStreamer display branch. Useful when running "
                "headless and watching the processed stream through /debug."
            ),
        )
        parser.add_argument(
            "--disable-low-latency",
            action="store_true",
            help=(
                "Keep the default Hailo queue buffering behavior. By default this app "
                "uses shallow live-video queues to avoid multi-second camera delay."
            ),
        )
        parser.add_argument(
            "--low-latency-queue-size",
            type=int,
            default=DEFAULT_LOW_LATENCY_QUEUE_SIZE,
            help=(
                "Maximum buffers kept in each GStreamer queue while low-latency mode is enabled. "
                "Default is 1, which favors fresh live frames over smooth backlog playback."
            ),
        )
        parser.add_argument(
            "--pipeline-latency-ms",
            type=int,
            default=DEFAULT_PIPELINE_LATENCY_MS,
            help=(
                "Pipeline latency in milliseconds. Lower values reduce live stream lag; "
                "increase this if the display becomes unstable."
            ),
        )
        return parser

    @staticmethod
    def _resolve_live_config_path(value: str) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute() or path.exists():
            return path

        project_relative_path = PROJECT_DIR / path
        if project_relative_path.exists():
            return project_relative_path

        return path

    def _validate_entry_options(self) -> None:
        self._validate_entry_values(
            self.options_menu.entry_line_a_y,
            self.options_menu.entry_line_b_y,
            self.options_menu.entry_line_margin,
        )

        if self.options_menu.entry_track_ttl_seconds < 0:
            raise ValueError("--entry-track-ttl-seconds must be >= 0.")
        if self.options_menu.entry_min_frames_between_lines < 0:
            raise ValueError("--entry-min-frames-between-lines must be >= 0.")
        if self.options_menu.entry_pending_resolution_seconds < 0:
            raise ValueError("--entry-pending-resolution-seconds must be >= 0.")

    def _validate_exit_options(self) -> None:
        self._validate_crossing_values(
            self.options_menu.exit_line_a_y,
            self.options_menu.exit_line_b_y,
            self.options_menu.exit_line_margin,
            "exit",
        )
        if self.options_menu.exit_track_ttl_seconds < 0:
            raise ValueError("--exit-track-ttl-seconds must be >= 0.")
        if self.options_menu.exit_min_frames_between_lines < 0:
            raise ValueError("--exit-min-frames-between-lines must be >= 0.")

    @staticmethod
    def _validate_entry_values(line_a_y: float, line_b_y: float, margin: float) -> None:
        PersonFaceIdApp._validate_crossing_values(line_a_y, line_b_y, margin, "entry")

    @staticmethod
    def _validate_crossing_values(
        line_a_y: float,
        line_b_y: float,
        margin: float,
        prefix: str,
    ) -> None:
        values = {
            f"{prefix}-line-a-y": line_a_y,
            f"{prefix}-line-b-y": line_b_y,
            f"{prefix}-line-margin": margin,
        }
        for cli_name, value in values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"--{cli_name} must be normalized from 0.0 to 1.0.")

        if line_a_y == line_b_y:
            raise ValueError(
                f"--{prefix}-line-a-y and --{prefix}-line-b-y must be different."
            )

    @staticmethod
    def _parse_normalized_zone(
        value: str | None,
        option_name: str,
    ) -> list[tuple[float, float]] | None:
        if value is None or not value.strip():
            return None

        try:
            coordinates = [float(part.strip()) for part in value.split(",")]
        except ValueError as exc:
            raise ValueError(
                f"{option_name} must contain comma-separated numbers: x1,y1,x2,y2,..."
            ) from exc

        if len(coordinates) == 4:
            x1, y1, x2, y2 = coordinates
            coordinates = [x1, y1, x2, y1, x2, y2, x1, y2]

        if len(coordinates) < 6 or len(coordinates) % 2 != 0:
            raise ValueError(f"{option_name} must define at least three x,y points.")

        points = [
            (coordinates[index], coordinates[index + 1])
            for index in range(0, len(coordinates), 2)
        ]
        for x, y in points:
            if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
                raise ValueError(
                    f"{option_name} coordinates must be normalized from 0.0 to 1.0."
                )
        return points

    @staticmethod
    def _parse_enroll_zone(value: str | None) -> list[tuple[float, float]] | None:
        return PersonFaceIdApp._parse_normalized_zone(value, "--enroll-zone")

    @staticmethod
    def _read_enroll_zone_file(path: Path) -> str | None:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip().lower().replace("-", "_")
                if key in {
                    "entry_line_a_y",
                    "entry_line_b_y",
                    "entry_line_margin",
                    "exit_line_a_y",
                    "exit_line_b_y",
                    "exit_line_margin",
                }:
                    continue
                line = value.strip()
            return line
        return None

    @staticmethod
    def _read_entry_lines_file(path: Path) -> dict[str, float] | None:
        values: dict[str, float] = {}
        aliases = {
            "entry_line_a_y": "entry_line_a_y",
            "line_a_y": "entry_line_a_y",
            "line_a": "entry_line_a_y",
            "a": "entry_line_a_y",
            "entry_line_b_y": "entry_line_b_y",
            "line_b_y": "entry_line_b_y",
            "line_b": "entry_line_b_y",
            "b": "entry_line_b_y",
            "entry_line_margin": "entry_line_margin",
            "line_margin": "entry_line_margin",
            "margin": "entry_line_margin",
        }

        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, raw_value = line.split("=", 1)
                key = key.strip().lower().replace("-", "_")
                target_key = aliases.get(key)
                if target_key is None:
                    continue
                values[target_key] = float(raw_value.strip())
                continue

            try:
                coordinates = [float(part.strip()) for part in line.split(",")]
            except ValueError:
                continue
            if 2 <= len(coordinates) <= 3:
                values["entry_line_a_y"] = coordinates[0]
                values["entry_line_b_y"] = coordinates[1]
                if len(coordinates) == 3:
                    values["entry_line_margin"] = coordinates[2]

        return values or None

    @staticmethod
    def _read_exit_lines_file(path: Path) -> dict[str, float] | None:
        values: dict[str, float] = {}
        aliases = {
            "exit_line_a_y": "exit_line_a_y",
            "line_a_y": "exit_line_a_y",
            "line_a": "exit_line_a_y",
            "a": "exit_line_a_y",
            "exit_line_b_y": "exit_line_b_y",
            "line_b_y": "exit_line_b_y",
            "line_b": "exit_line_b_y",
            "b": "exit_line_b_y",
            "exit_line_margin": "exit_line_margin",
            "line_margin": "exit_line_margin",
            "margin": "exit_line_margin",
        }
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, raw_value = line.split("=", 1)
            target_key = aliases.get(key.strip().lower().replace("-", "_"))
            if target_key is not None:
                values[target_key] = float(raw_value.strip())
        return values or None

    def _reload_enroll_zone_file(self, force: bool = False) -> None:
        if self.enroll_zone_file is None:
            return

        try:
            mtime = self.enroll_zone_file.stat().st_mtime
        except OSError as exc:
            if force:
                logger.warning("Enrollment zone file is not readable: %s (%s)", self.enroll_zone_file, exc)
            return

        if not force and self._enroll_zone_file_mtime == mtime:
            return

        try:
            zone_value = self._read_enroll_zone_file(self.enroll_zone_file)
            self.enroll_zone = self._parse_enroll_zone(zone_value)
            self._enroll_zone_file_mtime = mtime
            logger.info("Reloaded enrollment zone from %s: %s", self.enroll_zone_file, zone_value)
        except (OSError, ValueError) as exc:
            self._enroll_zone_file_mtime = mtime
            logger.warning("Keeping previous enrollment zone; failed to read %s: %s", self.enroll_zone_file, exc)

    def _reload_exit_recognition_zone_file(self, force: bool = False) -> None:
        if self.exit_recognition_zone_file is None:
            return

        try:
            mtime = self.exit_recognition_zone_file.stat().st_mtime
        except OSError as exc:
            if force:
                logger.warning(
                    "Exit recognition zone file is not readable: %s (%s)",
                    self.exit_recognition_zone_file,
                    exc,
                )
            return

        if not force and self._exit_recognition_zone_file_mtime == mtime:
            return

        try:
            zone_value = self._read_enroll_zone_file(self.exit_recognition_zone_file)
            self.exit_recognition_zone = self._parse_normalized_zone(
                zone_value,
                "--exit-recognition-zone",
            )
            self._exit_recognition_zone_file_mtime = mtime
            logger.info(
                "Reloaded exit recognition zone from %s: %s",
                self.exit_recognition_zone_file,
                zone_value,
            )
        except (OSError, ValueError) as exc:
            self._exit_recognition_zone_file_mtime = mtime
            logger.warning(
                "Keeping previous exit recognition zone; failed to read %s: %s",
                self.exit_recognition_zone_file,
                exc,
            )

    def _reload_entry_lines_file(self, force: bool = False) -> None:
        if self.entry_lines_file is None:
            return

        try:
            mtime = self.entry_lines_file.stat().st_mtime
        except OSError as exc:
            if force and self.entry_lines_file_explicit:
                logger.warning("Entry lines file is not readable: %s (%s)", self.entry_lines_file, exc)
            return

        if not force and self._entry_lines_file_mtime == mtime:
            return

        try:
            values = self._read_entry_lines_file(self.entry_lines_file)
            self._entry_lines_file_mtime = mtime
            if not values:
                return

            line_a_y = values.get("entry_line_a_y", self.entry_detector.line_a_y)
            line_b_y = values.get("entry_line_b_y", self.entry_detector.line_b_y)
            margin = values.get("entry_line_margin", self.entry_detector.margin)
            self._validate_entry_values(line_a_y, line_b_y, margin)
            self.entry_detector.line_a_y = line_a_y
            self.entry_detector.line_b_y = line_b_y
            self.entry_detector.margin = margin
            logger.info(
                "Reloaded entry lines from %s: line_a_y=%.3f line_b_y=%.3f margin=%.3f",
                self.entry_lines_file,
                line_a_y,
                line_b_y,
                margin,
            )
        except (OSError, ValueError) as exc:
            self._entry_lines_file_mtime = mtime
            logger.warning(
                "Keeping previous entry lines; failed to read %s: %s",
                self.entry_lines_file,
                exc,
            )

    def _reload_exit_lines_file(self, force: bool = False) -> None:
        if self.exit_lines_file is None:
            return
        try:
            mtime = self.exit_lines_file.stat().st_mtime
        except OSError as exc:
            if force and self.exit_lines_file_explicit:
                logger.warning("Exit lines file is not readable: %s (%s)", self.exit_lines_file, exc)
            return
        if not force and self._exit_lines_file_mtime == mtime:
            return

        try:
            values = self._read_exit_lines_file(self.exit_lines_file)
            self._exit_lines_file_mtime = mtime
            if not values:
                return
            line_a_y = values.get("exit_line_a_y", self.exit_detector.line_a_y)
            line_b_y = values.get("exit_line_b_y", self.exit_detector.line_b_y)
            margin = values.get("exit_line_margin", self.exit_detector.margin)
            self._validate_crossing_values(line_a_y, line_b_y, margin, "exit")
            self.exit_detector.line_a_y = line_a_y
            self.exit_detector.line_b_y = line_b_y
            self.exit_detector.margin = margin
            logger.info(
                "Reloaded exit lines from %s: line_a_y=%.3f line_b_y=%.3f margin=%.3f",
                self.exit_lines_file,
                line_a_y,
                line_b_y,
                margin,
            )
        except (OSError, ValueError) as exc:
            self._exit_lines_file_mtime = mtime
            logger.warning(
                "Keeping previous exit lines; failed to read %s: %s",
                self.exit_lines_file,
                exc,
            )

    # ------------------------------------------------------------------
    # MJPEG debug stream
    # ------------------------------------------------------------------

    def _start_debug_stream_server(self) -> None:
        try:
            from flask import Flask, Response, jsonify, stream_with_context
        except ImportError as exc:
            raise RuntimeError(
                "Flask is required for the MJPEG debug stream. Install it with `pip install flask` "
                "or run with --disable-debug-stream."
            ) from exc

        flask_app = Flask(f"{__name__}.debug_stream")

        @flask_app.get("/")
        def index():
            return (
                "<!doctype html><html><head><title>Person Face ID Debug</title>"
                "<style>html,body{margin:0;background:#111;height:100%;}"
                "img{display:block;width:100%;height:100%;object-fit:contain;}</style>"
                "</head><body><img src=\"/debug\" alt=\"debug stream\"></body></html>"
            )

        @flask_app.get("/debug")
        def debug_stream():
            return Response(
                stream_with_context(self._mjpeg_debug_frames()),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @flask_app.get("/health")
        def health():
            return jsonify(
                {
                    "ok": True,
                    "frame_number": self.user_data.latest_debug_frame_number,
                    "last_frame_timestamp": self.user_data.latest_debug_frame_timestamp,
                }
            )

        self._debug_server_thread = threading.Thread(
            target=flask_app.run,
            kwargs={
                "host": self.debug_stream_host,
                "port": self.debug_stream_port,
                "threaded": True,
                "use_reloader": False,
            },
            daemon=True,
            name="person-face-id-debug-stream",
        )
        self._debug_server_thread.start()
        logger.info(
            "MJPEG debug stream listening on http://%s:%d/debug",
            self.debug_stream_host,
            self.debug_stream_port,
        )

    def _mjpeg_debug_frames(self):
        last_frame_number = None
        while self.user_data.running:
            jpeg, frame_number = self.user_data.wait_for_debug_frame(
                last_frame_number,
                timeout=1.0,
            )
            if jpeg is None or frame_number == last_frame_number:
                continue

            last_frame_number = frame_number
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache\r\n"
                b"Content-Length: " + str(len(jpeg)).encode("ascii") + b"\r\n\r\n"
                + jpeg
                + b"\r\n"
            )

    # ------------------------------------------------------------------
    # Persistent identity and entry-event state
    # ------------------------------------------------------------------

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

    @staticmethod
    def _person_photo_refs(person: dict) -> list[dict]:
        """Return all stored photo references that belong to a person record."""
        photos = []
        for sample in person.get("samples_json") or []:
            sample_path = sample.get("sample_path")
            if not sample_path:
                continue
            photos.append(
                {
                    "kind": "face_sample",
                    "id": sample.get("id"),
                    "path": sample_path,
                    "timestamp": sample.get("timestamp"),
                }
            )

        for visit in person.get("visits_json") or []:
            photo_path = visit.get("photo_path")
            if not photo_path:
                continue
            photos.append(
                {
                    "kind": "visit",
                    "id": visit.get("id"),
                    "path": photo_path,
                    "timestamp": visit.get("timestamp"),
                    "visit_number": visit.get("visit_number"),
                    "track_id": visit.get("track_id"),
                }
            )
        return photos

    def _make_entered_people_item(self, person: dict, entry_event: dict) -> dict:
        """Create one entered_people item with an explicit entry -> person link."""
        return {
            "entry_event_id": entry_event["id"],
            "entered_at": entry_event.get("confirmed_at") or entry_event.get("detected_at"),
            "track_id": entry_event.get("track_id"),
            "person_global_id": person["global_id"],
            "person_label": person["label"],
            "person": person,
            "photos": self._person_photo_refs(person),
            "entry_event": entry_event,
        }

    def _remember_entered_person(
        self,
        person: dict,
        entry_event: dict | None = None,
    ) -> None:
        """Add or refresh one confirmed entered person in memory."""
        entry_event = entry_event or person.get("entry_event")
        if not entry_event or not entry_event.get("id"):
            return
        if entry_event.get("status") != "confirmed":
            return

        item = self._make_entered_people_item(person, entry_event)
        entry_event_id = item["entry_event_id"]
        self.entered_people_by_entry_event_id[entry_event_id] = item

        for index, existing_item in enumerate(self.entered_people):
            if existing_item.get("entry_event_id") == entry_event_id:
                self.entered_people[index] = item
                return
        self.entered_people.append(item)

    def _load_entered_people(self) -> None:
        """Restore confirmed entries as entered_people links from SQLite."""
        self.entered_people.clear()
        self.entered_people_by_entry_event_id.clear()
        for entered_person in self.db_handler.get_entered_people(limit=None):
            self._remember_entered_person(
                entered_person["person"],
                entered_person["entry_event"],
            )

    def _forget_entered_person(self, entry_event_id: str) -> None:
        """Remove one active entry from the exit camera's in-memory view."""
        self.entered_people_by_entry_event_id.pop(entry_event_id, None)
        self.entered_people = [
            item
            for item in self.entered_people
            if item.get("entry_event_id") != entry_event_id
        ]

    def _record_visit_snapshot(
        self,
        person: dict,
        track_id: int,
        frame: np.ndarray | None,
        person_detection,
        width: int,
        height: int,
    ) -> None:
        """Save visual evidence for a visit_count increment."""
        if frame is None:
            logger.warning(
                "Visit count changed for %s but no frame was available for a snapshot.",
                person.get("label", person.get("global_id")),
            )
            return

        visit_number = int(person["visit_count"])
        timestamp = int(person.get("last_seen_at") or time.time())
        image_path = self._save_visit_snapshot(
            label=person["label"],
            visit_number=visit_number,
            timestamp=timestamp,
            track_id=track_id,
            frame=frame,
            detection=person_detection,
            width=width,
            height=height,
        )
        self.db_handler.add_visit_record(
            global_id=person["global_id"],
            visit_number=visit_number,
            timestamp=timestamp,
            photo_path=image_path,
            track_id=track_id,
        )

    def _notify_frontend(
        self,
        event: str,
        global_id: str | None,
        label: str,
        track_id: int,
        confidence: float,
        visit_count: int | None = None,
        entered: int | None = None,
        entry_event_id: str | None = None,
        total_entered: int | None = None,
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
            "entered": entered,
            "entry_event_id": entry_event_id,
            "total_entered": total_entered,
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

    def _create_person_from_pending_samples(
        self,
        track_id: int,
        pending: PendingIdentity,
        label: str,
        timestamp: int,
    ) -> dict:
        """Create a new searchable person from whatever face samples are available."""
        avg_embedding = np.mean([sample.embedding for sample in pending.samples], axis=0)
        for sample in pending.samples:
            self._move_sample_to_person_dir(sample, label)

        best_sample = max(pending.samples, key=lambda sample: sample.confidence)
        new_person = self.db_handler.create_record(
            embedding=avg_embedding,
            sample=best_sample.image_path,
            timestamp=timestamp,
            label=label,
            last_seen_track_id=track_id,
        )

        for sample in pending.samples:
            if sample.image_path == best_sample.image_path:
                continue
            current_record = self.db_handler.get_record_by_id(new_person["global_id"])
            if current_record is None:
                continue
            self.db_handler.insert_new_sample(
                record=current_record,
                embedding=sample.embedding,
                sample=sample.image_path,
                timestamp=sample.timestamp,
            )
        return self.db_handler.get_record_by_id(new_person["global_id"]) or new_person

    def _create_placeholder_person(
        self,
        track_id: int,
        label: str,
        timestamp: int,
        entry_photo_path: str | None,
    ) -> dict:
        """Create a non-searchable person row for an entry without a face embedding yet."""
        new_person = self.db_handler.create_placeholder_record(
            timestamp=timestamp,
            label=label,
            last_seen_track_id=track_id,
        )
        if entry_photo_path:
            self.db_handler.add_visit_record(
                global_id=new_person["global_id"],
                visit_number=int(new_person["visit_count"]),
                timestamp=timestamp,
                photo_path=entry_photo_path,
                track_id=track_id,
            )
            refreshed_person = self.db_handler.get_record_by_id(new_person["global_id"])
            if refreshed_person is not None:
                new_person = refreshed_person
        return new_person

    def _save_known_person_sample_if_empty(
        self,
        global_id: str,
        label: str,
        track_id: int,
        frame: np.ndarray,
        face_detection,
        embedding_vector: np.ndarray,
        width: int,
        height: int,
    ) -> None:
        """Fill a placeholder person with its first usable face sample."""
        person = self.db_handler.get_record_by_id(global_id)
        if person is None or person.get("samples_json"):
            return
        if not self._is_good_enrollment_sample(frame, face_detection, width, height):
            return

        sample_dir = self._person_sample_dir(label)
        sample_dir.mkdir(parents=True, exist_ok=True)
        image_path = sample_dir / f"{uuid.uuid4()}.jpeg"
        cropped = self.crop_frame(frame, face_detection.get_bbox(), width, height)
        if cropped.size == 0:
            return
        self.save_image_file(cropped, str(image_path))
        self.db_handler.insert_new_sample(
            record=person,
            embedding=embedding_vector,
            sample=str(image_path),
            timestamp=int(time.time()),
        )
        logger.info(
            "Added first face sample for placeholder person label=%s track_id=%d",
            label,
            track_id,
        )

    def _resolve_pending_entry_as_new_person(
        self,
        context: PendingEntryContext,
        reason: str,
        person_detection=None,
        frame: np.ndarray | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        """Guarantee that a completed A -> B event lands in entered_person."""
        event = context.event
        if not self.entry_detector.has_uncounted_entry(event.track_id):
            self.pending_entry_contexts.pop(event.track_id, None)
            return

        label = self._make_person_label()
        timestamp = int(time.time())
        pending = self.pending_unknowns.get(event.track_id)
        created_from_samples = pending is not None and bool(pending.samples)
        if created_from_samples:
            new_person = self._create_person_from_pending_samples(
                event.track_id,
                pending,
                label,
                timestamp,
            )
            self.pending_unknowns.pop(event.track_id, None)
        else:
            new_person = self._create_placeholder_person(
                event.track_id,
                label,
                timestamp,
                context.entry_photo_path,
            )

        if created_from_samples and context.entry_photo_path:
            self.db_handler.add_visit_record(
                global_id=new_person["global_id"],
                visit_number=int(new_person["visit_count"]),
                timestamp=timestamp,
                photo_path=context.entry_photo_path,
                track_id=event.track_id,
            )
            refreshed_person = self.db_handler.get_record_by_id(new_person["global_id"])
            if refreshed_person is not None:
                new_person = refreshed_person

        self.track_to_global_id[event.track_id] = new_person["global_id"]
        self.track_to_label[event.track_id] = label
        if person_detection is not None:
            self._add_identity_classification(
                person_detection,
                label,
                1.0,
                self.person_tracker_name,
                event.track_id,
            )

        print(
            f"entry-created-person: entry_event_id={event.entry_event_id} "
            f"track_id={event.track_id} global_id={new_person['global_id']} "
            f"label={label} reason={reason}"
        )
        self._mark_known_person_entered(
            global_id=new_person["global_id"],
            label=label,
            track_id=event.track_id,
            confidence=1.0,
            entry_event=event,
            person_detection=person_detection,
            frame=frame,
            width=width,
            height=height,
        )

    def _resolve_stale_pending_entries(
        self,
        active_person_detections: dict[int, object],
        frame: np.ndarray | None,
        width: int | None,
        height: int | None,
    ) -> None:
        if not self.pending_entry_contexts:
            return

        now = time.time()
        for track_id, context in list(self.pending_entry_contexts.items()):
            if not self.entry_detector.has_uncounted_entry(track_id):
                self.pending_entry_contexts.pop(track_id, None)
                continue

            global_id = self.track_to_global_id.get(track_id)
            if global_id is not None:
                label = self.track_to_label.get(track_id, global_id)
                self._mark_known_person_entered(
                    global_id=global_id,
                    label=label,
                    track_id=track_id,
                    confidence=1.0,
                    entry_event=context.event,
                    person_detection=active_person_detections.get(track_id),
                    frame=frame,
                    width=width,
                    height=height,
                )
                continue

            age = now - context.created_at
            if age < self.entry_pending_resolution_seconds:
                continue

            self._resolve_pending_entry_as_new_person(
                context,
                reason="entry-pending-timeout",
                person_detection=active_person_detections.get(track_id),
                frame=frame,
                width=width,
                height=height,
            )

    def _handle_entry_event(
        self,
        event: EntryEvent,
        person_detection,
        frame: np.ndarray | None,
        width: int | None,
        height: int | None,
    ) -> None:
        entry_photo_path = self._save_entry_snapshot(
            entry_event_id=event.entry_event_id,
            kind="entry",
            timestamp=event.entered_at,
            track_id=event.track_id,
            frame=frame,
            detection=person_detection,
            width=width,
            height=height,
        )
        pending_event = self.db_handler.register_entry_pending(
            entry_event_id=event.entry_event_id,
            timestamp=event.entered_at,
            track_id=event.track_id,
            frame_number=event.frame_number,
            point=event.point,
            entry_photo_path=entry_photo_path,
        )
        label = self.track_to_label.get(event.track_id, "Unknown")
        global_id = self.track_to_global_id.get(event.track_id)
        if global_id is None:
            self.pending_entry_contexts[event.track_id] = PendingEntryContext(
                event=event,
                entry_photo_path=entry_photo_path,
            )
            print(
                f"entered-pending: entry_event_id={event.entry_event_id} "
                f"track_id={event.track_id} label=Unknown"
            )
            self._notify_frontend(
                "entered_pending",
                None,
                "Unknown",
                event.track_id,
                0.0,
                entry_event_id=event.entry_event_id,
                total_entered=pending_event.get("total_entered"),
            )
            if self.entry_pending_resolution_seconds == 0:
                self._resolve_pending_entry_as_new_person(
                    self.pending_entry_contexts[event.track_id],
                    reason="entry-pending-immediate",
                    person_detection=person_detection,
                    frame=frame,
                    width=width,
                    height=height,
                )
            return

        self._mark_known_person_entered(
            global_id=global_id,
            label=label,
            track_id=event.track_id,
            confidence=1.0,
            entry_event=event,
            person_detection=person_detection,
            frame=frame,
            width=width,
            height=height,
        )

    def _mark_known_person_entered(
        self,
        global_id: str,
        label: str,
        track_id: int,
        confidence: float,
        entry_event: EntryEvent,
        person_detection=None,
        frame: np.ndarray | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        timestamp = int(time.time())
        confirmed_photo_path = self._save_entry_snapshot(
            entry_event_id=entry_event.entry_event_id,
            kind="confirmed",
            timestamp=timestamp,
            track_id=track_id,
            frame=frame,
            detection=person_detection,
            width=width,
            height=height,
        )
        record = self.db_handler.mark_person_entered(
            global_id=global_id,
            timestamp=timestamp,
            track_id=track_id,
            entry_event_id=entry_event.entry_event_id,
            detected_at=entry_event.entered_at,
            frame_number=entry_event.frame_number,
            point=entry_event.point,
            confirmed_photo_path=confirmed_photo_path,
        )
        if record is None:
            return

        self.entry_detector.mark_entry_counted(track_id, global_id)
        self.pending_entry_contexts.pop(track_id, None)
        self._remember_entered_person(record)
        entered = int(record.get("entered", 0))
        total_entered = int(record.get("total_entered", 0))
        print(
            f"entered: entry_event_id={entry_event.entry_event_id} "
            f"track_id={track_id} global_id={global_id} "
            f"label={label} entered={entered} total_entered={total_entered}"
        )
        self._notify_frontend(
            "entered",
            global_id,
            label,
            track_id,
            confidence,
            entered=entered,
            entry_event_id=entry_event.entry_event_id,
            total_entered=total_entered,
        )

    def _apply_pending_entry_for_track(
        self,
        track_id: int,
        global_id: str,
        label: str,
        confidence: float,
        frame: np.ndarray | None = None,
        person_detection=None,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        entry_event = self.entry_detector.uncounted_entry_event(track_id)
        if entry_event is None:
            return
        self._mark_known_person_entered(
            global_id=global_id,
            label=label,
            track_id=track_id,
            confidence=confidence,
            entry_event=entry_event,
            person_detection=person_detection,
            frame=frame,
            width=width,
            height=height,
        )

    def _mark_known_person_exited(
        self,
        global_id: str,
        label: str,
        track_id: int,
        confidence: float,
        exit_event: EntryEvent,
    ) -> None:
        record = self.db_handler.mark_person_exited(
            global_id=global_id,
            timestamp=exit_event.entered_at,
            track_id=track_id,
        )
        # Mark this physical crossing as handled even if another process already
        # removed the active row, so one track cannot repeatedly emit an exit.
        self.exit_detector.mark_entry_counted(track_id, global_id)
        if record is None:
            logger.warning(
                "Exit crossed but no active entered_person row exists: "
                "track_id=%d global_id=%s",
                track_id,
                global_id,
            )
            return

        exited_entry_event_id = record["exited_entry_event_id"]
        self._forget_entered_person(exited_entry_event_id)
        total_entered = int(record["total_entered"])
        print(
            f"exited: exit_event_id={exit_event.entry_event_id} "
            f"entry_event_id={exited_entry_event_id} track_id={track_id} "
            f"global_id={global_id} label={label} total_entered={total_entered}"
        )
        self._notify_frontend(
            "exited",
            global_id,
            label,
            track_id,
            confidence,
            entry_event_id=exited_entry_event_id,
            total_entered=total_entered,
        )

    def _handle_exit_event(self, event: EntryEvent) -> None:
        global_id = self.track_to_global_id.get(event.track_id)
        if global_id is None:
            logger.info(
                "Exit A -> B is waiting for identity: exit_event_id=%s track_id=%d",
                event.entry_event_id,
                event.track_id,
            )
            return
        self._mark_known_person_exited(
            global_id=global_id,
            label=self.track_to_label.get(event.track_id, global_id),
            track_id=event.track_id,
            confidence=1.0,
            exit_event=event,
        )

    def _apply_pending_exit_for_track(
        self,
        track_id: int,
        global_id: str,
        label: str,
        confidence: float,
    ) -> None:
        exit_event = self.exit_detector.uncounted_entry_event(track_id)
        if exit_event is None:
            return
        self._mark_known_person_exited(
            global_id=global_id,
            label=label,
            track_id=track_id,
            confidence=confidence,
            exit_event=exit_event,
        )

    # ------------------------------------------------------------------
    # Pipeline hooks
    # ------------------------------------------------------------------

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

        # Face recognition embeddings are produced in the face branch before the
        # person branch finishes. Cache them by face track_id so the main
        # identity callback can match each face back to the containing person.
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

    # ------------------------------------------------------------------
    # Debug drawing
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_detection(
        frame: np.ndarray,
        detection,
        width: int,
        height: int,
        label: str,
        color: tuple[int, int, int] | None = None,
    ) -> None:
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

        if color is None:
            color = (0, 255, 0) if label.startswith("face") else (255, 0, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_width, text_height = text_size
        text_x = x1
        text_y = max(text_height + baseline + 4, y1 - 6)
        cv2.rectangle(
            frame,
            (text_x, text_y - text_height - baseline - 4),
            (min(width - 1, text_x + text_width + 6), text_y + baseline),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (text_x + 3, text_y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_zone(
        frame: np.ndarray,
        width: int,
        height: int,
        zone: list[tuple[float, float]] | None,
        label_text: str,
        color: tuple[int, int, int],
    ) -> None:
        if not zone:
            return

        points = np.array(
            [
                (int(x * width), int(y * height))
                for x, y in zone
            ],
            dtype=np.int32,
        )
        cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)
        for index, (x_norm, y_norm) in enumerate(zone, start=1):
            x = int(x_norm * width)
            y = int(y_norm * height)
            label = f"{index}:{x_norm:.2f},{y_norm:.2f}"
            cv2.circle(frame, (x, y), 5, color, -1)
            cv2.putText(
                frame,
                label,
                (min(width - 120, max(8, x + 8)), min(height - 8, max(18, y - 8))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )
        label_x = int(min(x for x, _ in zone) * width)
        label_y = int(min(y for _, y in zone) * height)
        cv2.putText(
            frame,
            label_text,
            (max(8, label_x), max(24, label_y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    def _draw_active_recognition_zone(
        self,
        frame: np.ndarray,
        width: int,
        height: int,
    ) -> None:
        if self.camera_mode == CameraMode.EXIT:
            self._draw_zone(
                frame,
                width,
                height,
                self.exit_recognition_zone,
                "exit recognition zone",
                (190, 80, 255),
            )
            return
        self._draw_zone(
            frame,
            width,
            height,
            self.enroll_zone,
            "enroll zone",
            (255, 190, 40),
        )

    def _draw_enroll_anchor(
        self,
        frame: np.ndarray,
        person_detection,
        width: int,
        height: int,
    ) -> None:
        if self.enroll_zone_anchor == "face-center":
            return

        point = self._enroll_zone_anchor_point(person_detection, None)
        x = int(max(0.0, min(point[0], 1.0)) * width)
        y = int(max(0.0, min(point[1], 1.0)) * height)
        inside = self._point_in_polygon(point, self.enroll_zone) if self.enroll_zone else True
        color = (40, 220, 90) if inside else (80, 80, 255)

        cv2.circle(frame, (x, y), 6, color, -1)
        cv2.circle(frame, (x, y), 10, color, 2)
        cv2.line(frame, (x - 12, y), (x + 12, y), color, 1)
        cv2.line(frame, (x, y - 12), (x, y + 12), color, 1)

    def _draw_exit_recognition_anchor(
        self,
        frame: np.ndarray,
        person_detection,
        width: int,
        height: int,
    ) -> None:
        if (
            self.camera_mode != CameraMode.EXIT
            or self.exit_recognition_zone_anchor == "face-center"
        ):
            return

        point = self._zone_anchor_point(
            self.exit_recognition_zone_anchor,
            person_detection,
            None,
        )
        x = int(max(0.0, min(point[0], 1.0)) * width)
        y = int(max(0.0, min(point[1], 1.0)) * height)
        inside = (
            self._point_in_polygon(point, self.exit_recognition_zone)
            if self.exit_recognition_zone
            else True
        )
        color = (40, 220, 90) if inside else (80, 80, 255)
        cv2.circle(frame, (x, y), 6, color, -1)
        cv2.circle(frame, (x, y), 10, color, 2)

    def _draw_entry_lines(self, frame: np.ndarray, width: int, height: int) -> None:
        if not self.entry_counter_enabled:
            return

        lines = [
            ("A", self.entry_detector.line_a_y, (70, 180, 255)),
            ("B", self.entry_detector.line_b_y, (255, 210, 70)),
        ]
        for name, y_norm, color in lines:
            y = int(max(0.0, min(y_norm, 1.0)) * height)
            cv2.line(frame, (0, y), (width - 1, y), color, 2)
            cv2.putText(
                frame,
                f"LINE {name}",
                (12, max(18, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    def _draw_exit_lines(self, frame: np.ndarray, width: int, height: int) -> None:
        if not self.exit_counter_enabled:
            return
        lines = [
            ("EXIT A", self.exit_detector.line_a_y, (190, 80, 255)),
            ("EXIT B", self.exit_detector.line_b_y, (80, 220, 255)),
        ]
        for name, y_norm, color in lines:
            y = int(max(0.0, min(y_norm, 1.0)) * height)
            cv2.line(frame, (0, y), (width - 1, y), color, 2)
            cv2.putText(
                frame,
                name,
                (12, max(18, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    def _draw_entry_anchor(
        self,
        frame: np.ndarray,
        person_detection,
        track_id: int | None,
        width: int,
        height: int,
    ) -> None:
        if not self.entry_counter_enabled or track_id is None:
            return

        point = self.entry_detector.anchor_point(person_detection)
        x = int(max(0.0, min(point[0], 1.0)) * width)
        y = int(max(0.0, min(point[1], 1.0)) * height)
        state = self.entry_detector.tracks.get(track_id)
        stage = state.stage if state is not None else "outside"
        color = (60, 220, 120) if stage == "entered" else (70, 180, 255)
        cv2.circle(frame, (x, y), 5, color, -1)
        cv2.putText(
            frame,
            stage,
            (min(width - 120, max(8, x + 8)), min(height - 8, max(18, y - 8))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    def _draw_exit_anchor(
        self,
        frame: np.ndarray,
        person_detection,
        track_id: int | None,
        width: int,
        height: int,
    ) -> None:
        if not self.exit_counter_enabled or track_id is None:
            return

        point = self.exit_detector.anchor_point(person_detection)
        x = int(max(0.0, min(point[0], 1.0)) * width)
        y = int(max(0.0, min(point[1], 1.0)) * height)
        state = self.exit_detector.tracks.get(track_id)
        stage = state.stage if state is not None else "outside"
        display_stage = "exited" if stage == "entered" else stage
        color = (60, 220, 120) if display_stage == "exited" else (190, 80, 255)
        cv2.circle(frame, (x, y), 5, color, -1)
        cv2.putText(
            frame,
            display_stage,
            (min(width - 120, max(8, x + 8)), min(height - 8, max(18, y - 8))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    def _update_debug_fps(self) -> None:
        self._debug_fps_frames += 1
        now = time.time()
        elapsed = now - self._debug_fps_updated_at
        if elapsed < 1.0:
            return
        self._debug_fps = self._debug_fps_frames / elapsed
        self._debug_fps_frames = 0
        self._debug_fps_updated_at = now

    def _draw_debug_overlay(
        self,
        frame: np.ndarray,
        person_detections,
        face_detections,
        width: int,
        height: int,
        frame_number: int,
    ) -> None:
        self._draw_active_recognition_zone(frame, width, height)
        self._draw_entry_lines(frame, width, height)
        self._draw_exit_lines(frame, width, height)

        for person_detection in person_detections:
            track_id = self._get_track_id(person_detection)
            if track_id is None:
                label = "person track_id=-"
            else:
                person_id = self.track_to_label.get(track_id, "Unknown")
                label = f"person_id={person_id} track_id={track_id}"
            self._draw_detection(
                frame,
                person_detection,
                width,
                height,
                label,
                color=(255, 70, 70),
            )
            if self.camera_mode == CameraMode.ENTRY:
                self._draw_enroll_anchor(frame, person_detection, width, height)
            else:
                self._draw_exit_recognition_anchor(frame, person_detection, width, height)
            self._draw_entry_anchor(frame, person_detection, track_id, width, height)
            self._draw_exit_anchor(frame, person_detection, track_id, width, height)

        for face_detection in face_detections:
            track_id = self._get_track_id(face_detection)
            if track_id is None:
                label = "face track_id=-"
            else:
                label = f"face track_id={track_id}"
            self._draw_detection(
                frame,
                face_detection,
                width,
                height,
                label,
                color=(40, 220, 90),
            )

        if not self.debug_show_stats:
            return

        self._update_debug_fps()
        stats_text = (
            f"frame={frame_number} fps={self._debug_fps:.1f} "
            f"persons={len(person_detections)} faces={len(face_detections)}"
        )
        cv2.rectangle(frame, (8, 8), (min(width - 1, 520), 36), (0, 0, 0), -1)
        cv2.putText(
            frame,
            stats_text,
            (16, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

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
    def _point_in_polygon(
        point: tuple[float, float],
        polygon: list[tuple[float, float]],
    ) -> bool:
        x, y = point
        inside = False
        previous_x, previous_y = polygon[-1]
        for current_x, current_y in polygon:
            crosses_y = (current_y > y) != (previous_y > y)
            if crosses_y:
                denominator = previous_y - current_y
                if abs(denominator) < 1e-12:
                    previous_x, previous_y = current_x, current_y
                    continue
                slope_x = (
                    (previous_x - current_x)
                    * (y - current_y)
                    / denominator
                    + current_x
                )
                if x < slope_x:
                    inside = not inside
            previous_x, previous_y = current_x, current_y
        return inside

    @staticmethod
    def _zone_anchor_point(
        anchor: str,
        person_detection,
        face_detection,
    ) -> tuple[float, float]:
        if anchor == "face-center":
            bbox = face_detection.get_bbox()
        else:
            bbox = person_detection.get_bbox()

        x_center = (bbox.xmin() + bbox.xmax()) / 2.0
        if anchor == "person-feet":
            return x_center, bbox.ymax()
        return x_center, (bbox.ymin() + bbox.ymax()) / 2.0

    def _enroll_zone_anchor_point(self, person_detection, face_detection) -> tuple[float, float]:
        return self._zone_anchor_point(
            self.enroll_zone_anchor,
            person_detection,
            face_detection,
        )

    def _is_inside_enroll_zone(self, person_detection, face_detection) -> bool:
        if not self.enroll_zone:
            return True
        point = self._enroll_zone_anchor_point(person_detection, face_detection)
        return self._point_in_polygon(point, self.enroll_zone)

    def _is_inside_exit_recognition_zone(self, person_detection, face_detection) -> bool:
        if not self.exit_recognition_zone:
            return True
        point = self._zone_anchor_point(
            self.exit_recognition_zone_anchor,
            person_detection,
            face_detection,
        )
        return self._point_in_polygon(point, self.exit_recognition_zone)

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

    # ------------------------------------------------------------------
    # Image and sample storage
    # ------------------------------------------------------------------

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

    def _pending_sample_dir(self, track_id: int) -> Path:
        return self.samples_dir / "_pending" / f"track_{track_id}"

    def _person_sample_dir(self, label: str) -> Path:
        return self.samples_dir / label

    def _visit_sample_dir(self, label: str, visit_number: int) -> Path:
        return self._person_sample_dir(label) / "visit_count" / f"visit_{visit_number}"

    def _entry_event_sample_dir(self, entry_event_id: str) -> Path:
        return self.samples_dir / "_entries" / entry_event_id

    def _cleanup_empty_sample_dirs(self, start_dir: Path) -> None:
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

    def _move_sample_to_person_dir(self, sample: PendingSample, label: str) -> None:
        source = Path(sample.image_path)
        target_dir = self._person_sample_dir(label)
        target_dir.mkdir(parents=True, exist_ok=True)

        if source.parent == target_dir:
            return

        target = target_dir / source.name
        if target.exists():
            target = target_dir / f"{source.stem}_{uuid.uuid4().hex}{source.suffix}"

        try:
            source.replace(target)
        except OSError as exc:
            logger.warning("Failed to move sample %s to %s: %s", source, target_dir, exc)
            return

        sample.image_path = str(target)
        self._cleanup_empty_sample_dirs(source.parent)

    def _save_face_sample(
        self,
        track_id: int,
        frame: np.ndarray,
        detection,
        width: int,
        height: int,
    ) -> str:
        sample_dir = self._pending_sample_dir(track_id)
        sample_dir.mkdir(parents=True, exist_ok=True)
        image_path = sample_dir / f"{uuid.uuid4()}.jpeg"
        cropped = self.crop_frame(frame, detection.get_bbox(), width, height)
        self.save_image_file(cropped, str(image_path))
        return str(image_path)

    def _save_visit_snapshot(
        self,
        label: str,
        visit_number: int,
        timestamp: int,
        track_id: int,
        frame: np.ndarray,
        detection,
        width: int,
        height: int,
    ) -> str:
        visit_dir = self._visit_sample_dir(label, visit_number)
        visit_dir.mkdir(parents=True, exist_ok=True)
        image_path = visit_dir / f"snapshot_{timestamp}_track_{track_id}.jpeg"
        cropped = self.crop_frame(frame, detection.get_bbox(), width, height)
        if cropped.size == 0:
            cropped = frame
        self.save_image_file(cropped, str(image_path))
        return str(image_path)

    def _save_entry_snapshot(
        self,
        entry_event_id: str,
        kind: str,
        timestamp: int,
        track_id: int,
        frame: np.ndarray | None,
        detection,
        width: int | None,
        height: int | None,
    ) -> str | None:
        if frame is None or detection is None or width is None or height is None:
            return None

        entry_dir = self._entry_event_sample_dir(entry_event_id)
        entry_dir.mkdir(parents=True, exist_ok=True)
        image_path = entry_dir / f"{kind}_{timestamp}_track_{track_id}.jpeg"
        cropped = self.crop_frame(frame, detection.get_bbox(), width, height)
        if cropped.size == 0:
            cropped = frame
        self.save_image_file(cropped, str(image_path))
        return str(image_path)

    # ------------------------------------------------------------------
    # Recognition, voting, and automatic enrollment
    # ------------------------------------------------------------------

    def _recognize_embedding(self, embedding_vector: np.ndarray) -> tuple[dict, float]:
        person = self.db_handler.search_record_deep(embedding=embedding_vector)
        confidence = 0.0 if person["label"] == "Unknown" else 1 - person["_distance"]
        return person, confidence

    def _recognize_entered_embedding(self, embedding_vector: np.ndarray) -> tuple[dict, float]:
        person = self.db_handler.search_entered_record_best(embedding=embedding_vector)
        confidence = 0.0 if person["label"] == "Unknown" else 1 - person["_distance"]
        return person, confidence

    def _record_pending_vote(
        self,
        track_id: int,
        person: dict,
        confidence: float,
    ) -> PendingVote | None:
        pending = self.pending_unknowns.setdefault(track_id, PendingIdentity())
        if person["label"] == "Unknown":
            vote = PendingVote(
                global_id=None,
                label=None,
                confidence=0.0,
                timestamp=int(time.time()),
            )
        else:
            vote = PendingVote(
                global_id=person["global_id"],
                label=person["label"],
                confidence=confidence,
                timestamp=int(time.time()),
            )

        pending.recognition_votes.append(vote)
        if len(pending.recognition_votes) > self.recognition_vote_window:
            pending.recognition_votes = pending.recognition_votes[-self.recognition_vote_window :]
        return self._stable_pending_vote(pending)

    def _stable_pending_vote(self, pending: PendingIdentity) -> PendingVote | None:
        votes = pending.recognition_votes[-self.recognition_vote_window :]
        if len(votes) < self.recognition_vote_threshold:
            return None

        counts: dict[str, int] = {}
        best_vote_by_id: dict[str, PendingVote] = {}
        for vote in votes:
            if vote.global_id is None:
                continue
            counts[vote.global_id] = counts.get(vote.global_id, 0) + 1
            current_best = best_vote_by_id.get(vote.global_id)
            if current_best is None or vote.confidence > current_best.confidence:
                best_vote_by_id[vote.global_id] = vote

        if not counts:
            return None

        global_id, count = max(counts.items(), key=lambda item: item[1])
        if count < self.recognition_vote_threshold:
            return None
        return best_vote_by_id[global_id]

    def _delete_pending_sample_files(self, pending: PendingIdentity) -> int:
        deleted = 0
        for sample in pending.samples:
            sample_path = Path(sample.image_path)
            try:
                sample_path.unlink(missing_ok=True)
                self._cleanup_empty_sample_dirs(sample_path.parent)
                deleted += 1
            except OSError as exc:
                logger.debug("Failed to delete pending sample %s: %s", sample.image_path, exc)
        return deleted

    def _discard_pending_identity(self, track_id: int, reason: str) -> None:
        pending = self.pending_unknowns.pop(track_id, None)
        if pending is None:
            return

        sample_count = len(pending.samples)
        deleted_count = self._delete_pending_sample_files(pending)
        logger.info(
            "Removed pending identity track_id=%d reason=%s age=%.1fs samples=%d deleted=%d",
            track_id,
            reason,
            time.time() - pending.first_seen_time,
            sample_count,
            deleted_count,
        )

    def _cleanup_expired_pending_unknowns(self) -> None:
        if self.pending_identity_ttl_seconds <= 0:
            return

        now = time.time()
        expired_track_ids = [
            track_id
            for track_id, pending in self.pending_unknowns.items()
            if now - pending.first_seen_time >= self.pending_identity_ttl_seconds
        ]
        for track_id in expired_track_ids:
            self._discard_pending_identity(track_id, "expired")

    def _bind_existing_person(
        self,
        track_id: int,
        person: dict,
        confidence: float,
        state: str,
        person_detection,
        frame: np.ndarray | None,
        width: int,
        height: int,
    ) -> None:
        pending = self.pending_unknowns.get(track_id)
        if pending is not None:
            self._consume_pending_samples_for_existing_person(pending, person)
        self.track_to_global_id[track_id] = person["global_id"]
        self.track_to_label[track_id] = person["label"]
        updated_record = self._mark_person_seen(person["global_id"], track_id)
        self.pending_unknowns.pop(track_id, None)
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
            state,
        )
        self._apply_pending_entry_for_track(
            track_id,
            person["global_id"],
            person["label"],
            confidence,
            frame=frame,
            person_detection=person_detection,
            width=width,
            height=height,
        )
        if updated_record and updated_record.get("visit_incremented"):
            self._record_visit_snapshot(
                updated_record,
                track_id,
                frame,
                person_detection,
                width,
                height,
            )
            self._notify_frontend(
                "recognized",
                person["global_id"],
                person["label"],
                track_id,
                confidence,
                visit_count=updated_record["visit_count"],
            )

    def _bind_entered_person_for_exit_camera(
        self,
        track_id: int,
        person: dict,
        confidence: float,
        person_detection,
    ) -> None:
        """Bind an exit-camera track to the closest already-entered person."""
        self.track_to_global_id[track_id] = person["global_id"]
        self.track_to_label[track_id] = person["label"]
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
            "recognized-entered",
        )
        self._apply_pending_exit_for_track(
            track_id,
            person["global_id"],
            person["label"],
            confidence,
        )

    def _bind_existing_person_from_vote(
        self,
        track_id: int,
        vote: PendingVote,
        state: str,
        person_detection,
        frame: np.ndarray | None,
        width: int,
        height: int,
    ) -> bool:
        if vote.global_id is None:
            return False
        person = self.db_handler.get_record_by_id(vote.global_id)
        if person is None:
            return False
        self._bind_existing_person(
            track_id,
            person,
            vote.confidence,
            state,
            person_detection,
            frame,
            width,
            height,
        )
        return True

    def _consume_pending_samples_for_existing_person(
        self,
        pending: PendingIdentity,
        person: dict,
    ) -> None:
        for sample in pending.samples:
            matched_person, _ = self._recognize_embedding(sample.embedding)
            if matched_person["global_id"] == person["global_id"]:
                self._move_sample_to_person_dir(sample, person["label"])
                current_record = self.db_handler.get_record_by_id(person["global_id"])
                if current_record is not None:
                    self.db_handler.insert_new_sample(
                        record=current_record,
                        embedding=sample.embedding,
                        sample=sample.image_path,
                        timestamp=sample.timestamp,
                    )
                continue

            sample_path = Path(sample.image_path)
            sample_path.unlink(missing_ok=True)
            self._cleanup_empty_sample_dirs(sample_path.parent)

    def _is_good_enrollment_sample(
        self,
        frame: np.ndarray,
        face_detection,
        width: int,
        height: int,
    ) -> bool:
        bbox = face_detection.get_bbox()
        bbox_width = bbox.xmax() - bbox.xmin()
        bbox_height = bbox.ymax() - bbox.ymin()
        if bbox_width < self.min_enroll_face_width_ratio:
            return False
        if bbox_height < self.min_enroll_face_height_ratio:
            return False

        margin = self.max_enroll_edge_margin
        if (
            bbox.xmin() <= margin
            or bbox.ymin() <= margin
            or bbox.xmax() >= 1.0 - margin
            or bbox.ymax() >= 1.0 - margin
        ):
            return False

        if not self._is_front_facing_enrollment_sample(face_detection):
            return False

        if self.min_enroll_blur_score <= 0:
            return True

        x1, y1, x2, y2 = self._bbox_to_pixels(face_detection, width, height)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return blur_score >= self.min_enroll_blur_score

    @staticmethod
    def _face_landmark_points(face_detection) -> list[tuple[float, float]] | None:
        landmarks = face_detection.get_objects_typed(hailo.HAILO_LANDMARKS)
        if not landmarks:
            return None

        points = landmarks[0].get_points()
        if len(points) < 5:
            return None

        return [(float(point.x()), float(point.y())) for point in points[:5]]

    def _is_front_facing_enrollment_sample(self, face_detection) -> bool:
        points = self._face_landmark_points(face_detection)
        if points is None:
            return not self.require_enroll_landmarks

        left_eye, right_eye, nose, left_mouth, right_mouth = points
        eye_distance = abs(right_eye[0] - left_eye[0])
        if eye_distance <= 1e-6:
            return False

        if self.max_enroll_nose_offset > 0:
            eye_mid_x = (left_eye[0] + right_eye[0]) / 2.0
            nose_offset = abs(nose[0] - eye_mid_x) / eye_distance
            if nose_offset > self.max_enroll_nose_offset:
                return False

            mouth_mid_x = (left_mouth[0] + right_mouth[0]) / 2.0
            mouth_offset = abs(mouth_mid_x - nose[0]) / eye_distance
            if mouth_offset > self.max_enroll_nose_offset * 1.25:
                return False

        if self.min_enroll_eye_balance > 0:
            left_nose_distance = abs(nose[0] - left_eye[0])
            right_nose_distance = abs(right_eye[0] - nose[0])
            larger_distance = max(left_nose_distance, right_nose_distance)
            if larger_distance <= 1e-6:
                return False
            eye_balance = min(left_nose_distance, right_nose_distance) / larger_distance
            if eye_balance < self.min_enroll_eye_balance:
                return False

        eye_tilt = abs(left_eye[1] - right_eye[1]) / eye_distance
        if eye_tilt > 0.35:
            return False

        return True

    def _enroll_if_ready(
        self,
        track_id: int,
        person_detection,
        frame: np.ndarray,
        width: int,
        height: int,
    ) -> None:
        pending = self.pending_unknowns.get(track_id)
        if pending is None or len(pending.samples) < self.samples_per_person:
            return
        if time.time() - pending.first_seen_time < self.min_unknown_age_seconds:
            return

        stable_vote = self._stable_pending_vote(pending)
        if stable_vote is not None and self._bind_existing_person_from_vote(
            track_id,
            stable_vote,
            "recognized-by-votes",
            person_detection,
            frame,
            width,
            height,
        ):
            return

        for sample in pending.samples:
            person, confidence = self._recognize_embedding(sample.embedding)
            stable_vote = self._record_pending_vote(track_id, person, confidence)
            if stable_vote is not None and self._bind_existing_person_from_vote(
                track_id,
                stable_vote,
                "recognized-by-samples",
                person_detection,
                frame,
                width,
                height,
            ):
                return

        avg_embedding = np.mean([sample.embedding for sample in pending.samples], axis=0)
        person, confidence = self._recognize_embedding(avg_embedding)
        if person["label"] != "Unknown":
            self._bind_existing_person(
                track_id,
                person,
                confidence,
                "recognized-after-samples",
                person_detection,
                frame,
                width,
                height,
            )
            return

        label = self._make_person_label()
        new_person = self._create_person_from_pending_samples(
            track_id=track_id,
            pending=pending,
            label=label,
            timestamp=int(time.time()),
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
        self._apply_pending_entry_for_track(
            track_id,
            new_person["global_id"],
            label,
            1.0,
            frame=frame,
            person_detection=person_detection,
            width=width,
            height=height,
        )
        self._record_visit_snapshot(
            new_person,
            track_id,
            frame,
            person_detection,
            width,
            height,
        )
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
        self._enroll_if_ready(track_id, person_detection, frame, width, height)
        if track_id not in self.pending_unknowns:
            return

        if not self._is_inside_enroll_zone(person_detection, face_detection):
            return

        detection_confidence = face_detection.get_confidence()
        if detection_confidence < self.min_enroll_confidence:
            return
        if not self._is_good_enrollment_sample(frame, face_detection, width, height):
            return

        pending = self.pending_unknowns.setdefault(track_id, PendingIdentity())
        if (
            pending.last_sample_frame >= 0
            and frame_number - pending.last_sample_frame < self.unknown_sample_interval
        ):
            return

        image_path = self._save_face_sample(track_id, frame, face_detection, width, height)
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
        self._enroll_if_ready(track_id, person_detection, frame, width, height)

    # ------------------------------------------------------------------
    # GStreamer pipeline construction and frame processing
    # ------------------------------------------------------------------

    @staticmethod
    def _is_pairing_sensitive_queue(queue_name: str) -> bool:
        """Queues inside cropper/aggregator branches should not drop one side of a frame pair."""
        if queue_name.endswith("_bypass_q"):
            return True
        if queue_name == "detector_pos_face_align_q":
            return True
        if queue_name.startswith("face_recognition_inference_"):
            return True
        if (
            queue_name.startswith("person_detection_")
            and not queue_name.startswith("person_detection_wrapper_")
        ):
            return True
        if (
            queue_name.startswith("inference_")
            and not queue_name.startswith("inference_wrapper_")
        ):
            return True
        return False

    def _low_latency_queue_config(self, match: re.Match) -> str:
        queue_name = match.group("name")
        leaky = "no" if self._is_pairing_sensitive_queue(queue_name) else "downstream"
        return (
            f"queue name={queue_name} "
            f"leaky={leaky} "
            f"max-size-buffers={self.low_latency_queue_size} "
            "max-size-bytes=0 "
            "max-size-time=0"
        )

    def _apply_low_latency_queue_policy(self, pipeline_string: str) -> str:
        if not self.low_latency_enabled:
            return pipeline_string
        return QUEUE_PATTERN.sub(self._low_latency_queue_config, pipeline_string)

    def _update_entry_counter(
        self,
        person_detections,
        frame_number: int,
        frame: np.ndarray | None,
        width: int | None,
        height: int | None,
    ) -> None:
        if not self.entry_counter_enabled:
            return

        active_track_ids: set[int] = set()
        active_person_detections: dict[int, object] = {}
        timestamp = int(time.time())
        for person_detection in person_detections:
            track_id = self._get_track_id(person_detection)
            if track_id is None:
                continue
            active_track_ids.add(track_id)
            active_person_detections[track_id] = person_detection
            event = self.entry_detector.update(
                track_id=track_id,
                detection=person_detection,
                frame_number=frame_number,
                timestamp=timestamp,
            )
            if event is not None:
                self._handle_entry_event(event, person_detection, frame, width, height)

        self._resolve_stale_pending_entries(active_person_detections, frame, width, height)
        self.entry_detector.cleanup(active_track_ids)

    def _update_exit_counter(self, person_detections, frame_number: int) -> None:
        if not self.exit_counter_enabled:
            return

        active_track_ids: set[int] = set()
        timestamp = int(time.time())
        for person_detection in person_detections:
            track_id = self._get_track_id(person_detection)
            if track_id is None:
                continue
            active_track_ids.add(track_id)
            event = self.exit_detector.update(
                track_id=track_id,
                detection=person_detection,
                frame_number=frame_number,
                timestamp=timestamp,
            )
            if event is not None:
                self._handle_exit_event(event)

        self.exit_detector.cleanup(active_track_ids)

    def get_pipeline_string(self):
        source_kwargs = {}
        if self.frame_rate is not None:
            # The shared helper only emits framerate caps when its sync flag is true.
            # For live HTTP cameras we still want --frame-rate to throttle input.
            source_kwargs["sync"] = True
        source_pipeline = self.get_source_pipeline(**source_kwargs)

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

        # Pipeline order matters:
        # face detection -> face tracking -> face embedding callback
        # person detection -> person tracking -> identity callback.
        # The identity callback needs both the cached face embedding and the
        # final person track metadata to bind a face to a person.
        pipeline_string = (
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
        return self._apply_low_latency_queue_policy(pipeline_string)

    def pipeline_callback(self, element, buffer, user_data):
        if buffer is None:
            logger.warning("Received None buffer.")
            return

        # Live config files can be edited while the app is running. Reloading at
        # callback time keeps the camera process alive while tuning zones/lines.
        self._reload_enroll_zone_file()
        self._reload_exit_recognition_zone_file()
        self._reload_entry_lines_file()
        self._reload_exit_lines_file()

        # Keep only person and face detections in the ROI. Other detections from
        # shared post-processors would confuse the face-to-person matching below.
        roi = hailo.get_roi_from_buffer(buffer)
        removed_detections = self._prune_non_person_detections(roi)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
        face_detections = [d for d in detections if d.get_label() == "face"]
        person_detections = [d for d in detections if d.get_label() == "person"]

        # Always attach the best currently known label to every person track so
        # the display/debug branches can show stable IDs even before a new
        # embedding arrives.
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
        needs_frame = (
            self.debug_face_overlay
            or self.debug_stream_enabled
            or self.entry_counter_enabled
        )
        frame = (
            get_numpy_from_buffer_efficient(buffer, fmt, width, height)
            if needs_frame
            else None
        )
        self._update_entry_counter(person_detections, frame_number, frame, width, height)
        self._update_exit_counter(person_detections, frame_number)
        needs_debug_frame = self.debug_face_overlay or self.debug_stream_enabled

        # Recognition flow per face:
        # 1. Find the person box that contains the face.
        # 2. Get the embedding from the face branch cache or from this ROI.
        # 3. If the person track is already known, reuse the existing identity.
        # 4. Otherwise vote across recent embeddings before enrolling a new person.
        stats = self.recognition_stats
        stats["frames"] += 1
        stats["persons"] += len(person_detections)
        stats["faces"] += len(face_detections)
        if removed_detections:
            logger.debug("Removed %d non-person/non-face detections on frame %d", removed_detections, frame_number)
        self._cleanup_expired_pending_unknowns()

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
                self._print_identity(
                    matched_person_track_id,
                    known_global_id,
                    known_label,
                    1.0,
                    "recognized",
                )
                if self.camera_mode == CameraMode.ENTRY and frame is None:
                    frame = get_numpy_from_buffer_efficient(buffer, fmt, width, height)
                if self.camera_mode == CameraMode.ENTRY and frame is not None:
                    self._save_known_person_sample_if_empty(
                        known_global_id,
                        known_label,
                        matched_person_track_id,
                        frame,
                        detection,
                        embedding_vector,
                        width,
                        height,
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

            if self.camera_mode == CameraMode.EXIT:
                if not self._is_inside_exit_recognition_zone(matched_person, detection):
                    logger.debug(
                        "Exit track_id=%d is outside the recognition zone",
                        matched_person_track_id,
                    )
                    continue
                person, confidence = self._recognize_entered_embedding(embedding_vector)
                if person["label"] == "Unknown":
                    stats["unknown"] += 1
                    logger.debug(
                        "Exit camera has no entered-person candidates for track_id=%d",
                        matched_person_track_id,
                    )
                    continue

                self._bind_entered_person_for_exit_camera(
                    matched_person_track_id,
                    person,
                    confidence,
                    matched_person,
                )
                stats["known"] += 1
                continue

            person, confidence = self._recognize_embedding(embedding_vector)
            stable_vote = self._record_pending_vote(
                matched_person_track_id,
                person,
                confidence,
            )
            if stable_vote is not None and frame is None:
                frame = get_numpy_from_buffer_efficient(buffer, fmt, width, height)
            if stable_vote is not None and self._bind_existing_person_from_vote(
                matched_person_track_id,
                stable_vote,
                "recognized-by-votes",
                matched_person,
                frame,
                width,
                height,
            ):
                stats["known"] += 1
                continue

            if person["label"] != "Unknown":
                logger.debug(
                    "Pending recognition vote: track_id=%d global_id=%s label=%s "
                    "confidence=%.2f",
                    matched_person_track_id,
                    person["global_id"],
                    person["label"],
                    confidence,
                )
                continue

            stats["unknown"] += 1
            if frame is None:
                frame = get_numpy_from_buffer_efficient(buffer, fmt, width, height)

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

        if frame is not None and needs_debug_frame:
            debug_frame = frame.copy()
            self._draw_debug_overlay(
                debug_frame,
                person_detections,
                face_detections,
                width,
                height,
                frame_number,
            )
            if self.debug_face_overlay:
                user_data.set_frame(debug_frame)
            if self.debug_stream_enabled:
                user_data.set_debug_frame(debug_frame, frame_number, self.debug_jpeg_quality)

        self._log_recognition_stats(frame_number)
        return Gst.FlowReturn.OK


def main() -> None:
    logger.info("Starting person-face ID app.")
    user_data = PersonFaceIdData()
    app = PersonFaceIdApp(user_data)
    app.run()


if __name__ == "__main__":
    main()
