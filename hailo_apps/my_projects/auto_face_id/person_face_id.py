#!/usr/bin/env python3
"""Person tracking with face detection/recognition merged from parallel Hailo branches."""

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

from PIL import Image
import cv2

from hailo_apps.python.core.common.buffer_utils import (
    get_caps_from_pad,
    get_numpy_from_buffer_efficient,
)
from hailo_apps.python.core.common.db_handler import DatabaseHandler, Record
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

logger = get_logger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
DATABASE_DIR = PROJECT_DIR / "database"
SAMPLES_DIR = PROJECT_DIR / "samples"
DB_NAME = "persons.db"
TABLE_NAME = "persons"


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
    """Detect persons, then detect and recognize faces inside each person crop."""

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
        self.person_class_id = self.options_menu.person_class_id
        self.debug_face_overlay = self.options_menu.use_frame
        self._shutdown_started = False

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

        self.db_handler = DatabaseHandler(
            db_name=DB_NAME,
            table_name=TABLE_NAME,
            schema=Record,
            threshold=0.55,
            database_dir=str(self.database_dir),
            samples_dir=str(self.samples_dir),
        )

        self.pending_unknowns: dict[int, PendingIdentity] = {}
        self.track_to_global_id: dict[int, str] = {}
        self.person_track_to_global_id: dict[int, str] = {}
        self.track_to_label: dict[int, str] = {}
        self.last_printed_identity: dict[int, str] = {}
        self.next_person_index = self._load_next_person_index()

        self.app_callback = self.pipeline_callback

        logger.info("Person-face database: %s", self.database_dir / DB_NAME)
        logger.info("Person-face samples: %s", self.samples_dir)

        self.create_pipeline()

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
            default=0,
            help="Class ID to track as person. Default is 0.",
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

    def _add_identity_classification(self, detection, label: str, confidence: float) -> None:
        classifications = detection.get_objects_typed(hailo.HAILO_CLASSIFICATION)
        for classification in classifications:
            detection.remove_object(classification)

        detection.add_object(
            hailo.HailoClassification(
                type="face_recon",
                label=label,
                confidence=confidence,
            )
        )

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

    def shutdown(self, signum=None, frame=None):
        """Prevent double shutdown and noisy HailoRT transfer errors on Ctrl-C."""
        if self._shutdown_started:
            return
        self._shutdown_started = True
        super().shutdown(signum, frame)

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

    def _find_person_for_face(self, face_detection, person_detections, width: int, height: int):
        face_x1, face_y1, face_x2, face_y2 = self._bbox_to_pixels(face_detection, width, height)
        face_center = ((face_x1 + face_x2) // 2, (face_y1 + face_y2) // 2)

        best_person = None
        best_area = None
        for person in person_detections:
            if person.get_label() != "person":
                continue
            person_box = self._bbox_to_pixels(person, width, height)
            if not self._box_contains_point(person_box, face_center):
                continue
            area = max(0, person_box[2] - person_box[0]) * max(0, person_box[3] - person_box[1])
            if best_person is None or area < best_area:
                best_person = person
                best_area = area

        return best_person

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
        confidence = 1 - person["_distance"]
        return person, confidence

    def _enroll_if_ready(
        self,
        track_id: int,
        detection,
    ) -> None:
        pending = self.pending_unknowns.get(track_id)
        if pending is None or len(pending.samples) < self.samples_per_person:
            return

        avg_embedding = np.mean([sample.embedding for sample in pending.samples], axis=0)
        person, confidence = self._recognize_embedding(avg_embedding)
        if person["label"] != "Unknown":
            self.track_to_global_id[track_id] = person["global_id"]
            self.track_to_label[track_id] = person["label"]
            self._add_identity_classification(detection, person["label"], confidence)
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
        self._add_identity_classification(detection, label, 1.0)
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
        self._enroll_if_ready(track_id, detection)

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
            name="face_detection",
        )
        face_detection_wrapper = INFERENCE_PIPELINE_WRAPPER(
            face_detection_pipeline,
            name="face_detection_wrapper",
        )
        face_tracker_pipeline = TRACKER_PIPELINE(
            class_id=-1,
            kalman_dist_thr=0.7,
            iou_thr=0.8,
            init_iou_thr=0.9,
            keep_new_frames=2,
            keep_tracked_frames=6,
            keep_lost_frames=8,
            keep_past_metadata=True,
            name="face_tracker",
        )
        face_id_pipeline = INFERENCE_PIPELINE(
            hef_path=self.face_recognition_hef_path,
            post_process_so=self.face_recognition_post_process_so,
            post_function_name=self.face_recognition_post_function_name,
            batch_size=self.batch_size,
            config_json=None,
            name="face_recognition",
        )
        face_cropper_pipeline = CROPPER_PIPELINE(
            inner_pipeline=(
                f'hailofilter so-path={self.face_align_post_process_so} '
                f'name=face_align_hailofilter use-gst-buffer=true qos=false ! '
                f'{QUEUE(name="face_align_q")} ! '
                f'{face_id_pipeline}'
            ),
            so_path=self.face_cropper_post_process_so,
            function_name=self.face_cropper_function_name,
            internal_offset=True,
            name="face_cropper",
        )

        display_pipeline = DISPLAY_PIPELINE(
            video_sink=self.video_sink,
            sync=self.sync,
            show_fps=self.show_fps,
        )

        return (
            f"{source_pipeline} ! tee name=split "
            f"hailomuxer name=mux "
            f"split. ! {QUEUE(name='person_branch_q')} ! "
            f"{person_detection_wrapper} ! "
            f"{person_tracker_pipeline} ! "
            f"mux.sink_0 "
            f"split. ! {QUEUE(name='face_branch_q')} ! "
            f"{face_detection_wrapper} ! "
            f"{face_tracker_pipeline} ! "
            f"{face_cropper_pipeline} ! "
            f"mux.sink_1 "
            f"mux. ! "
            f"{USER_CALLBACK_PIPELINE(name='identity_callback')} ! "
            f"{display_pipeline}"
        )

    def pipeline_callback(self, element, buffer, user_data):
        if buffer is None:
            logger.warning("Received None buffer.")
            return

        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
        face_detections = [d for d in detections if d.get_label() == "face"]
        person_detections = [d for d in detections if d.get_label() == "person"]

        pad = element.get_static_pad("src")
        fmt, width, height = get_caps_from_pad(pad)
        frame_number = user_data.get_count()
        frame = get_numpy_from_buffer_efficient(buffer, fmt, width, height) if self.debug_face_overlay else None

        if face_detections:
            logger.info(
                "Detections on frame %d: persons=%d faces=%d",
                frame_number,
                len(person_detections),
                len(face_detections),
            )
        elif frame_number % 30 == 0:
            logger.info("No face detections on frame %d", frame_number)

        for detection in face_detections:
            matched_person = self._find_person_for_face(detection, person_detections, width, height)
            if matched_person is None:
                continue

            track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if not track:
                continue

            track_id = track[0].get_id()
            embedding = detection.get_objects_typed(hailo.HAILO_MATRIX)
            if len(embedding) != 1:
                continue

            embedding_vector = np.array(embedding[0].get_data())
            detection.remove_object(embedding[0])

            bbox = detection.get_bbox()
            logger.info(
                "Face candidate: frame=%d track_id=%d conf=%.2f bbox=(%.3f, %.3f, %.3f, %.3f)",
                frame_number,
                track_id,
                detection.get_confidence(),
                bbox.xmin(),
                bbox.ymin(),
                bbox.xmax(),
                bbox.ymax(),
            )

            person, confidence = self._recognize_embedding(embedding_vector)
            if person["label"] != "Unknown":
                self.track_to_global_id[track_id] = person["global_id"]
                self.track_to_label[track_id] = person["label"]
                self.pending_unknowns.pop(track_id, None)
                self._add_identity_classification(detection, person["label"], confidence)
                self._add_identity_classification(matched_person, person["label"], confidence)
                matched_person_track = matched_person.get_objects_typed(hailo.HAILO_UNIQUE_ID)
                if matched_person_track:
                    self.person_track_to_global_id[matched_person_track[0].get_id()] = person["global_id"]
                if frame is not None and self.debug_face_overlay:
                    self._draw_detection(frame, detection, width, height, "face")
                self._print_identity(
                    track_id,
                    person["global_id"],
                    person["label"],
                    confidence,
                    "recognized",
                )
                if frame is not None and self.debug_face_overlay:
                    user_data.set_frame(frame)
                continue

            if frame is None:
                frame = get_numpy_from_buffer_efficient(buffer, fmt, width, height)

            if self.debug_face_overlay:
                self._draw_detection(frame, detection, width, height, "face")
                user_data.set_frame(frame)

            self._handle_unknown_face(
                track_id,
                frame_number,
                frame,
                detection,
                embedding_vector,
                width,
                height,
            )

        return Gst.FlowReturn.OK


def main() -> None:
    logger.info("Starting person-face ID app.")
    user_data = PersonFaceIdData()
    app = PersonFaceIdApp(user_data)
    app.run()


if __name__ == "__main__":
    main()
