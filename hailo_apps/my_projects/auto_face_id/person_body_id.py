#!/usr/bin/env python3
"""Whole-body person ReID for ENTRY/EXIT cameras on Hailo-10H.

ENTRY always creates a fresh admission identity. EXIT compares a body embedding
only with identities whose latest visit is currently inside. Face detection and
face recognition are deliberately not part of this pipeline.
"""

from __future__ import annotations

import argparse

import cv2
import hailo
import numpy as np
from gi.repository import Gst

from hailo_apps.python.core.common.buffer_utils import (
    get_caps_from_pad,
    get_numpy_from_buffer_efficient,
)
from hailo_apps.python.core.common.core import get_resource_path, resolve_hef_path
from hailo_apps.python.core.common.defines import (
    ALL_DETECTIONS_CROPPER_POSTPROCESS_SO_FILENAME,
    REID_CROPPER_POSTPROCESS_FUNCTION,
    REID_POSTPROCESS_FUNCTION,
    REID_POSTPROCESS_SO_FILENAME,
    RESOURCES_SO_DIR_NAME,
)
from hailo_apps.python.core.gstreamer.gstreamer_helper_pipelines import (
    CROPPER_PIPELINE,
    DISPLAY_PIPELINE,
    INFERENCE_PIPELINE,
    INFERENCE_PIPELINE_WRAPPER,
    TRACKER_PIPELINE,
    USER_CALLBACK_PIPELINE,
)
from hailo_apps.my_projects.auto_face_id.person_face_id import (
    CameraMode,
    PendingIdentity,
    PersonFaceIdApp,
    PersonFaceIdData,
    logger,
)


BODY_REID_PIPELINE = "person_body_id"
BODY_REID_MODEL = "repvgg_a0_person_reid_512"


class PersonBodyIdApp(PersonFaceIdApp):
    """Reuse visit/counting storage while replacing face inference with body ReID."""

    def __init__(self, user_data, parser: argparse.ArgumentParser | None = None):
        super().__init__(user_data, parser or self._build_body_parser())
        self.min_enroll_body_width_ratio = self.options_menu.min_enroll_body_width_ratio
        self.min_enroll_body_height_ratio = self.options_menu.min_enroll_body_height_ratio

        logger.info("Identity input: whole person (RepVGG body ReID), no face model")
        logger.info("Person-body database: %s", self.database_dir / self.database_name)
        logger.info("Person-body samples: %s", self.samples_dir)

    @staticmethod
    def _build_body_parser() -> argparse.ArgumentParser:
        parser = PersonFaceIdApp._build_parser()
        parser.description = (
            "Track whole people and identify ENTRY/EXIT admissions with body ReID embeddings."
        )
        parser.add_argument(
            "--person-reid-hef-path",
            default=None,
            help=(
                "Path or model name for the 512D person ReID HEF. "
                f"Default for Hailo-10H: {BODY_REID_MODEL}."
            ),
        )
        parser.add_argument(
            "--min-enroll-body-width-ratio",
            type=float,
            default=0.08,
            help="Minimum person bbox width relative to the full frame for saving a body photo.",
        )
        parser.add_argument(
            "--min-enroll-body-height-ratio",
            type=float,
            default=0.22,
            help="Minimum person bbox height relative to the full frame for saving a body photo.",
        )
        parser.set_defaults(
            database_name="persons_body.sqlite3",
            samples_directory="body_samples",
            samples_per_person=5,
            max_pending_embeddings=10,
            unknown_sample_interval=1,
            min_unknown_age_seconds=0.5,
            min_enroll_confidence=0.30,
            min_enroll_blur_score=40.0,
            max_enroll_edge_margin=0.0,
            identity_track_max_gap_frames=25,
        )
        return parser

    def _resolve_identity_pipeline_resources(self) -> None:
        self.body_reid_hef_path = resolve_hef_path(
            self.options_menu.person_reid_hef_path,
            app_name=BODY_REID_PIPELINE,
            arch=self.arch,
            app_type="pipeline",
        )
        if self.body_reid_hef_path is None:
            raise RuntimeError(
                "Failed to resolve the person ReID HEF. Pass "
                "--person-reid-hef-path /path/to/repvgg_a0_person_reid_512.hef."
            )
        self.body_reid_post_process_so = get_resource_path(
            pipeline_name=None,
            resource_type=RESOURCES_SO_DIR_NAME,
            arch=self.arch,
            model=REID_POSTPROCESS_SO_FILENAME,
        )
        self.body_cropper_post_process_so = get_resource_path(
            pipeline_name=None,
            resource_type=RESOURCES_SO_DIR_NAME,
            arch=self.arch,
            model=ALL_DETECTIONS_CROPPER_POSTPROCESS_SO_FILENAME,
        )

    def _connect_face_embedding_callback(self) -> None:
        """Prune the detector output before the every-frame body cropper sees it."""
        callback = self.pipeline.get_by_name("body_person_filter_callback")
        if callback is None:
            logger.warning("body_person_filter_callback not found in pipeline")
            return
        callback.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER,
            self._body_person_filter_callback,
        )

    def _body_person_filter_callback(self, pad, info):
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK
        roi = hailo.get_roi_from_buffer(buffer)
        for detection in list(roi.get_objects_typed(hailo.HAILO_DETECTION)):
            if detection.get_label() != "person":
                roi.remove_object(detection)
        return Gst.PadProbeReturn.OK

    def _is_good_enrollment_sample(
        self,
        frame: np.ndarray,
        person_detection,
        width: int,
        height: int,
    ) -> bool:
        bbox = person_detection.get_bbox()
        if bbox.xmax() - bbox.xmin() < self.min_enroll_body_width_ratio:
            return False
        if bbox.ymax() - bbox.ymin() < self.min_enroll_body_height_ratio:
            return False

        margin = self.max_enroll_edge_margin
        if margin > 0 and (
            bbox.xmin() <= margin
            or bbox.ymin() <= margin
            or bbox.xmax() >= 1.0 - margin
            or bbox.ymax() >= 1.0 - margin
        ):
            return False
        if self.min_enroll_blur_score <= 0:
            return True

        x1, y1, x2, y2 = self._bbox_to_pixels(person_detection, width, height)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var() >= self.min_enroll_blur_score

    def get_pipeline_string(self):
        source_kwargs = {}
        if self.frame_rate is not None:
            source_kwargs["sync"] = True
        source_pipeline = self.get_source_pipeline(**source_kwargs)
        if self.source_type == "rtsp":
            source_pipeline = source_pipeline.replace(
                "rtspsrc ",
                f"rtspsrc protocols=tcp latency={max(0, self.options_menu.rtsp_latency_ms)} "
                "drop-on-latency=true ",
                1,
            )

        detection = INFERENCE_PIPELINE(
            hef_path=self.person_hef_path,
            post_process_so=self.person_post_process_so,
            post_function_name=self.person_post_function_name,
            batch_size=self.batch_size,
            config_json=self.person_labels_json,
            additional_params=self.person_thresholds_str,
            name="person_detection",
        )
        detection_wrapper = INFERENCE_PIPELINE_WRAPPER(
            detection,
            name="person_detection_wrapper",
        )
        tracker = TRACKER_PIPELINE(
            class_id=self.person_class_id,
            kalman_dist_thr=0.7,
            iou_thr=0.8,
            init_iou_thr=0.9,
            keep_new_frames=2,
            keep_tracked_frames=6,
            keep_lost_frames=8,
            keep_past_metadata=True,
            name=self.person_tracker_name,
        )
        reid = INFERENCE_PIPELINE(
            hef_path=self.body_reid_hef_path,
            post_process_so=self.body_reid_post_process_so,
            post_function_name=REID_POSTPROCESS_FUNCTION,
            batch_size=self.batch_size,
            config_json=None,
            name="body_reid_inference",
        )
        cropper = CROPPER_PIPELINE(
            inner_pipeline=reid,
            so_path=self.body_cropper_post_process_so,
            function_name=REID_CROPPER_POSTPROCESS_FUNCTION,
            use_letterbox=False,
            internal_offset=True,
            name="body_reid_cropper_wrapper",
        )
        display = DISPLAY_PIPELINE(
            video_sink=self.video_sink,
            sync=self.sync,
            show_fps=self.show_fps,
        )
        pipeline = (
            f"{source_pipeline} ! "
            f"{detection_wrapper} ! "
            f"{tracker} ! "
            f"{USER_CALLBACK_PIPELINE(name='body_person_filter_callback')} ! "
            f"{cropper} ! "
            f"{USER_CALLBACK_PIPELINE(name='identity_callback')} ! "
            f"{display}"
        )
        return self._apply_low_latency_queue_policy(pipeline)

    def pipeline_callback(self, element, buffer, user_data):
        if buffer is None:
            logger.warning("Received None buffer.")
            return

        self._reload_enroll_zone_file()
        self._reload_exit_recognition_zone_file()
        self._reload_entry_lines_file()
        self._reload_exit_lines_file()

        roi = hailo.get_roi_from_buffer(buffer)
        self._body_person_filter_callback_for_roi(roi)
        person_detections = list(roi.get_objects_typed(hailo.HAILO_DETECTION))
        frame_number = user_data.get_count()
        active_track_ids = {
            track_id
            for track_id in (self._get_track_id(item) for item in person_detections)
            if track_id is not None
        }
        self._cleanup_stale_person_tracks(active_track_ids, frame_number)

        for person_detection in person_detections:
            track_id = self._get_track_id(person_detection)
            if track_id is None:
                continue
            self._add_identity_classification(
                person_detection,
                self.track_to_label.get(track_id, "Unknown"),
                1.0,
                self.person_tracker_name,
                track_id,
            )

        pad = element.get_static_pad("src")
        fmt, width, height = get_caps_from_pad(pad)
        debug_due = self.debug_stream_enabled and (
            self.rtsp_debug is None or self.rtsp_debug.due()
        )
        needs_frame = (
            self.debug_face_overlay
            or debug_due
            or self.entry_counter_enabled
            or self.exit_counter_enabled
        )
        frame = (
            get_numpy_from_buffer_efficient(buffer, fmt, width, height)
            if needs_frame
            else None
        )
        self._update_entry_counter(person_detections, frame_number, frame, width, height)
        self._update_exit_counter(person_detections, frame_number, frame, width, height)

        stats = self.recognition_stats
        stats["frames"] += 1
        stats["persons"] += len(person_detections)
        self._cleanup_expired_pending_unknowns()

        for person_detection in person_detections:
            track_id = self._get_track_id(person_detection)
            if track_id is None:
                stats["no_person_track"] += 1
                continue

            matrices = person_detection.get_objects_typed(hailo.HAILO_MATRIX)
            if not matrices:
                stats["no_embedding"] += 1
                continue
            if len(matrices) > 1:
                stats["multiple_embeddings"] += 1
            embedding_vector = np.array(matrices[0].get_data())
            for matrix in matrices:
                person_detection.remove_object(matrix)
            stats["embeddings"] += 1
            stats["matched"] += 1

            known_global_id = self.track_to_global_id.get(track_id)
            if known_global_id is not None:
                known_label = self.track_to_label.get(track_id, known_global_id)
                self._add_identity_classification(
                    person_detection,
                    known_label,
                    1.0,
                    self.person_tracker_name,
                    track_id,
                )
                stats["known"] += 1
                self._print_identity(
                    track_id,
                    known_global_id,
                    known_label,
                    1.0,
                    "recognized-body",
                )
                if self.camera_mode == CameraMode.ENTRY:
                    if frame is None:
                        frame = get_numpy_from_buffer_efficient(buffer, fmt, width, height)
                    self._save_known_person_sample_if_needed(
                        known_global_id,
                        known_label,
                        track_id,
                        frame,
                        person_detection,
                        embedding_vector,
                        width,
                        height,
                        frame_number,
                    )
                continue

            if self.camera_mode == CameraMode.EXIT:
                if not self._is_inside_exit_recognition_zone(
                    person_detection,
                    person_detection,
                ):
                    continue
                person, confidence = self._recognize_entered_embedding(embedding_vector)
                if person["label"] == "Unknown":
                    stats["unknown"] += 1
                    continue
                stable_vote = self._record_pending_vote(track_id, person, confidence)
                if stable_vote is None or stable_vote.global_id is None:
                    stats["unknown"] += 1
                    continue
                stable_person = self.db_handler.get_record_by_id(stable_vote.global_id)
                if stable_person is None:
                    stats["unknown"] += 1
                    continue
                self._bind_entered_person_for_exit_camera(
                    track_id,
                    stable_person,
                    stable_vote.confidence,
                    person_detection,
                    frame,
                    width,
                    height,
                )
                stats["known"] += 1
                continue

            # ENTRY never searches the old database: every new camera track is a
            # fresh admission identity, later searchable only while it is inside.
            self.pending_unknowns.setdefault(track_id, PendingIdentity())
            stats["unknown"] += 1
            if frame is None:
                frame = get_numpy_from_buffer_efficient(buffer, fmt, width, height)
            self._handle_unknown_person(
                track_id,
                frame_number,
                frame,
                person_detection,
                person_detection,
                embedding_vector,
                width,
                height,
            )

        if frame is not None and (self.debug_face_overlay or debug_due):
            debug_frame = (
                frame.copy() if self.debug_face_overlay else self._prepare_debug_frame(frame)
            )
            debug_height, debug_width = debug_frame.shape[:2]
            self._draw_debug_overlay(
                debug_frame,
                person_detections,
                [],
                debug_width,
                debug_height,
                frame_number,
            )
            if self.debug_face_overlay:
                user_data.set_frame(debug_frame)
            if debug_due:
                if self.debug_face_overlay and self.debug_stream_width:
                    debug_frame = self._prepare_debug_frame(debug_frame)
                if self.rtsp_debug is not None:
                    self.rtsp_debug.submit(debug_frame)
                else:
                    user_data.set_debug_frame(
                        debug_frame,
                        frame_number,
                        self.debug_jpeg_quality,
                    )

        self._log_recognition_stats(frame_number)
        return Gst.FlowReturn.OK

    @staticmethod
    def _body_person_filter_callback_for_roi(roi) -> None:
        for detection in list(roi.get_objects_typed(hailo.HAILO_DETECTION)):
            if detection.get_label() != "person":
                roi.remove_object(detection)


def main() -> None:
    logger.info("Starting whole-person body ReID app.")
    user_data = PersonFaceIdData()
    app = PersonBodyIdApp(user_data)
    try:
        app.run()
    finally:
        if app.rtsp_debug is not None:
            app.rtsp_debug.close()


if __name__ == "__main__":
    main()
