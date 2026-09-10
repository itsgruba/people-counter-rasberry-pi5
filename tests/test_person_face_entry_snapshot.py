"""Exercise delayed entry snapshots without Hailo or GStreamer."""
import ast
from dataclasses import dataclass, field
import logging
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image

SOURCE = Path(__file__).resolve().parents[1] / 'hailo_apps/my_projects/auto_face_id/person_face_id.py'
names = {'_handle_entry_crossing', '_mark_known_person_entered', '_record_visit_event_snapshot',
         '_save_visit_event_snapshot', '_visit_event_sample_dir', '_person_sample_dir',
         'crop_frame', 'save_image_file'}
nodes = []
for node in ast.parse(SOURCE.read_text()).body:
    if isinstance(node, ast.ClassDef) and node.name == 'PendingEntryContext':
        nodes.append(node)
    elif isinstance(node, ast.ClassDef) and node.name == 'PersonFaceIdApp':
        node.bases = []
        node.body = [m for m in node.body if isinstance(m, ast.FunctionDef) and m.name in names]
        nodes.append(node)
ns = dict(np=np, Image=Image, Path=Path, dataclass=dataclass, field=field, time=time, logger=logging.getLogger(__name__))
module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
exec(compile(ast.fix_missing_locations(module), str(SOURCE), 'exec'), ns)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.app = ns['PersonFaceIdApp']()
        self.app.samples_dir = Path(self.tmp.name)
        self.app.track_to_label = {}
        self.app.track_to_global_id = {}
        self.app.pending_entry_contexts = {}
        self.app.entry_pending_resolution_seconds = 2
        self.app.db_handler = Mock()
        self.app.db_handler.set_person_inside.return_value = dict(global_id='new', label='Person1', visit_count=1, total_entered=1)
        self.app.entry_detector = Mock()
        self.app._notify_frontend = Mock()
        self.app._remember_entered_person = Mock()
        self.event = SimpleNamespace(track_id=7, entered_at=100)
        bbox = SimpleNamespace(xmin=lambda: .3, xmax=lambda: .7, ymin=lambda: .3, ymax=lambda: .7)
        self.detection = SimpleNamespace(get_bbox=lambda: bbox)

    def finish(self, **kwargs):
        self.app._mark_known_person_entered('new', 'Person1', 7, 1.0, self.event, **kwargs)
        event = self.app.db_handler.add_visit_event.call_args.kwargs
        self.assertEqual(event['timestamp'], 100)
        self.assertEqual(event['global_id'], 'new')
        path = Path(event['photo_path'])
        self.assertTrue(path.is_file())
        self.assertEqual(self.app.db_handler.add_visit_record.call_args.kwargs['photo_path'], str(path))
        self.assertNotIn(7, self.app.pending_entry_contexts)
        with Image.open(path) as image:
            return np.array(image)

    def test_disappeared_person_and_reused_video_buffer(self):
        frame = np.full((100, 100, 3), 180, dtype=np.uint8)
        self.app._handle_entry_crossing(self.event, self.detection, frame, 100, 100)
        snapshot = self.app.pending_entry_contexts[7].snapshot
        self.assertFalse(np.shares_memory(frame, snapshot))
        self.assertLess(snapshot.size, frame.size)
        frame[:] = 0
        result = self.finish()  # Neither frame nor detection remains available.
        self.assertTrue(np.all(result == 180))

    def test_crossing_image_preferred_over_later_image(self):
        frame = np.full((100, 100, 3), 180, dtype=np.uint8)
        self.app._handle_entry_crossing(self.event, self.detection, frame, 100, 100)
        result = self.finish(frame=np.zeros_like(frame), person_detection=self.detection, width=100, height=100)
        self.assertTrue(np.all(result == 180))

    def test_immediate_known_entry_keeps_existing_path(self):
        result = self.finish(frame=np.full((100,100,3), 90, dtype=np.uint8), person_detection=self.detection, width=100, height=100)
        self.assertTrue(np.all(result == 90))

    def test_empty_crop_copies_full_frame(self):
        self.app.crop_frame = Mock(return_value=np.empty((0, 0, 3), dtype=np.uint8))
        frame = np.full((100,100,3), 70, dtype=np.uint8)
        self.app._handle_entry_crossing(self.event, self.detection, frame, 100, 100)
        frame[:] = 0
        result = self.finish()
        self.assertEqual(result.shape, (100, 100, 3))
        self.assertTrue(np.all(result == 70))


if __name__ == '__main__':
    unittest.main()
