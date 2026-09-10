"""Hardware-independent tests of the application's geometry and live config.

Extract the actual classes/methods to avoid importing Hailo/GStreamer on CI.
Run with: python3 -m unittest discover -s tests -p test_person_face_lines.py
"""
import ast
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
import time
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import Mock

SOURCE = Path(__file__).resolve().parents[1] / 'hailo_apps/my_projects/auto_face_id/person_face_id.py'
tree = ast.parse(SOURCE.read_text())
methods = {'_read_enroll_zone_file', '_read_crossing_lines_file', '_read_entry_lines_file',
           '_read_exit_lines_file', '_apply_crossing_lines', '_draw_active_recognition_zone',
           '_draw_zone'}
nodes = []
for node in tree.body:
    if isinstance(node, ast.ClassDef):
        if node.name in {'EntryDetector', 'EntryTrackState', 'LineCrossingEvent'}:
            nodes.append(node)
        elif node.name == 'PersonFaceIdApp':
            node.bases = []
            node.body = [m for m in node.body if isinstance(m, ast.FunctionDef) and m.name in methods]
            nodes.append(node)
ns = dict(dataclass=dataclass, field=field, time=time, uuid=uuid, Path=Path)
exec(compile('from __future__ import annotations', '<annotations>', 'exec'), ns)
module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
exec(compile(ast.fix_missing_locations(module), str(SOURCE), 'exec'), ns)
Detector, App = ns['EntryDetector'], ns['PersonFaceIdApp']


def detection(x, y):
    return SimpleNamespace(get_bbox=lambda: SimpleNamespace(xmin=lambda: x, xmax=lambda: x, ymax=lambda: y))


class CrossingTests(unittest.TestCase):
    def detector(self, a=((0, .3), (1, .3)), b=((0, .7), (1, .7)), margin=.02):
        detector = Detector(.3, .7, margin, 60, 0)
        App._apply_crossing_lines(detector, {'entry_line_a': a, 'entry_line_b': b}, 'entry')
        return detector

    def walk(self, detector, points):
        return [detector.update(1, detection(*p), i, i) for i, p in enumerate(points)]

    def test_horizontal_and_reverse(self):
        self.assertIsNotNone(self.walk(self.detector(), [(.5, .1), (.5, .9)])[-1])
        self.assertTrue(all(e is None for e in self.walk(self.detector(), [(.5, .9), (.5, .1)])))

    def test_vertical(self):
        d = self.detector(((.3, 0), (.3, 1)), ((.7, 1), (.7, 0)))
        self.assertIsNotNone(self.walk(d, [(.1, .5), (.5, .5), (.9, .5)])[-1])

    def test_diagonal(self):
        d = self.detector(((0, .1), (1, .5)), ((0, .5), (1, .9)))
        self.assertIsNotNone(self.walk(d, [(.5, .1), (.5, .5), (.5, .95)])[-1])

    def test_outside_segment(self):
        d = self.detector(((.2, .3), (.6, .3)), ((.2, .7), (.6, .7)))
        self.assertTrue(all(e is None for e in self.walk(d, [(.9, .1), (.9, .5), (.9, .9)])))

    def test_hysteresis_and_no_duplicate(self):
        d = self.detector()
        result = self.walk(d, [(.5, .1), (.5, .299), (.5, .301), (.5, .5), (.5, .699), (.5, .701), (.5, .9), (.5, .1), (.5, .9)])
        self.assertEqual(sum(e is not None for e in result), 1)
        self.assertIsNotNone(result[6])

    def test_reload_validation_and_reset(self):
        d = self.detector()
        self.walk(d, [(.5, .1), (.5, .5)])
        for line in [((0, 0), (0, 0)), ((-1, 0), (1, 1)), ((float('nan'), 0), (1, 1)), d.line_b]:
            with self.assertRaises(ValueError):
                App._apply_crossing_lines(d, {'entry_line_a': line}, 'entry')
            self.assertTrue(d.tracks)
        App._apply_crossing_lines(d, {'entry_line_a': ((0, .2), (1, .4))}, 'entry')
        self.assertFalse(d.tracks)

    def test_config_and_polygon_independent_of_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'zone.txt'
            for prefix in ('entry', 'exit'):
                path.write_text(f'{prefix}_line_a=0.1,0.2,0.8,0.4\n{prefix}_line_b_y=0.8\n0.01,0.5,0.3,0.5,0.75,0.95,0.08,0.95\n')
                values = App._read_crossing_lines_file(path, prefix)
                self.assertEqual(values[f'{prefix}_line_a'], ((.1, .2), (.8, .4)))
                self.assertEqual(values[f'{prefix}_line_b'], ((0, .8), (1, .8)))
                self.assertEqual(App._read_enroll_zone_file(path), '0.01,0.5,0.3,0.5,0.75,0.95,0.08,0.95')

    def test_exit_polygon_outline(self):
        ns['CameraMode'] = SimpleNamespace(EXIT='exit')
        ns['cv2'] = Mock()
        ns['np'] = SimpleNamespace(array=lambda points, dtype: points, int32=int)
        app = App()
        app.camera_mode = 'exit'
        app.exit_recognition_zone = [(.01,.5),(.3,.5),(.75,.95),(.08,.95)]
        app._draw_active_recognition_zone(object(), 1001, 1001)
        args, kwargs = ns['cv2'].polylines.call_args
        self.assertEqual(args[1], [[(10,500),(300,500),(750,950),(80,950)]])
        self.assertTrue(kwargs['isClosed'])


if __name__ == '__main__':
    unittest.main()
