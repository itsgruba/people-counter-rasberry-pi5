"""Regression tests for entry identities without Hailo/GStreamer imports."""
import ast
from pathlib import Path
from types import SimpleNamespace
import time
import unittest
from unittest.mock import Mock

SOURCE = Path(__file__).resolve().parents[1] / 'hailo_apps/my_projects/auto_face_id/person_face_id.py'
TREE = ast.parse(SOURCE.read_text())
APP = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == 'PersonFaceIdApp')
METHODS = {n.name: n for n in APP.body if isinstance(n, ast.FunctionDef)}


def isolated_app():
    namespace = {'time': time}
    cls = ast.ClassDef(name='App', bases=[], keywords=[], decorator_list=[],
                       body=[METHODS['_enroll_if_ready']])
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), 'exec'), namespace)
    app = namespace['App']()
    app.samples_per_person = 2
    app.min_unknown_age_seconds = 0
    app.entry_counter_enabled = True
    app.entry_detector = Mock()
    app.pending_unknowns = {}
    app.track_to_global_id, app.track_to_label = {}, {}
    app._make_person_label = Mock(side_effect=['Person1', 'Person2'])
    app._create_person_from_pending_samples = Mock(side_effect=[{'global_id': 'new-1', 'visit_count': 1}, {'global_id': 'new-2', 'visit_count': 1}])
    for name in ('_recognize_embedding', '_bind_existing_person_from_vote', '_stable_pending_vote'):
        setattr(app, name, Mock(side_effect=AssertionError('Historical recognition must never run on entry')))
    for name in ('_add_identity_classification', '_print_identity', '_apply_pending_entry_for_track', '_notify_frontend'):
        setattr(app, name, Mock())
    app.person_tracker_name = 'person'
    return app


class AdmissionTests(unittest.TestCase):
    def test_repeat_face_creates_distinct_admissions(self):
        app = isolated_app()
        # The same sample data must not cause reuse of a previous admission.
        samples = [object(), object()]
        for track in (10, 20):
            app.pending_unknowns[track] = SimpleNamespace(samples=samples, first_seen_time=0)
            app._enroll_if_ready(track, object(), object(), 100, 100)
        self.assertEqual(app.track_to_global_id, {10: 'new-1', 20: 'new-2'})
        self.assertFalse(app.pending_unknowns)
        self.assertEqual(app._apply_pending_entry_for_track.call_count, 2)

    def test_does_not_create_before_crossing(self):
        app = isolated_app()
        app.pending_unknowns[10] = SimpleNamespace(samples=[1, 2], first_seen_time=0)
        app.entry_detector.uncounted_crossing.return_value = None
        app._enroll_if_ready(10, object(), object(), 100, 100)
        app._create_person_from_pending_samples.assert_not_called()

    def test_no_historical_search_in_callback_or_enrollment(self):
        for name in ('pipeline_callback', '_enroll_if_ready'):
            calls = {n.func.attr for n in ast.walk(METHODS[name]) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
            self.assertNotIn('_recognize_embedding', calls)
            self.assertNotIn('_bind_existing_person_from_vote', calls)
        calls = {n.func.attr for n in ast.walk(METHODS['pipeline_callback']) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertIn('_recognize_entered_embedding', calls)


if __name__ == '__main__':
    unittest.main()
