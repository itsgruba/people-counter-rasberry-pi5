"""Debug resizing must preserve the source used for recognition/snapshots."""
import ast
from pathlib import Path
import unittest
import cv2
import numpy as np

source = Path(__file__).resolve().parents[1] / 'hailo_apps/my_projects/auto_face_id/person_face_id.py'
app = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'PersonFaceIdApp')
app.bases = []
app.body = [n for n in app.body if isinstance(n, ast.FunctionDef) and n.name == '_prepare_debug_frame']
ns = {'cv2': cv2, 'np': np}
exec(compile(ast.fix_missing_locations(ast.Module(body=[app], type_ignores=[])), str(source), 'exec'), ns)


class DebugResizeTests(unittest.TestCase):
    def test_downscale_and_preserve_source(self):
        app = ns['PersonFaceIdApp']()
        app.debug_stream_width = 960
        frame = np.full((1080, 1920, 3), 100, dtype=np.uint8)
        result = app._prepare_debug_frame(frame)
        self.assertEqual(result.shape, (540, 960, 3))
        result[:] = 0
        self.assertTrue(np.all(frame == 100))

    def test_original_or_no_upscale_returns_copy(self):
        app = ns['PersonFaceIdApp']()
        frame = np.ones((480, 640, 3), dtype=np.uint8)
        for width in (0, 640, 960):
            app.debug_stream_width = width
            result = app._prepare_debug_frame(frame)
            self.assertEqual(result.shape, frame.shape)
            self.assertFalse(np.shares_memory(result, frame))


if __name__ == '__main__':
    unittest.main()
