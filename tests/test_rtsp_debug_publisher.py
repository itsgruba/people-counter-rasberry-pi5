"""Hardware-independent tests for bounded debug publishing and recovery."""
import importlib.util
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    'rtsp_debug', Path(__file__).resolve().parents[1]
    / 'hailo_apps/my_projects/auto_face_id/rtsp_debug.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Publisher = module.RtspDebugPublisher


class Frame:
    shape = (640, 640, 3)


class PublisherTests(unittest.TestCase):
    def test_missing_ffmpeg_fails_before_start(self):
        with patch.object(module.shutil, 'which', return_value=None):
            with self.assertRaises(RuntimeError):
                Publisher('rtsp://localhost:8554/debug')

    def test_slow_output_keeps_only_newest_pending_frame(self):
        entered, release, received = threading.Event(), threading.Event(), threading.Event()
        first, second, third = Frame(), Frame(), Frame()
        frames = []

        def write(process, frame):
            frames.append(frame)
            if frame is first:
                entered.set()
                release.wait(2)
            else:
                received.set()

        with patch.object(module.shutil, 'which', return_value='/mock/ffmpeg'), \
             patch.object(Publisher, '_start', return_value=object()), \
             patch.object(Publisher, '_stop'), \
             patch.object(Publisher, '_write', side_effect=write):
            publisher = Publisher('rtsp://localhost:8554/debug')
            try:
                publisher.submit(first)
                self.assertTrue(entered.wait(1))
                publisher.submit(second)
                publisher.submit(third)
                self.assertFalse(publisher.due())
                release.set()
                self.assertTrue(received.wait(1))
                self.assertEqual(frames, [first, third])
            finally:
                release.set()
                publisher.close()
            self.assertFalse(publisher._thread.is_alive())

    def test_close_interrupts_retry_backoff(self):
        failed = threading.Event()

        def write(*args):
            failed.set()
            raise BrokenPipeError('server disconnected')

        with patch.object(module.shutil, 'which', return_value='/mock/ffmpeg'), \
             patch.object(Publisher, '_start', return_value=object()), \
             patch.object(Publisher, '_stop'), \
             patch.object(Publisher, '_write', side_effect=write), \
             patch.object(module.logger, 'warning'):
            publisher = Publisher('rtsp://localhost:8554/debug')
            publisher.submit(Frame())
            self.assertTrue(failed.wait(1))
            start = time.monotonic()
            publisher.close()
            self.assertLess(time.monotonic() - start, 1)
            self.assertFalse(publisher._thread.is_alive())

    def test_restart_after_output_failure(self):
        recovered = threading.Event()
        calls = []

        def write(process, frame):
            calls.append(frame)
            if len(calls) == 1:
                raise BrokenPipeError('server disconnected')
            recovered.set()

        with patch.object(module.shutil, 'which', return_value='/mock/ffmpeg'), \
             patch.object(Publisher, '_start', return_value=object()) as start, \
             patch.object(Publisher, '_stop'), \
             patch.object(Publisher, '_write', side_effect=write), \
             patch.object(module.logger, 'warning'):
            publisher = Publisher('rtsp://localhost:8554/debug')
            try:
                publisher.submit(Frame())
                deadline = time.monotonic() + 1
                while not calls and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(calls)
                publisher.submit(Frame())
                self.assertTrue(recovered.wait(3))
                self.assertEqual(start.call_count, 2)
            finally:
                publisher.close()


if __name__ == '__main__':
    unittest.main()
