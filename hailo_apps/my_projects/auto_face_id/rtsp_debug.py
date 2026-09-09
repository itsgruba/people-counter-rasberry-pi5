"""Bounded, asynchronous RGB -> H.264 -> MediaMTX debug publisher."""
import logging
import os
import select
import shutil
import subprocess
import threading
import time

logger = logging.getLogger(__name__)


class RtspDebugPublisher:
    def __init__(self, url, fps=10, bitrate=2000):
        if fps <= 0 or bitrate <= 0:
            raise ValueError('Debug FPS and bitrate must be positive')
        if not url.startswith('rtsp://'):
            raise ValueError('Debug URL must start with rtsp://')
        if not shutil.which('ffmpeg'):
            raise RuntimeError('RTSP debug output requires ffmpeg with libx264')
        self.url, self.fps, self.bitrate = url, fps, bitrate
        self._condition = threading.Condition()
        self._pending = None
        self._stopped = False
        self._next_frame = 0.0
        self._thread = threading.Thread(target=self._run, name='rtsp-debug', daemon=True)
        self._thread.start()

    def due(self):
        return not self._stopped and time.monotonic() >= self._next_frame

    def submit(self, frame):
        # Caller owns this already copied RGB frame; never retain Gst buffer views.
        with self._condition:
            self._pending = frame
            self._next_frame = time.monotonic() + 1 / self.fps
            self._condition.notify()

    def close(self):
        with self._condition:
            self._stopped = True
            self._pending = None
            self._condition.notify_all()
        self._thread.join(timeout=5)

    def _start(self, width, height):
        process = subprocess.Popen([
            'ffmpeg', '-hide_banner', '-loglevel', 'warning',
            '-f', 'rawvideo', '-pixel_format', 'rgb24',
            '-video_size', f'{width}x{height}', '-framerate', str(self.fps),
            '-use_wallclock_as_timestamps', '1', '-i', 'pipe:0',
            '-fps_mode', 'passthrough', '-an', '-c:v', 'libx264', '-preset', 'ultrafast',
            '-tune', 'zerolatency', '-threads', '2', '-pix_fmt', 'yuv420p',
            '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
            '-b:v', f'{self.bitrate}k', '-maxrate', f'{self.bitrate}k',
            '-bufsize', f'{self.bitrate}k', '-g', str(max(1, round(self.fps))),
            '-bf', '0', '-f', 'rtsp', '-rtsp_transport', 'tcp', self.url,
        ], stdin=subprocess.PIPE, bufsize=0)
        os.set_blocking(process.stdin.fileno(), False)
        return process

    @staticmethod
    def _stop(process):
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdin.close()

    def _write(self, process, frame):
        data = memoryview(frame.tobytes())
        deadline = time.monotonic() + 1
        while data:
            if self._stopped or time.monotonic() >= deadline:
                raise TimeoutError('RTSP publisher stalled')
            if process.poll() is not None:
                raise BrokenPipeError('ffmpeg exited')
            fd = process.stdin.fileno()
            if select.select([], [fd], [], 0.1)[1]:
                try:
                    data = data[os.write(fd, data):]
                except BlockingIOError:
                    pass

    def _run(self):
        process, shape = None, None
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._stopped or self._pending is not None)
                    if self._stopped:
                        return
                    frame, self._pending = self._pending, None
                try:
                    if process is None or shape != frame.shape:
                        self._stop(process)
                        process = None
                        process = self._start(frame.shape[1], frame.shape[0])
                        shape = frame.shape
                    self._write(process, frame)
                except (OSError, TimeoutError):
                    if not self._stopped:
                        logger.warning('RTSP debug publish failed; retrying in 2 seconds', exc_info=True)
                    self._stop(process)
                    process = None
                    with self._condition:
                        self._condition.wait_for(lambda: self._stopped, timeout=2)
        finally:
            self._stop(process)
