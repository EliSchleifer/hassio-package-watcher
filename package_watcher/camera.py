"""Frame sources: RTSP streams (Unifi Protect) and video files.

RTSP handling notes:
- We `grab()` every frame to keep the decoder's buffer drained (otherwise
  OpenCV serves stale frames), but only `retrieve()`/decode at the sampling
  interval — that is what keeps this CPU-cheap.
- Unifi Protect exposes rtsps:// URLs (port 7441) per camera; enable the
  RTSP stream in Protect's camera settings and paste the URL into config.
- Connections drop; we reconnect with capped exponential backoff.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

FrameCallback = Callable[[np.ndarray, float], None]


def _is_stream(source: str) -> bool:
    return source.startswith(("rtsp://", "rtsps://", "http://", "https://"))


class FrameSource:
    """Reads a camera or file and invokes a callback at `sample_fps`."""

    def __init__(self, name: str, source: str, sample_fps: float = 1.0):
        self.name = name
        self.source = source
        self.sample_interval = 1.0 / max(sample_fps, 0.01)
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self, on_frame: FrameCallback) -> None:
        if _is_stream(self.source):
            self._run_stream(on_frame)
        else:
            self._run_file(on_frame)

    # ------------------------------------------------------------------
    def _open(self) -> Optional[cv2.VideoCapture]:
        # Prefer TCP for RTSP: fewer corrupt frames than UDP over wifi/VLANs.
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _run_stream(self, on_frame: FrameCallback) -> None:
        backoff = 2.0
        while not self._stop:
            cap = self._open()
            if cap is None:
                log.warning("[%s] cannot open %s; retrying in %.0fs",
                            self.name, self.source, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
                continue
            log.info("[%s] connected to stream", self.name)
            backoff = 2.0
            last_sample = 0.0
            while not self._stop:
                if not cap.grab():
                    log.warning("[%s] stream dropped; reconnecting", self.name)
                    break
                now = time.time()
                if now - last_sample < self.sample_interval:
                    continue
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    continue
                last_sample = now
                on_frame(frame, now)
            cap.release()
            if not self._stop:
                time.sleep(backoff)

    def _run_file(self, on_frame: FrameCallback) -> None:
        """One-shot mode for recorded clips: timestamps come from the video
        clock, and we sample every Nth frame to mimic live pacing."""
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video source: {self.source}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(fps * self.sample_interval)))
        index = 0
        start = time.time()
        while not self._stop:
            ok = cap.grab()
            if not ok:
                break
            if index % step == 0:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    on_frame(frame, start + index / fps)
            index += 1
        cap.release()
