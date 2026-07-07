"""Orchestration: one detector + frame source per camera, shared sinks.

Each camera gets its own thread (frame decoding dominates, and the GIL is
released inside OpenCV/ffmpeg calls, so threads scale fine for a fixed
handful of cameras). Trigger events set a per-camera attention deadline;
frames sampled before that deadline run the detector in high-attention mode.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from .camera import FrameSource
from .config import AppConfig, CameraConfig
from .detector import build_detector
from .events import TriggerInfo, build_event
from .evidence import write_evidence
from .sinks import EventSinks

log = logging.getLogger(__name__)


class CameraWorker:
    def __init__(self, cam: CameraConfig, app: "WatcherService"):
        self.cam = cam
        self.app = app
        self.detector = build_detector(app.config.detector, zone=cam.zone)
        self._gated = app.config.detector.mode == "person_gated"
        self.source = FrameSource(cam.name, cam.source, cam.sample_fps)
        self.attention_until = 0.0
        self.person_until = 0.0
        self.last_trigger: TriggerInfo | None = None
        self._lock = threading.Lock()

    def notify_trigger(self, kind: str, source: str, ts: float,
                       end_ts: float | None = None) -> None:
        unifi = self.app.config.unifi
        window = unifi.attention_seconds if unifi else 120.0
        with self._lock:
            self.attention_until = max(self.attention_until, ts + window)
            self.last_trigger = TriggerInfo(kind=kind, source=source, at=ts)
            if kind.lower() == "person":
                # Person presence for gated mode: hold from the event's end
                # (or from now while it is still ongoing); each websocket
                # update refreshes the hold.
                hold = unifi.presence_hold_seconds if unifi else 10.0
                anchor = end_ts if end_ts is not None else time.time()
                self.person_until = max(self.person_until, anchor + hold)
        log.info("[%s] attention window opened by %s trigger (until +%.0fs)",
                 self.cam.name, kind, window)

    def on_frame(self, frame: np.ndarray, ts: float) -> None:
        with self._lock:
            attention = ts <= self.attention_until
            person = ts <= self.person_until
            trigger = self.last_trigger if (attention or person) else None
        if self._gated:
            reports = self.detector.process(frame, ts,
                                            person_present=person)
        else:
            reports = self.detector.process(frame, ts, attention=attention)
        for report in reports:
            event = build_event(self.cam.name, report, trigger=trigger)
            try:
                write_evidence(event, report, self.app.config.events_dir)
            except Exception as exc:  # noqa: BLE001 - keep watching even if disk fails
                log.error("[%s] failed to write evidence: %s", self.cam.name, exc)
            log.info(
                "[%s] new static object at x=%d y=%d w=%d h=%d "
                "(confidence %.2f, persisted %d samples%s)",
                self.cam.name, *report.bbox, report.confidence,
                report.samples_persisted,
                ", triggered" if report.triggered else "")
            self.app.sinks.emit(event)

    def run(self) -> None:
        self.source.run(self.on_frame)

    def stop(self) -> None:
        self.source.stop()


class WatcherService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.sinks = EventSinks(config.sinks)
        self.workers = {cam.name: CameraWorker(cam, self)
                        for cam in config.cameras}
        self._threads: list[threading.Thread] = []
        self._trigger_listener = None

    def notify_trigger(self, camera: str, kind: str, ts: float,
                       end_ts: float | None = None,
                       source: str = "unifi-protect") -> None:
        worker = self.workers.get(camera)
        if worker is None:
            log.debug("trigger for unknown camera %r ignored", camera)
            return
        worker.notify_trigger(kind, source, ts, end_ts=end_ts)

    def start(self) -> None:
        if self.config.unifi:
            from .triggers.unifi import UnifiTriggerListener
            self._trigger_listener = UnifiTriggerListener(
                self.config.unifi, self.notify_trigger)
            self._trigger_listener.start()
        for name, worker in self.workers.items():
            t = threading.Thread(target=worker.run, name=f"cam-{name}",
                                 daemon=True)
            t.start()
            self._threads.append(t)
        log.info("watching %d camera(s); events -> %s",
                 len(self.workers), self.config.events_dir)

    def stop(self) -> None:
        for worker in self.workers.values():
            worker.stop()
        if self._trigger_listener:
            self._trigger_listener.stop()

    def run_forever(self) -> None:
        self.start()
        try:
            while any(t.is_alive() for t in self._threads):
                time.sleep(1.0)
            log.info("all camera workers finished")
        except KeyboardInterrupt:
            log.info("shutting down")
        finally:
            self.stop()
