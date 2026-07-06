"""Event payload + evidence bundle tests, end to end through the service."""

from __future__ import annotations

import json
import os

import numpy as np

from package_watcher.config import (AppConfig, CameraConfig, SinkConfig)
from package_watcher.detector import DetectorConfig, StaticObjectDetector
from package_watcher.events import TriggerInfo, build_event
from package_watcher.evidence import write_evidence
from package_watcher.service import WatcherService

from test_detector import PACKAGE, RNG_SEED, W, H, fast_config, make_frame


def _get_report():
    rng = np.random.default_rng(RNG_SEED)
    detector = StaticObjectDetector(fast_config())
    for i in range(30):
        detector.process(make_frame(rng), ts=1000.0 + i)
    reports = []
    for i in range(20):
        reports.extend(detector.process(
            make_frame(rng, package=PACKAGE), ts=1030.0 + i))
    assert len(reports) == 1
    return reports[0]


def test_event_payload_is_llm_ready():
    report = _get_report()
    trigger = TriggerInfo(kind="person", source="unifi-protect", at=1029.0)
    event = build_event("front-door", report, trigger=trigger)

    assert event["kind"] == "new_static_object"
    assert event["camera"] == "front-door"
    assert event["id"].startswith("front-door-")
    bbox = event["bbox_pixels"]
    assert all(k in bbox for k in ("x", "y", "w", "h"))
    norm = event["bbox_normalized"]
    assert 0 <= norm["x"] <= 1 and 0 <= norm["w"] <= 1
    assert event["trigger"]["kind"] == "person"
    assert 0 < event["signals"]["confidence"] <= 1
    prompt = event["llm_verification"]["suggested_prompt"]
    assert "front-door" in prompt and str(bbox["x"]) in prompt
    json.dumps(event)  # must be JSON-serializable as-is


def test_evidence_bundle_written(tmp_path):
    report = _get_report()
    event = build_event("front-door", report)
    paths = write_evidence(event, report, str(tmp_path))

    for kind in ("raw", "annotated", "crop", "baseline", "mask", "event"):
        assert kind in paths, f"missing evidence: {kind}"
        full = tmp_path / paths[kind]
        assert full.is_file() and full.stat().st_size > 0

    with open(tmp_path / paths["event"], encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["id"] == event["id"]
    assert saved["evidence"]["annotated"].endswith("annotated.jpg")


def test_service_end_to_end_on_synthetic_video(tmp_path):
    """Full pipeline: video file -> worker -> detector -> evidence + jsonl."""
    import cv2

    rng = np.random.default_rng(RNG_SEED)
    video_path = str(tmp_path / "clip.mp4")
    writer = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (W, H))
    assert writer.isOpened()
    for _ in range(30 * 10):
        writer.write(make_frame(rng))
    for _ in range(20 * 10):
        writer.write(make_frame(rng, package=PACKAGE))
    writer.release()

    jsonl = str(tmp_path / "events.jsonl")
    config = AppConfig(
        cameras=[CameraConfig(name="clip", source=video_path, sample_fps=1.0)],
        detector=DetectorConfig(resize_width=W, persist_samples=5,
                                slow_alpha=0.002, fast_alpha=0.2),
        sinks=SinkConfig(stdout=False, jsonl_path=jsonl),
        events_dir=str(tmp_path / "events"))
    WatcherService(config).run_forever()

    with open(jsonl, encoding="utf-8") as f:
        events = [json.loads(line) for line in f]
    assert len(events) == 1
    event = events[0]
    assert event["camera"] == "clip"
    evidence_dir = os.path.join(str(tmp_path / "events"), "clip", event["id"])
    assert os.path.isfile(os.path.join(evidence_dir, "annotated.jpg"))
    assert os.path.isfile(os.path.join(evidence_dir, "event.json"))
