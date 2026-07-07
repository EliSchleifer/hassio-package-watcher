"""Integration-test harness: play a clip through the detector and grade it.

A *fixture case* is a real video clip plus an expectation — "this should
detect a package" or "this should not". The harness runs the detector over
the clip with **video-relative timestamps** (sample `i` at `i / fps` seconds,
no wall clock) so a case grades identically every run, then compares the
reports against the expectation.

Clips are real footage and stay local (fixtures/clips/ is gitignored); a
case whose clip is absent is skipped, not failed, so the manifest can be
shared without sharing the footage.

This is the backbone the pytest fixture suite and the authoring UI both call.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import numpy as np
import yaml

from .detector import DetectorConfig, NewObjectReport, build_detector


@dataclass
class FixtureCase:
    name: str
    expect: str                       # "detect" | "no_detect"
    clip: Optional[str] = None        # path relative to the fixtures dir
    fps: float = 2.0
    zone: Optional[list[tuple[float, float]]] = None
    detector: dict[str, Any] = field(default_factory=dict)  # config overrides
    attention: list[tuple[float, float]] = field(default_factory=list)
    # Person-in-frame windows [(start_s, end_s), ...], clip-relative. Drives
    # detector mode "person_gated" (set via detector: {mode: person_gated});
    # typically auto-imported from Protect smart-detect events with the clip.
    presence: list[tuple[float, float]] = field(default_factory=list)
    # Optional tighter expectations for a "detect" case:
    region: Optional[tuple[float, float, float, float]] = None  # normalized
    after: Optional[float] = None     # detection must occur at/after this (s)
    before: Optional[float] = None    # ...and at/before this (s)
    max_reports: Optional[int] = None
    description: str = ""

    def __post_init__(self):
        if self.expect not in ("detect", "no_detect"):
            raise ValueError(
                f"case {self.name!r}: expect must be 'detect' or 'no_detect'")
        if not self.clip:
            raise ValueError(f"case {self.name!r}: needs a clip")


@dataclass
class Detection:
    t: float                                   # clip-relative seconds
    bbox_norm: tuple[float, float, float, float]
    confidence: float
    triggered: bool
    candidate_id: int


@dataclass
class ClipResult:
    case: FixtureCase
    detections: list[Detection]
    samples: int
    scene_resets: int
    frames_for_preview: dict[str, np.ndarray] = field(default_factory=dict)
    # Annotated frame + mask per detection (aligned with `detections`), only
    # populated when capture_preview=True. Lets the UI show the frame of the
    # detection that actually matched the expectation, not just the first.
    det_frames: list[np.ndarray] = field(default_factory=list)
    det_masks: list[np.ndarray] = field(default_factory=list)


@dataclass
class CaseOutcome:
    case: FixtureCase
    result: ClipResult
    passed: bool
    reason: str
    matched_index: Optional[int] = None  # index into result.detections that
                                          # satisfied a "detect" expectation


_VALID_DETECTOR_FIELDS = set(DetectorConfig.__dataclass_fields__)


def load_cases(manifest_path: str) -> list[FixtureCase]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cases = []
    for entry in raw.get("cases", []):
        entry = dict(entry)
        if "zone" in entry and entry["zone"] is not None:
            entry["zone"] = [tuple(pt) for pt in entry["zone"]]
        if "region" in entry and entry["region"] is not None:
            entry["region"] = tuple(entry["region"])
        if "attention" in entry and entry["attention"] is not None:
            entry["attention"] = [tuple(w) for w in entry["attention"]]
        if "presence" in entry and entry["presence"] is not None:
            entry["presence"] = [tuple(w) for w in entry["presence"]]
        unknown = set(entry) - set(FixtureCase.__dataclass_fields__)
        if unknown:
            raise ValueError(
                f"case {entry.get('name')!r}: unknown keys {sorted(unknown)}")
        bad = set(entry.get("detector", {})) - _VALID_DETECTOR_FIELDS
        if bad:
            raise ValueError(
                f"case {entry.get('name')!r}: unknown detector keys {sorted(bad)}")
        cases.append(FixtureCase(**entry))
    return cases


def iter_samples(case: FixtureCase, fixtures_dir: str
                 ) -> Iterator[tuple[np.ndarray, float]]:
    """Yield (frame, clip_relative_seconds) for a case, deterministically."""
    import cv2

    path = case.clip if os.path.isabs(case.clip) else \
        os.path.join(fixtures_dir, case.clip)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open clip: {path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(video_fps / case.fps)))
    index = 0
    try:
        while True:
            if not cap.grab():
                break
            if index % step == 0:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    yield frame, index / video_fps
            index += 1
    finally:
        cap.release()


def _in_windows(t: float, windows: list[tuple[float, float]]) -> bool:
    return any(a <= t <= b for a, b in windows)


def run_case(case: FixtureCase, fixtures_dir: str = ".",
             capture_preview: bool = False) -> ClipResult:
    cfg = DetectorConfig(**case.detector)
    detector = build_detector(cfg, zone=case.zone)
    gated = cfg.mode == "person_gated"
    detections: list[Detection] = []
    det_frames: list[np.ndarray] = []
    det_masks: list[np.ndarray] = []
    samples = 0
    preview: dict[str, np.ndarray] = {}
    first_frame: Optional[np.ndarray] = None

    for frame, t in iter_samples(case, fixtures_dir):
        if capture_preview and first_frame is None:
            first_frame = frame.copy()
        if gated:
            reports = detector.process(
                frame, ts=t, person_present=_in_windows(t, case.presence))
        else:
            reports = detector.process(
                frame, ts=t, attention=_in_windows(t, case.attention))
        for report in reports:
            detections.append(Detection(
                t=report.reported_at, bbox_norm=report.bbox_norm,
                confidence=report.confidence, triggered=report.triggered,
                candidate_id=report.candidate_id))
            if capture_preview:
                annotated = _annotate(report)
                det_frames.append(annotated)
                det_masks.append(report.mask)
                if "detection" not in preview:
                    preview["detection"] = annotated
                    preview["baseline"] = report.baseline
                    preview["mask"] = report.mask
        samples += 1

    if capture_preview:
        if first_frame is not None:
            preview.setdefault("first", first_frame)
    return ClipResult(case=case, detections=detections, samples=samples,
                      scene_resets=detector.scene_resets,
                      frames_for_preview=preview,
                      det_frames=det_frames, det_masks=det_masks)


def _annotate(report: NewObjectReport) -> np.ndarray:
    import cv2

    frame = report.frame.copy()
    x, y, w, h = report.bbox
    thickness = max(2, frame.shape[1] // 640)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 220, 60), thickness)
    cv2.putText(frame, f"{report.confidence:.2f}", (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 220, 60), thickness)
    return frame


def _iou_norm(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _center_in(det, region) -> bool:
    cx, cy = det[0] + det[2] / 2, det[1] + det[3] / 2
    rx, ry, rw, rh = region
    return rx <= cx <= rx + rw and ry <= cy <= ry + rh


def evaluate(result: ClipResult) -> CaseOutcome:
    case = result.case
    dets = result.detections

    def outcome(passed: bool, reason: str,
                matched: Optional[int] = None) -> CaseOutcome:
        return CaseOutcome(case=case, result=result, passed=passed,
                           reason=reason, matched_index=matched)

    if case.expect == "no_detect":
        if dets:
            first = dets[0]
            return outcome(
                False,
                f"expected no detection but got {len(dets)} "
                f"(first at t={first.t:.1f}s, region={_fmt(first.bbox_norm)})")
        return outcome(True, f"no detection across {result.samples} samples")

    # expect == "detect"
    if not dets:
        return outcome(
            False, f"expected a detection but none fired across "
                   f"{result.samples} samples")
    if case.max_reports is not None and len(dets) > case.max_reports:
        return outcome(
            False, f"expected at most {case.max_reports} detection(s), "
                   f"got {len(dets)}")

    matching = dets
    if case.region is not None:
        matching = [d for d in dets
                    if _iou_norm(d.bbox_norm, case.region) > 0.1
                    or _center_in(d.bbox_norm, case.region)]
        if not matching:
            return outcome(
                False, f"detection(s) fired but none overlapped expected "
                       f"region {_fmt(case.region)} "
                       f"(got {_fmt(dets[0].bbox_norm)})")
    if case.after is not None or case.before is not None:
        lo = case.after if case.after is not None else float("-inf")
        hi = case.before if case.before is not None else float("inf")
        timed = [d for d in matching if lo <= d.t <= hi]
        if not timed:
            return outcome(
                False, f"detection(s) fired but none in window "
                       f"[{case.after}, {case.before}]s "
                       f"(got t={matching[0].t:.1f}s)")
        matching = timed

    d = matching[0]
    matched_idx = next(i for i, x in enumerate(dets) if x is d)
    return outcome(
        True, f"detected at t={d.t:.1f}s, region={_fmt(d.bbox_norm)}, "
              f"confidence={d.confidence:.2f}", matched=matched_idx)


def _fmt(bbox) -> str:
    return "(" + ", ".join(f"{v:.2f}" for v in bbox) + ")"


def run_and_evaluate(case: FixtureCase, fixtures_dir: str = ".",
                     capture_preview: bool = False) -> CaseOutcome:
    return evaluate(run_case(case, fixtures_dir, capture_preview))
