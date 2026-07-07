"""PersonGatedDetector: before/after-visit comparison, on synthetic frames.

The contract under test: while a person is in frame nothing is detected and
the reference freezes; after they leave and the scene settles, anything new,
static, and package-shaped relative to the pre-visit reference is reported
exactly once; the reference then re-baselines for the next visit.
"""

from __future__ import annotations

import numpy as np
import pytest

from package_watcher.detector import (DetectorConfig, PersonGatedDetector,
                                      build_detector)

RNG_SEED = 99
W, H = 320, 240
PACKAGE = (140, 150, 46, 32)  # x, y, w, h ground truth
PERSON_COL = (150, 96, 26, 108)  # tall dark figure, mid-frame


def make_frame(rng: np.random.Generator, package: bool = False,
               person: bool = False, brightness: float = 0.0) -> np.ndarray:
    frame = np.full((H, W, 3), 120, dtype=np.float32)
    frame[: H // 3] = 160
    frame += brightness
    frame += rng.normal(0, 3, size=frame.shape).astype(np.float32)
    if package:
        x, y, w, h = PACKAGE
        frame[y:y + h, x:x + w] = (55, 70, 95)
    if person:
        x, y, w, h = PERSON_COL
        frame[y:y + h, x:x + w] = (28, 28, 30)
    return np.clip(frame, 0, 255).astype(np.uint8)


def gated_config(**overrides) -> DetectorConfig:
    base = dict(mode="person_gated", resize_width=W, settle_samples=3)
    base.update(overrides)
    return DetectorConfig(**base)


def run(det: PersonGatedDetector, seq) -> list:
    """seq: iterable of (frame, person_present). Samples 0.5s apart."""
    reports = []
    for i, (frame, present) in enumerate(seq):
        reports.extend(det.process(frame, ts=1000.0 + i * 0.5,
                                   person_present=present))
    return reports


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / float(aw * ah + bw * bh - inter) if inter else 0.0


def delivery_sequence(rng, empty=10, visit=8, after=10):
    """empty porch -> person visits (package appears mid-visit) -> package."""
    seq = [(make_frame(rng), False) for _ in range(empty)]
    seq += [(make_frame(rng, person=True), True) for _ in range(visit // 2)]
    seq += [(make_frame(rng, person=True, package=True), True)
            for _ in range(visit - visit // 2)]
    seq += [(make_frame(rng, package=True), False) for _ in range(after)]
    return seq


def test_delivery_reported_once_after_person_leaves():
    rng = np.random.default_rng(RNG_SEED)
    det = build_detector(gated_config())
    reports = run(det, delivery_sequence(rng))
    assert len(reports) == 1
    r = reports[0]
    assert iou(r.bbox, PACKAGE) > 0.5
    assert r.triggered is True
    assert r.samples_persisted >= 3
    # Baseline is the pre-visit truth: package region looks like background.
    x, y, w, h = PACKAGE
    assert abs(float(r.baseline[y:y + h, x:x + w].mean()) - 120) < 15


def test_report_comes_shortly_after_settle():
    rng = np.random.default_rng(RNG_SEED)
    det = build_detector(gated_config(settle_samples=3))
    seq = delivery_sequence(rng, empty=10, visit=8, after=12)
    firing_index = None
    for i, (frame, present) in enumerate(seq):
        if det.process(frame, ts=1000.0 + i * 0.5, person_present=present):
            firing_index = i
            break
    # Person leaves at index 18; the transition frame shows motion, then
    # settle_samples quiet frames are required.
    assert firing_index is not None
    assert 18 < firing_index <= 18 + 3 + 3


def test_person_pauses_and_leaves_nothing():
    rng = np.random.default_rng(RNG_SEED)
    det = build_detector(gated_config())
    seq = [(make_frame(rng), False) for _ in range(10)]
    seq += [(make_frame(rng, person=True), True) for _ in range(20)]  # long pause
    seq += [(make_frame(rng), False) for _ in range(10)]
    assert run(det, seq) == []


def test_no_person_signal_means_no_report():
    # Documented tradeoff: an object appearing without a person event is
    # never reported in gated mode.
    rng = np.random.default_rng(RNG_SEED)
    det = build_detector(gated_config())
    seq = [(make_frame(rng), False) for _ in range(10)]
    seq += [(make_frame(rng, package=True), False) for _ in range(20)]
    assert run(det, seq) == []


def test_second_delivery_rebaselines():
    rng = np.random.default_rng(RNG_SEED)
    det = build_detector(gated_config())
    reports = run(det, delivery_sequence(rng))
    assert len(reports) == 1

    second = (40, 60, 40, 30)

    def two_pkg_frame(person=False):
        f = make_frame(rng, package=True, person=person)
        x, y, w, h = second
        f[y:y + h, x:x + w] = (200, 40, 40)
        return f

    # Second visit: first package still there, second box appears mid-visit.
    seq = [(make_frame(rng, package=True), False) for _ in range(6)]
    seq += [(make_frame(rng, package=True, person=True), True) for _ in range(3)]
    seq += [(two_pkg_frame(person=True), True) for _ in range(3)]
    seq += [(two_pkg_frame(), False) for _ in range(10)]
    reports2 = run(det, seq)
    assert len(reports2) == 1
    assert iou(reports2[0].bbox, second) > 0.5


def test_lighting_flip_during_visit_is_not_reported():
    rng = np.random.default_rng(RNG_SEED)
    det = build_detector(gated_config())
    seq = [(make_frame(rng), False) for _ in range(10)]
    seq += [(make_frame(rng, person=True), True) for _ in range(6)]
    # Person leaves; porch light also came on during the visit — a global
    # change, not a package.
    seq += [(make_frame(rng, brightness=80.0), False) for _ in range(10)]
    assert run(det, seq) == []
    assert det.scene_resets >= 1


def test_idle_motion_does_not_poison_reference():
    # A cat crossing while idle (no person signal) must not smear into the
    # reference, so the next visit still compares against a clean truth.
    rng = np.random.default_rng(RNG_SEED)
    det = build_detector(gated_config())
    seq = [(make_frame(rng), False) for _ in range(10)]
    for i in range(8):  # moving dark blob, no person flag
        f = make_frame(rng)
        x = int((0.1 + 0.7 * i / 7) * W)
        f[170:200, x:x + 24] = (20, 20, 20)
        seq.append((f, False))
    seq += [(make_frame(rng), False) for _ in range(6)]
    # Now a visit that leaves nothing.
    seq += [(make_frame(rng, person=True), True) for _ in range(5)]
    seq += [(make_frame(rng), False) for _ in range(8)]
    assert run(det, seq) == []


def test_reference_tracks_slow_drift_while_idle():
    rng = np.random.default_rng(RNG_SEED)
    det = build_detector(gated_config())
    # 60 idle samples of slow brightening (sunset-style), then a delivery.
    seq = [(make_frame(rng, brightness=i * 0.5), False) for i in range(60)]
    b = 30.0  # brightness by the end of the drift
    seq += [(make_frame(rng, person=True, brightness=b), True) for _ in range(3)]
    seq += [(make_frame(rng, person=True, package=True, brightness=b), True)
            for _ in range(3)]
    seq += [(make_frame(rng, package=True, brightness=b), False)
            for _ in range(10)]
    reports = run(det, seq)
    assert len(reports) == 1
    assert iou(reports[0].bbox, PACKAGE) > 0.5


def test_zone_excludes_out_of_zone_package():
    rng = np.random.default_rng(RNG_SEED)
    zone = [(0.0, 0.0), (0.35, 0.0), (0.35, 1.0), (0.0, 1.0)]  # left third
    det = build_detector(gated_config(), zone=zone)  # PACKAGE sits right of it
    assert run(det, delivery_sequence(rng)) == []


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown detector mode"):
        build_detector(DetectorConfig(mode="telepathy"))
