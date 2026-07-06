"""Detector tests on synthetic footage — no cameras or network needed.

We render a noisy fixed 'porch' scene, then drop a package-sized rectangle
into it and assert the detector reports it once, at the right coordinates,
without false alarms from noise, lighting drift, or moving objects.
"""

from __future__ import annotations

import numpy as np
import pytest

from package_watcher.detector import DetectorConfig, StaticObjectDetector

RNG_SEED = 1234
W, H = 320, 240
PACKAGE = (140, 150, 46, 32)  # x, y, w, h ground truth


def make_frame(rng: np.random.Generator, brightness: float = 0.0,
               package: tuple[int, int, int, int] | None = None,
               mover_x: int | None = None) -> np.ndarray:
    """A gray porch with sensor noise, optional package, optional passerby."""
    frame = np.full((H, W, 3), 120, dtype=np.float32)
    frame[: H // 3] = 160          # "sky" band so the scene isn't uniform
    frame += brightness
    frame += rng.normal(0, 3, size=frame.shape).astype(np.float32)
    if package is not None:
        x, y, w, h = package
        frame[y:y + h, x:x + w] = (55, 70, 95)  # cardboard-ish dark box
    if mover_x is not None:
        frame[100:180, mover_x:mover_x + 24] = (30, 30, 30)  # walking person
    return np.clip(frame, 0, 255).astype(np.uint8)


def fast_config(**overrides) -> DetectorConfig:
    base = dict(resize_width=W, persist_samples=5, persist_samples_triggered=3,
                slow_alpha=0.002, fast_alpha=0.2)
    base.update(overrides)
    return DetectorConfig(**base)


def run_sequence(detector: StaticObjectDetector, frames, t0: float = 1000.0,
                 attention: bool = False):
    reports = []
    for i, frame in enumerate(frames):
        reports.extend(detector.process(frame, ts=t0 + i, attention=attention))
    return reports


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / float(aw * ah + bw * bh - inter) if inter else 0.0


def test_detects_package_at_correct_coordinates():
    rng = np.random.default_rng(RNG_SEED)
    detector = StaticObjectDetector(fast_config())
    frames = [make_frame(rng) for _ in range(30)]
    frames += [make_frame(rng, package=PACKAGE) for _ in range(20)]

    reports = run_sequence(detector, frames)

    assert len(reports) == 1, f"expected exactly one report, got {len(reports)}"
    report = reports[0]
    assert iou(report.bbox, PACKAGE) > 0.5
    assert report.frame_size == (W, H)
    assert report.samples_persisted >= 5
    assert 0.0 < report.confidence <= 1.0
    # Evidence arrays are populated and usable.
    assert report.frame.shape == (H, W, 3)
    assert report.mask.max() == 255
    assert report.baseline.shape[:2] == report.mask.shape[:2]
    # Baseline predates the package: the package region should look like
    # background there, not like the box.
    x, y, w, h = PACKAGE
    assert abs(float(report.baseline[y:y + h, x:x + w].mean()) - 120) < 15


def test_reports_only_once_while_package_remains():
    rng = np.random.default_rng(RNG_SEED)
    detector = StaticObjectDetector(fast_config())
    frames = [make_frame(rng) for _ in range(30)]
    frames += [make_frame(rng, package=PACKAGE) for _ in range(60)]
    reports = run_sequence(detector, frames)
    assert len(reports) == 1


def test_no_false_alarm_on_noise_only():
    rng = np.random.default_rng(RNG_SEED)
    detector = StaticObjectDetector(fast_config())
    reports = run_sequence(detector, [make_frame(rng) for _ in range(80)])
    assert reports == []


def test_no_false_alarm_on_gradual_lighting_drift():
    rng = np.random.default_rng(RNG_SEED)
    detector = StaticObjectDetector(fast_config())
    frames = [make_frame(rng, brightness=i * 0.5) for i in range(80)]
    reports = run_sequence(detector, frames)
    assert reports == []


def test_sudden_lighting_change_triggers_scene_reset_not_report():
    rng = np.random.default_rng(RNG_SEED)
    detector = StaticObjectDetector(fast_config())
    frames = [make_frame(rng) for _ in range(20)]
    frames += [make_frame(rng, brightness=80.0) for _ in range(30)]
    reports = run_sequence(detector, frames)
    assert reports == []
    assert detector.scene_resets >= 1


def test_moving_object_is_not_reported():
    rng = np.random.default_rng(RNG_SEED)
    detector = StaticObjectDetector(fast_config())
    frames = [make_frame(rng) for _ in range(20)]
    frames += [make_frame(rng, mover_x=10 + i * 12) for i in range(24)]
    frames += [make_frame(rng) for _ in range(20)]
    reports = run_sequence(detector, frames)
    assert reports == []


def test_attention_lowers_persistence_bar():
    rng = np.random.default_rng(RNG_SEED)

    def first_report_delay(attention: bool) -> int:
        detector = StaticObjectDetector(fast_config())
        run_sequence(detector, [make_frame(rng) for _ in range(30)])
        for i in range(30):
            reports = detector.process(
                make_frame(rng, package=PACKAGE), ts=2000.0 + i,
                attention=attention)
            if reports:
                assert reports[0].triggered is attention
                return i
        pytest.fail("package never reported")

    assert first_report_delay(True) < first_report_delay(False)


def test_zone_masks_out_detections_outside_polygon():
    rng = np.random.default_rng(RNG_SEED)
    # Zone covers only the left half; package sits on the right.
    zone = [(0.0, 0.0), (0.4, 0.0), (0.4, 1.0), (0.0, 1.0)]
    detector = StaticObjectDetector(fast_config(), zone=zone)
    frames = [make_frame(rng) for _ in range(30)]
    frames += [make_frame(rng, package=PACKAGE) for _ in range(30)]
    assert run_sequence(detector, frames) == []


def test_rearms_after_healing_reported_object():
    rng = np.random.default_rng(RNG_SEED)
    detector = StaticObjectDetector(fast_config(heal_after_reported=5))
    second = (40, 60, 40, 30)
    frames = [make_frame(rng) for _ in range(30)]
    frames += [make_frame(rng, package=PACKAGE) for _ in range(30)]
    reports = run_sequence(detector, frames)
    assert len(reports) == 1
    # First package healed into background; a second box elsewhere while the
    # first remains must produce a fresh report.
    def two_boxes(f):
        x, y, w, h = second
        f = f.copy()
        f[y:y + h, x:x + w] = (200, 40, 40)
        return f
    more = [two_boxes(make_frame(rng, package=PACKAGE)) for _ in range(20)]
    reports2 = run_sequence(detector, more, t0=5000.0)
    assert len(reports2) == 1
    assert iou(reports2[0].bbox, second) > 0.5


def test_bbox_scales_back_to_native_resolution():
    rng = np.random.default_rng(RNG_SEED)
    detector = StaticObjectDetector(fast_config(resize_width=160))

    def upscale(f):
        return np.repeat(np.repeat(f, 2, axis=0), 2, axis=1)  # 640x480 native

    frames = [upscale(make_frame(rng)) for _ in range(30)]
    frames += [upscale(make_frame(rng, package=PACKAGE)) for _ in range(20)]
    reports = run_sequence(detector, frames)
    assert len(reports) == 1
    native_truth = (PACKAGE[0] * 2, PACKAGE[1] * 2, PACKAGE[2] * 2, PACKAGE[3] * 2)
    assert reports[0].frame_size == (W * 2, H * 2)
    assert iou(reports[0].bbox, native_truth) > 0.5
