"""Unit tests for the harness itself: timestamps, evaluation, loader.

Pipeline tests run on tiny generated video files (see videogen.py); the
evaluation-logic tests construct results directly and never touch video.
"""

from __future__ import annotations

import numpy as np
import pytest

from package_watcher.harness import (
    ClipResult, Detection, FixtureCase, evaluate, iter_samples, run_case)

from videogen import PKG_REGION, delivery_clip, empty_clip, package_clip


def test_clip_timestamps_are_video_relative_and_deterministic(tmp_path):
    empty_clip(tmp_path / "e.mp4", seconds=5)
    case = FixtureCase(name="t", expect="no_detect", clip="e.mp4", fps=2.0)
    a = list(iter_samples(case, str(tmp_path)))
    b = list(iter_samples(case, str(tmp_path)))
    assert [t for _, t in a] == [t for _, t in b]      # deterministic
    assert a[0][1] == 0.0 and a[1][1] == 0.5           # i / fps
    assert np.array_equal(a[3][0], b[3][0])            # frames match too


def test_detect_case_passes_on_package_clip(tmp_path):
    package_clip(tmp_path / "pkg.mp4")
    case = FixtureCase(name="pkg", expect="detect", clip="pkg.mp4",
                       fps=2.0, detector={"persist_samples": 6},
                       region=PKG_REGION)
    outcome = evaluate(run_case(case, str(tmp_path)))
    assert outcome.passed, outcome.reason


def test_no_detect_case_passes_on_empty_clip(tmp_path):
    empty_clip(tmp_path / "e.mp4")
    case = FixtureCase(name="empty", expect="no_detect", clip="e.mp4", fps=2.0)
    assert evaluate(run_case(case, str(tmp_path))).passed


def test_person_gated_case_on_delivery_clip(tmp_path):
    delivery_clip(tmp_path / "d.mp4", warmup_s=8, visit_s=6, tail_s=12)
    case = FixtureCase(name="d", expect="detect", clip="d.mp4", fps=2.0,
                       detector={"mode": "person_gated", "settle_samples": 3},
                       presence=[(8.0, 14.0)], region=PKG_REGION)
    outcome = evaluate(run_case(case, str(tmp_path)))
    assert outcome.passed, outcome.reason


def _detect_result(case, dets):
    return ClipResult(case=case, detections=dets, samples=40, scene_resets=0)


def test_region_mismatch_fails_a_detect_case():
    case = FixtureCase(name="r", expect="detect", clip="x.mp4",
                       region=(0.0, 0.0, 0.1, 0.1))
    det = Detection(t=5.0, bbox_norm=(0.8, 0.8, 0.1, 0.1), confidence=0.9,
                    triggered=False, candidate_id=1)
    outcome = evaluate(_detect_result(case, [det]))
    assert not outcome.passed and "region" in outcome.reason


def test_time_window_enforced():
    case = FixtureCase(name="w", expect="detect", clip="x.mp4", after=10.0)
    det = Detection(t=3.0, bbox_norm=(0.4, 0.6, 0.1, 0.1), confidence=0.9,
                    triggered=False, candidate_id=1)
    assert not evaluate(_detect_result(case, [det])).passed


def test_max_reports_enforced():
    case = FixtureCase(name="m", expect="detect", clip="x.mp4", max_reports=1)
    d = lambda i: Detection(t=float(i), bbox_norm=(0.4, 0.6, 0.1, 0.1),
                            confidence=0.9, triggered=False, candidate_id=i)
    assert not evaluate(_detect_result(case, [d(1), d(2)])).passed


def test_no_detect_reports_offending_detection():
    case = FixtureCase(name="nd", expect="no_detect", clip="x.mp4")
    det = Detection(t=7.0, bbox_norm=(0.5, 0.5, 0.1, 0.1), confidence=0.5,
                    triggered=False, candidate_id=1)
    outcome = evaluate(_detect_result(case, [det]))
    assert not outcome.passed and "7.0" in outcome.reason


def test_invalid_expect_rejected():
    with pytest.raises(ValueError, match="detect"):
        FixtureCase(name="x", expect="maybe", clip="x.mp4")


def test_case_needs_a_clip():
    with pytest.raises(ValueError, match="needs a clip"):
        FixtureCase(name="x", expect="detect")
