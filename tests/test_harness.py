"""Unit tests for the harness itself: timestamps, evaluation, loader."""

from __future__ import annotations

import numpy as np
import pytest

from package_watcher.harness import (
    ClipResult, Detection, FixtureCase, evaluate, iter_samples, run_case)


def test_synthetic_timestamps_are_video_relative_and_deterministic():
    case = FixtureCase(name="t", expect="no_detect",
                       scene={"scene": "empty", "seconds": 5}, fps=2.0)
    a = list(iter_samples(case, "."))
    b = list(iter_samples(case, "."))
    assert [t for _, t in a] == [t for _, t in b]      # deterministic
    assert a[0][1] == 0.0 and a[1][1] == 0.5           # i / fps
    assert np.array_equal(a[3][0], b[3][0])            # frames match too


def test_detect_case_passes_on_package_scene():
    case = FixtureCase(name="pkg", expect="detect",
                       scene={"scene": "package", "warmup_s": 8, "hold_s": 14},
                       fps=2.0, detector={"persist_samples": 6},
                       region=(0.40, 0.58, 0.22, 0.22))
    outcome = evaluate(run_case(case, "."))
    assert outcome.passed, outcome.reason


def test_no_detect_case_passes_on_empty_scene():
    case = FixtureCase(name="empty", expect="no_detect",
                       scene={"scene": "empty", "seconds": 20}, fps=2.0)
    assert evaluate(run_case(case, ".")).passed


def _detect_result(case, dets):
    return ClipResult(case=case, detections=dets, samples=40, scene_resets=0)


def test_region_mismatch_fails_a_detect_case():
    case = FixtureCase(name="r", expect="detect", scene={"scene": "package"},
                       region=(0.0, 0.0, 0.1, 0.1))
    det = Detection(t=5.0, bbox_norm=(0.8, 0.8, 0.1, 0.1), confidence=0.9,
                    triggered=False, candidate_id=1)
    outcome = evaluate(_detect_result(case, [det]))
    assert not outcome.passed and "region" in outcome.reason


def test_time_window_enforced():
    case = FixtureCase(name="w", expect="detect", scene={"scene": "package"},
                       after=10.0)
    det = Detection(t=3.0, bbox_norm=(0.4, 0.6, 0.1, 0.1), confidence=0.9,
                    triggered=False, candidate_id=1)
    assert not evaluate(_detect_result(case, [det])).passed


def test_max_reports_enforced():
    case = FixtureCase(name="m", expect="detect", scene={"scene": "package"},
                       max_reports=1)
    d = lambda i: Detection(t=float(i), bbox_norm=(0.4, 0.6, 0.1, 0.1),
                            confidence=0.9, triggered=False, candidate_id=i)
    assert not evaluate(_detect_result(case, [d(1), d(2)])).passed


def test_no_detect_reports_offending_detection():
    case = FixtureCase(name="nd", expect="no_detect", scene={"scene": "empty"})
    det = Detection(t=7.0, bbox_norm=(0.5, 0.5, 0.1, 0.1), confidence=0.5,
                    triggered=False, candidate_id=1)
    outcome = evaluate(_detect_result(case, [det]))
    assert not outcome.passed and "7.0" in outcome.reason


def test_invalid_expect_rejected():
    with pytest.raises(ValueError, match="detect"):
        FixtureCase(name="x", expect="maybe", scene={"scene": "empty"})


def test_case_needs_a_source():
    with pytest.raises(ValueError, match="clip.*scene|scene"):
        FixtureCase(name="x", expect="detect")
