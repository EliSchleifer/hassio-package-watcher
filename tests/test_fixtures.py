"""Integration tests: run every fixture case through the detector and grade.

These are data-driven from fixtures/cases.yaml. Synthetic cases run anywhere;
real-clip cases are skipped if their video file isn't present, so you can
commit fixtures that reference clips you keep locally without breaking CI.
"""

from __future__ import annotations

import os

import pytest

from package_watcher.harness import (FixtureCase, load_cases, run_and_evaluate)

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures")
MANIFEST = os.path.join(FIXTURES_DIR, "cases.yaml")

CASES = load_cases(MANIFEST)


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_fixture_case(case: FixtureCase):
    if case.clip:
        path = (case.clip if os.path.isabs(case.clip)
                else os.path.join(FIXTURES_DIR, case.clip))
        if not os.path.isfile(path):
            pytest.skip(f"clip not present: {case.clip}")
    outcome = run_and_evaluate(case, FIXTURES_DIR)
    assert outcome.passed, f"[{case.name}] {outcome.reason}"


def test_manifest_has_both_polarities():
    """Guard against a manifest that only ever asserts one direction."""
    kinds = {c.expect for c in CASES}
    assert "detect" in kinds and "no_detect" in kinds


def test_loader_rejects_unknown_detector_key(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "cases:\n  - name: x\n    scene: {scene: empty}\n    expect: detect\n"
        "    detector: {not_a_real_knob: 3}\n")
    with pytest.raises(ValueError, match="not_a_real_knob"):
        load_cases(str(bad))
