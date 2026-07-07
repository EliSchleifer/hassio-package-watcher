"""Integration tests: run every fixture case through the detector and grade.

Data-driven from fixtures/cases.yaml. Cases reference REAL clips which stay
local (fixtures/clips/ is gitignored), so a case whose clip isn't present on
this machine is skipped, not failed — the manifest is shareable without
sharing the footage.
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
    path = (case.clip if os.path.isabs(case.clip)
            else os.path.join(FIXTURES_DIR, case.clip))
    if not os.path.isfile(path):
        pytest.skip(f"clip not present: {case.clip}")
    outcome = run_and_evaluate(case, FIXTURES_DIR)
    assert outcome.passed, f"[{case.name}] {outcome.reason}"


def test_loader_rejects_unknown_detector_key(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "cases:\n  - name: x\n    clip: clips/x.mp4\n    expect: detect\n"
        "    detector: {not_a_real_knob: 3}\n")
    with pytest.raises(ValueError, match="not_a_real_knob"):
        load_cases(str(bad))
