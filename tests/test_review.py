"""Person-event replay engine + labeling endpoints, no NVR required."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from package_watcher.review import merge_events, review_person_events

from videogen import PKG, frame as make_frame

UTC = timezone.utc
T0 = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)


def _t(s: float) -> datetime:
    return T0 + timedelta(seconds=s)


def _snapshots(package_after_s: float | None):
    """Scene is empty until package_after_s (None = never)."""
    rng = np.random.default_rng(11)

    def fn(at: datetime):
        rel = (at - T0).total_seconds()
        has_pkg = package_after_s is not None and rel >= package_after_s
        return make_frame(rng, package=has_pkg)
    return fn


def test_merge_events_joins_close_visits():
    ev = [(_t(0), _t(30)), (_t(50), _t(80)), (_t(500), _t(520))]
    merged = merge_events(ev, gap_s=60)
    assert merged == [(_t(0), _t(80)), (_t(500), _t(520))]


def test_delivery_visit_produces_candidate():
    # Visit 100..160s; the package is on the ground from 130s (mid-visit).
    reviews = review_person_events(
        _snapshots(package_after_s=130), [(_t(100), _t(160))],
        before_margin_s=20, settle_s=45)
    assert len(reviews) == 1
    rv = reviews[0]
    assert rv.error is None
    assert rv.before_at == _t(80) and rv.after_at == _t(205)
    assert len(rv.candidates) == 1
    x, y, w, h = PKG
    assert abs(rv.candidates[0].bbox[0] - x) < 12


def test_empty_visit_produces_no_candidates():
    reviews = review_person_events(
        _snapshots(package_after_s=None), [(_t(100), _t(160))])
    assert reviews[0].candidates == []


def test_after_clamped_before_next_visit():
    # Two visits 60s apart with settle 45 — the first 'after' must not land
    # inside the second visit.
    reviews = review_person_events(
        _snapshots(package_after_s=None),
        [(_t(0), _t(30)), (_t(200), _t(230))],
        before_margin_s=10, settle_s=45)
    assert reviews[0].after_at <= _t(195)


def test_missing_footage_is_reported_not_fatal():
    def fn(at):
        return None
    reviews = review_person_events(fn, [(_t(0), _t(30))])
    assert reviews[0].error and reviews[0].candidates == []


def test_verifier_verdicts_attached(monkeypatch):
    from package_watcher.config import VerifierConfig
    from package_watcher.verify import FlorenceVerifier

    ver = FlorenceVerifier(VerifierConfig(backend="florence", refine="off"))
    monkeypatch.setattr(ver, "_caption", lambda crop: "a cardboard box")
    reviews = review_person_events(
        _snapshots(package_after_s=130), [(_t(100), _t(160))], verifier=ver)
    assert reviews[0].candidates[0].verification["accepted"] is True


def test_label_endpoint_appends_jsonl(tmp_path):
    pytest.importorskip("flask")
    from package_watcher.ui.app import create_app

    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "cases.yaml").write_text("cases: []\n")
    c = create_app(str(fixtures)).test_client()

    r = c.post("/api/label", json={
        "camera_id": "cam1", "event_start": "2026-07-06T12:00:00+00:00",
        "event_end": "2026-07-06T12:01:00+00:00",
        "before_path": "training/images/x-before.jpg",
        "after_path": "training/images/x-after.jpg",
        "label": "package", "bbox_norm": [0.4, 0.8, 0.1, 0.1],
        "caption": "a cardboard box"})
    assert r.status_code == 200 and r.get_json()["total_labels"] == 1
    r2 = c.post("/api/label", json={"label": "none"})
    assert r2.get_json()["total_labels"] == 2
    bad = c.post("/api/label", json={"label": "maybe"})
    assert bad.status_code == 400

    lines = (fixtures / "training" / "labels.jsonl").read_text().splitlines()
    rec = json.loads(lines[0])
    assert rec["label"] == "package" and rec["bbox_norm"] == [0.4, 0.8, 0.1, 0.1]
