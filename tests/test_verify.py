"""Verifier decision layer + backtest engine, no model or NVR required.

The Florence model itself only runs on the real box (heavy download); these
tests cover everything around it: cropping, caption→verdict decisions, the
off/unknown backends, and the full backtest sampling loop with an injected
snapshot source and a stubbed caption function.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from package_watcher.backtest import run_backtest
from package_watcher.config import VerifierConfig
from package_watcher.verify import (FlorenceVerifier, build_verifier,
                                    crop_with_margin, decide)

from videogen import PKG, frame as make_frame


# --- decision layer ---------------------------------------------------------

def test_decide_accepts_package_words():
    v = decide("A brown cardboard box sitting on a doorstep.",
               ["package", "box", "parcel"])
    assert v["accepted"] is True and v["label"] == "box"


def test_decide_rejects_and_labels_known_other():
    v = decide("A shadow cast on a blue wall.", ["package", "box"])
    assert v["accepted"] is False and v["label"] == "shadow"
    v2 = decide("Some unidentifiable smudge.", ["package"])
    assert v2["accepted"] is False and v2["label"] == "other"


def test_decide_matches_whole_words_only():
    # 'cat' must not match inside 'scattered'; 'box' not inside 'boxer'.
    v = decide("White dots scattered on a wall.", ["package"])
    assert v["label"] == "other"
    v2 = decide("A boxer training in a gym.", ["box"])
    assert v2["accepted"] is False


def test_crop_with_margin_clamps_to_frame():
    f = np.zeros((100, 200, 3), dtype=np.uint8)
    crop = crop_with_margin(f, (180, 80, 40, 40), margin=0.5)
    assert crop.shape[0] > 0 and crop.shape[1] > 0
    assert crop.shape[0] <= 100 and crop.shape[1] <= 200


def test_build_verifier_off_and_unknown():
    assert build_verifier(VerifierConfig(backend="off")) is None
    with pytest.raises(ValueError, match="unknown verifier backend"):
        build_verifier(VerifierConfig(backend="magic"))


def test_florence_verify_with_stubbed_caption(monkeypatch):
    ver = FlorenceVerifier(VerifierConfig(backend="florence"))
    monkeypatch.setattr(
        ver, "_caption",
        lambda crop: "A cardboard package on the porch steps.")
    verdict = ver.verify(make_frame(np.random.default_rng(1), package=True),
                         PKG)
    assert verdict["accepted"] is True
    assert verdict["label"] == "package"
    assert verdict["backend"] == "florence"


# --- backtest engine ---------------------------------------------------------

UTC = timezone.utc
DAY = datetime(2026, 7, 5, 0, 0, tzinfo=UTC)


def _timeline(events):
    """Build a snapshot_fn from {sample_index: kwargs-for-frame}; the scene
    at index i inherits the most recent event at or before i."""
    rng = np.random.default_rng(3)

    def fn(at: datetime):
        i = int((at - DAY).total_seconds() // 600)
        state = {}
        for k in sorted(events):
            if k <= i:
                state = events[k]
        if state.get("missing"):
            return None
        return make_frame(rng, **{k: v for k, v in state.items()
                                  if k in ("package", "person")})
    return fn


def _run(events, windows=(), samples=12, verifier=None):
    return run_backtest(
        _timeline(events), "cam", DAY,
        DAY + timedelta(seconds=600 * (samples - 1)),
        interval_s=600, person_windows=list(windows), verifier=verifier,
        person_pad_s=0.0)


def test_package_arrival_is_hit_once_at_right_spot():
    # Package appears at sample 6 and stays; exactly one hit, then absorbed.
    res = _run({0: {}, 6: {"package": True}})
    assert len(res.hits) == 1
    hit = res.hits[0]
    assert hit.at == DAY + timedelta(seconds=600 * 6)
    x, y, w, h = PKG
    hx, hy, hw, hh = hit.bbox
    assert abs(hx - x) < 12 and abs(hy - y) < 12


def test_person_windows_are_skipped():
    # Person present during samples 4-5 (2400-3000s); package appears with
    # them and remains. The hit lands on the first clean sample after.
    res = _run({0: {}, 4: {"person": True, "package": True},
                6: {"package": True}},
               windows=[(2400.0, 3300.0)])
    assert res.samples_skipped_person == 2
    assert len(res.hits) == 1
    assert res.hits[0].at == DAY + timedelta(seconds=600 * 6)


def test_missing_snapshots_counted_not_fatal():
    res = _run({0: {}, 3: {"missing": True}, 4: {}})
    assert res.samples_missing == 1
    assert res.hits == []


def test_empty_day_has_no_hits():
    assert _run({0: {}}).hits == []


def test_verifier_attached_to_hits(monkeypatch):
    ver = FlorenceVerifier(VerifierConfig(backend="florence"))
    monkeypatch.setattr(ver, "_caption", lambda crop: "a cardboard box")
    res = _run({0: {}, 6: {"package": True}}, verifier=ver)
    assert len(res.hits) == 1
    assert res.hits[0].verification["accepted"] is True
