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
    v = decide("White dots scattered on the ground.", ["package"])
    assert v["label"] == "other"          # NOT 'cat' (inside 'scattered')
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
    ver = FlorenceVerifier(VerifierConfig(backend="florence", refine="off"))
    monkeypatch.setattr(
        ver, "_caption",
        lambda crop: "A cardboard package on the porch steps.")
    verdict = ver.verify(make_frame(np.random.default_rng(1), package=True),
                         PKG)
    assert verdict["accepted"] is True
    assert verdict["label"] == "package"
    assert verdict["backend"] == "florence"
    assert verdict["refined"] is False


def test_refined_crop_used_when_segmenter_returns_bbox(monkeypatch):
    ver = FlorenceVerifier(VerifierConfig(backend="florence"))  # refine=sam2
    seen = {}
    monkeypatch.setattr(ver, "_mask_bbox",
                        lambda frame, bbox: (150, 160, 30, 20))
    def cap(crop):
        seen["shape"] = crop.shape
        return "a cardboard box"
    monkeypatch.setattr(ver, "_caption", cap)
    frame = make_frame(np.random.default_rng(1), package=True)
    verdict = ver.verify(frame, PKG)
    assert verdict["refined"] is True
    # crop follows the refined bbox (30x20 + 20% margin), not the loose one
    assert seen["shape"][0] < 40 and seen["shape"][1] < 50


def test_refinement_failure_falls_back_to_plain_crop(monkeypatch):
    ver = FlorenceVerifier(VerifierConfig(backend="florence"))
    monkeypatch.setattr(ver, "_mask_bbox", lambda frame, bbox: None)
    monkeypatch.setattr(ver, "_caption", lambda crop: "a cardboard box")
    verdict = ver.verify(make_frame(np.random.default_rng(1), package=True),
                         PKG)
    assert verdict["accepted"] is True and verdict["refined"] is False


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


def test_package_arrival_confirms_after_persisting():
    # Package appears at sample 6 and stays. It is seen at 6 (reference
    # held), CONFIRMS at 7 (still in the same spot), then is absorbed.
    res = _run({0: {}, 6: {"package": True}})
    assert len(res.hits) == 1
    hit = res.hits[0]
    assert hit.at == DAY + timedelta(seconds=600 * 7)
    x, y, w, h = PKG
    hx, hy, hw, hh = hit.bbox
    assert abs(hx - x) < 12 and abs(hy - y) < 12
    # The comparison pair rides along. The reference rolls every sample
    # (lighting tracks) but the candidate's region keeps its PRE-ARRIVAL
    # pixels, so the baseline shows the empty spot where the box now is.
    assert hit.baseline_at == DAY + timedelta(seconds=600 * 6)
    assert hit.baseline is not None and hit.baseline.shape == hit.frame.shape
    x, y, w, h = PKG
    region = hit.baseline[y:y + h, x:x + w]
    assert abs(float(region.mean()) - 120) < 15   # pre-arrival background


def test_person_windows_are_skipped():
    # Person present during samples 4-5 (2400-3000s); package appears with
    # them and remains. Seen at 6, confirmed at 7, baseline = last clean
    # sample before the visit (sample 3).
    res = _run({0: {}, 4: {"person": True, "package": True},
                6: {"package": True}},
               windows=[(2400.0, 3300.0)])
    assert res.samples_skipped_person == 2
    assert len(res.hits) == 1
    assert res.hits[0].at == DAY + timedelta(seconds=600 * 7)
    # Baseline label = previous processed sample; the package's region in it
    # still shows the pre-visit scene (masked absorption).
    assert res.hits[0].baseline_at == DAY + timedelta(seconds=600 * 6)
    x, y, w, h = PKG
    region = res.hits[0].baseline[y:y + h, x:x + w]
    assert abs(float(region.mean()) - 120) < 15


def test_moving_shadow_never_confirms():
    # A shadow patch that creeps across the frame overlaps its own previous
    # position (IoU-wise) but its PIXELS shift every sample — the stillness
    # check must keep it from ever confirming. (Known limitation: a shadow
    # sliding off the frame edge can read as still for a couple of samples;
    # the vision verdict is the net for that.)
    rng = np.random.default_rng(4)

    def fn(at):
        i = int((at - DAY).total_seconds() // 600)
        f = make_frame(rng)
        x = 30 + i * 30                       # moves 30 px per sample
        f[150:190, x:x + 50] = (60, 60, 60)   # 50x40 dark patch
        return f

    res = run_backtest(fn, "cam", DAY, DAY + timedelta(seconds=600 * 8),
                       interval_s=600)
    assert res.hits == []


def test_package_still_confirms_despite_nearby_shadow_motion():
    # The stillness gate must not throw away real arrivals: package lands at
    # sample 5 while a shadow creeps elsewhere in the frame.
    rng = np.random.default_rng(9)

    def fn(at):
        i = int((at - DAY).total_seconds() // 600)
        f = make_frame(rng, package=(i >= 5))
        x = 20 + i * 25
        f[30:60, x:x + 40] = (70, 70, 70)     # unrelated creeping shadow, top
        return f

    res = run_backtest(fn, "cam", DAY, DAY + timedelta(seconds=600 * 9),
                       interval_s=600)
    pkg_hits = [h for h in res.hits
                if abs(h.bbox[0] - PKG[0]) < 15 and abs(h.bbox[1] - PKG[1]) < 15]
    assert len(pkg_hits) == 1


def test_missing_snapshots_counted_not_fatal():
    res = _run({0: {}, 3: {"missing": True}, 4: {}})
    assert res.samples_missing == 1
    assert res.hits == []


def test_empty_day_has_no_hits():
    assert _run({0: {}}).hits == []


def test_same_sample_hits_grouped_into_one_card():
    """Three blobs from one comparison = ONE sample card with 3 numbered
    boxes, not three same-timestamp events."""
    pytest.importorskip("flask")
    from package_watcher.backtest import Hit
    from package_watcher.ui.app import _group_hits

    f = make_frame(np.random.default_rng(1))
    t = datetime(2026, 7, 6, 11, 30, tzinfo=UTC)
    t2 = t + timedelta(seconds=600)

    before = make_frame(np.random.default_rng(2))
    t0 = t - timedelta(seconds=600)

    def hit(at, bbox):
        return Hit(at=at, bbox=bbox, bbox_norm=(0.1, 0.1, 0.1, 0.1),
                   confidence=0.8, frame=f, crop=f,
                   baseline=before, baseline_at=t0)

    groups = _group_hits([hit(t, (10, 10, 30, 30)),
                          hit(t, (100, 50, 40, 30)),
                          hit(t, (200, 150, 46, 32)),
                          hit(t2, (10, 10, 30, 30))])
    assert len(groups) == 2
    by_at = {g["at"]: g for g in groups}
    assert len(by_at[t.isoformat()]["boxes"]) == 3
    assert len(by_at[t2.isoformat()]["boxes"]) == 1
    assert by_at[t.isoformat()]["jpg"].startswith("data:image/jpeg")
    # The before/after comparison pair is exposed on the group.
    assert by_at[t.isoformat()]["before_jpg"].startswith("data:image/jpeg")
    assert by_at[t.isoformat()]["before_at"] == t0.isoformat()


def test_verifier_attached_to_hits(monkeypatch):
    ver = FlorenceVerifier(VerifierConfig(backend="florence", refine="off"))
    monkeypatch.setattr(ver, "_caption", lambda crop: "a cardboard box")
    res = _run({0: {}, 6: {"package": True}}, verifier=ver)
    assert len(res.hits) == 1
    assert res.hits[0].verification["accepted"] is True


def test_verifier_status_endpoint(tmp_path, monkeypatch):
    pytest.importorskip("flask")
    from package_watcher.ui.app import create_app

    (tmp_path / "cases.yaml").write_text("cases: []\n")
    off = create_app(str(tmp_path)).test_client().get("/api/verifier").get_json()
    assert off["configured"] is False

    on = create_app(str(tmp_path),
                    verifier_cfg=VerifierConfig(backend="florence"))
    body = on.test_client().get("/api/verifier").get_json()
    assert body["configured"] is True and body["loaded"] is False
    assert body["backend"] == "florence"
