"""Smoke tests for the authoring UI via Flask's test client (no browser).

Covers the endpoints the UI relies on: listing cases, running them, the PNG
preview, saving a case (append + preserving other content), and graceful
degradation when Protect isn't configured.
"""

from __future__ import annotations

import io
import os

import pytest

flask = pytest.importorskip("flask")

from package_watcher.ui.app import create_app  # noqa: E402

from videogen import (PKG_REGION, delivery_clip, empty_clip,  # noqa: E402
                      package_clip)

MANIFEST = """cases:
  - name: seed-empty
    clip: clips/seed-empty.mp4
    fps: 2.0
    expect: no_detect
"""


@pytest.fixture()
def client(tmp_path):
    fixtures = tmp_path / "fixtures"
    (fixtures / "clips").mkdir(parents=True)
    empty_clip(fixtures / "clips" / "seed-empty.mp4", seconds=10)
    (fixtures / "cases.yaml").write_text(MANIFEST)
    app = create_app(str(fixtures), unifi=None)
    app.config.update(TESTING=True)
    return app.test_client(), fixtures


def test_index_serves(client):
    c, _ = client
    r = c.get("/")
    assert r.status_code == 200 and b"package-watcher" in r.data


def test_list_and_run_cases(client):
    c, _ = client
    cases = c.get("/api/cases").get_json()
    assert [x["name"] for x in cases] == ["seed-empty"]

    results = c.post("/api/run").get_json()
    assert results[0]["name"] == "seed-empty"
    assert results[0]["status"] == "pass"


def test_preview_returns_png(client):
    c, _ = client
    r = c.get("/api/preview/seed-empty.png?kind=first")
    assert r.status_code == 200
    assert r.data[:8] == b"\x89PNG\r\n\x1a\n"


def test_cameras_unavailable_without_unifi(client, monkeypatch):
    monkeypatch.delenv("SUPERVISOR_TOKEN", raising=False)
    monkeypatch.delenv("HASSIO_TOKEN", raising=False)
    c, _ = client
    body = c.get("/api/cameras").get_json()
    assert body["available"] is False


def test_hass_list_cameras_filters_and_sorts(monkeypatch):
    from package_watcher.ui import hass

    fake_states = [
        {"entity_id": "sensor.temperature", "state": "21"},
        {"entity_id": "camera.front_door",
         "attributes": {"friendly_name": "Front Door"}, "state": "idle"},
        {"entity_id": "camera.driveway", "attributes": {}, "state": "recording"},
    ]
    monkeypatch.setattr(hass, "_get", lambda path, **kw: fake_states)
    cams = hass.list_cameras()
    assert [c["id"] for c in cams] == ["camera.driveway", "camera.front_door"]
    # falls back to entity_id when friendly_name is absent
    assert cams[0]["name"] == "camera.driveway"
    assert cams[1]["name"] == "Front Door"


def test_cameras_from_home_assistant(client, monkeypatch):
    from package_watcher.ui import hass

    monkeypatch.setenv("SUPERVISOR_TOKEN", "test-token")
    monkeypatch.setattr(hass, "list_cameras",
                        lambda: [{"id": "camera.porch", "name": "Porch",
                                  "state": "idle"}])
    c, _ = client
    body = c.get("/api/cameras").get_json()
    assert body["available"] is True
    assert body["source"] == "homeassistant"
    assert body["supports_pull"] is False
    assert body["cameras"][0]["name"] == "Porch"


def _write_config_entries(ha_dir, entries):
    import json
    storage = ha_dir / ".storage"
    storage.mkdir(parents=True)
    (storage / "core.config_entries").write_text(
        json.dumps({"version": 1, "data": {"entries": entries}}))


def test_discover_unifi_protect_from_storage(tmp_path, monkeypatch):
    from package_watcher.ui import hass

    _write_config_entries(tmp_path, [
        {"domain": "sun", "data": {}},
        {"domain": "unifiprotect", "data": {
            "host": "10.0.0.5", "port": 443, "username": "u", "password": "p",
            "api_key": "k", "verify_ssl": False}},
    ])
    monkeypatch.setenv("PACKAGE_WATCHER_HA_CONFIG", str(tmp_path))
    cfg = hass.discover_unifi_protect()
    assert cfg is not None
    assert (cfg.host, cfg.port, cfg.username, cfg.api_key) == \
        ("10.0.0.5", 443, "u", "k")


def test_discover_unifi_protect_absent(tmp_path, monkeypatch):
    from package_watcher.ui import hass

    _write_config_entries(tmp_path, [{"domain": "sun", "data": {}}])
    monkeypatch.setenv("PACKAGE_WATCHER_HA_CONFIG", str(tmp_path))
    assert hass.discover_unifi_protect() is None


def test_page_js_parses(tmp_path):
    """A syntax error in the embedded JS breaks the whole page silently
    (buttons do nothing). If node is available, syntax-check the script."""
    import shutil, subprocess, re
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available to syntax-check page JS")
    from package_watcher.ui.app import create_app
    (tmp_path / "cases.yaml").write_text("cases: []\n")
    html = create_app(str(tmp_path)).test_client().get("/").get_data(as_text=True)
    js = re.search(r"<script>(.*?)</script>", html, re.S).group(1)
    (tmp_path / "page.js").write_text(js)
    r = subprocess.run([node, "--check", str(tmp_path / "page.js")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_load_config_unifi_only():
    import tempfile, os
    from package_watcher.config import load_config
    d = tempfile.mkdtemp()
    p = os.path.join(d, "c.yaml")
    with open(p, "w") as f:
        f.write("unifi:\n  host: 10.0.0.5\n  api_key: k\n")
    cfg = load_config(p, require_cameras=False)
    assert cfg.unifi.host == "10.0.0.5" and cfg.cameras == []


def test_livereload_only_in_reload_mode(tmp_path):
    from package_watcher.ui.app import create_app
    (tmp_path / "cases.yaml").write_text("cases: []\n")
    off = create_app(str(tmp_path)).test_client().get("/").get_data(as_text=True)
    assert "fetch('__alive')" not in off
    app = create_app(str(tmp_path), reload=True)
    on = app.test_client().get("/").get_data(as_text=True)
    assert "fetch('__alive')" in on
    assert app.test_client().get("/__alive").status_code == 200


def test_snapshot_returns_jpeg(client, monkeypatch):
    from package_watcher.ui import hass, protect
    from package_watcher.config import UnifiConfig

    monkeypatch.setattr(hass, "discover_unifi_protect",
                        lambda: UnifiConfig(host="10.0.0.5", api_key="k"))
    monkeypatch.setattr(protect, "available", lambda: True)
    captured = {}

    def fake_snap(cfg, cam, dt, width=640):
        captured["cam"] = cam
        captured["dt"] = dt
        captured["width"] = width
        return b"\xff\xd8jpegbytes"

    monkeypatch.setattr(protect, "snapshot_at", fake_snap)
    c, _ = client
    r = c.get("/api/snapshot?camera_id=abc&at=2026-07-05T15:30:00%2B00:00&width=480")
    assert r.status_code == 200
    assert r.mimetype == "image/jpeg"
    assert r.data == b"\xff\xd8jpegbytes"
    assert captured["cam"] == "abc" and captured["width"] == 480
    assert captured["dt"].year == 2026


def test_snapshot_without_at_means_live(client, monkeypatch):
    from package_watcher.ui import hass, protect
    from package_watcher.config import UnifiConfig

    monkeypatch.setattr(hass, "discover_unifi_protect",
                        lambda: UnifiConfig(host="10.0.0.5", api_key="k"))
    monkeypatch.setattr(protect, "available", lambda: True)
    seen = {}

    def fake_snap(cfg, cam, dt, width=640):
        seen["dt"] = dt
        return b"\xff\xd8live"

    monkeypatch.setattr(protect, "snapshot_at", fake_snap)
    c, _ = client
    r = c.get("/api/snapshot?camera_id=abc")
    assert r.status_code == 200 and seen["dt"] is None


def test_zone_roundtrip_and_config_pickup(client, tmp_path, monkeypatch):
    """Zone saved in the UI lands in zones.yaml, is returned by GET, keyed
    by id AND name, and load_config picks it up from the config space."""
    c, fixtures = client
    r = c.post("/api/zone", json={"camera_id": "cam123",
                                  "camera_name": "Front Door",
                                  "rect": [0.2, 0.5, 0.4, 0.3]})
    assert r.status_code == 200
    poly = r.get_json()["zone"]
    assert poly[0] == [0.2, 0.5] and poly[2] == [0.6000000000000001, 0.8]

    got = c.get("/api/zone?camera_id=cam123").get_json()
    assert got["zone"] == poly

    # zones.yaml sits in the fixtures dir here (no config passed) and is
    # keyed by both id and display name.
    import yaml as _yaml
    zones = _yaml.safe_load((fixtures / "zones.yaml").read_text())
    assert "cam123" in zones and "Front Door" in zones

    # A config file next to a zones.yaml picks the zones up.
    (tmp_path / "config.yaml").write_text(
        "cameras: [{name: front, source: x}]\n")
    (tmp_path / "zones.yaml").write_text(
        _yaml.safe_dump({"Front Door": poly}))
    from package_watcher.config import load_config
    cfg = load_config(str(tmp_path / "config.yaml"))
    assert cfg.zones["Front Door"] == poly

    # Clearing removes both keys.
    c.post("/api/zone", json={"camera_id": "cam123",
                              "camera_name": "Front Door", "rect": None})
    assert c.get("/api/zone?camera_id=cam123").get_json()["zone"] is None


def test_zone_accepts_full_polygon(client):
    c, _ = client
    tri = [[0.1, 0.9], [0.5, 0.4], [0.9, 0.9]]
    r = c.post("/api/zone", json={"camera_id": "camT", "poly": tri})
    assert r.status_code == 200
    assert c.get("/api/zone?camera_id=camT").get_json()["zone"] == tri
    # fewer than 3 points is rejected
    r2 = c.post("/api/zone", json={"camera_id": "camT",
                                   "poly": [[0, 0], [1, 1]]})
    assert r2.status_code == 400


def test_protect_zone_import_endpoint(client, monkeypatch):
    from package_watcher.ui import hass, protect
    from package_watcher.config import UnifiConfig

    monkeypatch.setattr(hass, "discover_unifi_protect",
                        lambda: UnifiConfig(host="10.0.0.5", api_key="k"))
    monkeypatch.setattr(protect, "available", lambda: True)
    monkeypatch.setattr(protect, "camera_zones", lambda cfg, cam: [
        {"name": "Porch", "kind": "smart",
         "points": [[0.1, 0.5], [0.9, 0.5], [0.9, 1.0], [0.1, 1.0]]}])
    c, _ = client
    body = c.get("/api/zone/protect?camera_id=abc").get_json()
    assert body["zones"][0]["name"] == "Porch"
    assert len(body["zones"][0]["points"]) == 4


def test_live_worker_resolves_zone_via_camera_map(tmp_path):
    from package_watcher.config import (AppConfig, CameraConfig, SinkConfig,
                                        UnifiConfig)
    from package_watcher.service import WatcherService

    poly = [[0.2, 0.5], [0.6, 0.5], [0.6, 0.8], [0.2, 0.8]]
    cfg = AppConfig(
        cameras=[CameraConfig(name="front", source="dummy://")],
        unifi=UnifiConfig(host="h", camera_map={"front": "Front Door"}),
        sinks=SinkConfig(stdout=False),
        zones={"Front Door": poly})
    svc = WatcherService(cfg)
    worker = svc.workers["front"]
    assert worker.detector._zone_norm == [tuple(p) for p in poly]


def test_snapshot_requires_protect(client, monkeypatch):
    from package_watcher.ui import hass
    monkeypatch.setattr(hass, "discover_unifi_protect", lambda: None)
    monkeypatch.delenv("SUPERVISOR_TOKEN", raising=False)
    c, _ = client
    r = c.get("/api/snapshot?camera_id=x&at=2026-07-05T15:30:00")
    assert r.status_code == 400


def test_preview_case_runs_unsaved(client):
    """The wizard's verify step grades a case that isn't in cases.yaml yet."""
    c, _ = client
    r = c.post("/api/preview_case", json={
        "name": "adhoc-empty", "expect": "no_detect",
        "clip": "clips/seed-empty.mp4", "fps": 2.0,
    })
    body = r.get_json()
    assert r.status_code == 200
    assert body["passed"] is True
    assert "no detection" in body["reason"]
    # nothing was written to the manifest
    assert "adhoc-empty" not in [x["name"] for x in c.get("/api/cases").get_json()]


def test_preview_case_reports_detection_image(client):
    c, fixtures = client
    package_clip(fixtures / "clips" / "pkg.mp4")
    r = c.post("/api/preview_case", json={
        "name": "adhoc-pkg", "expect": "detect",
        "clip": "clips/pkg.mp4", "fps": 2.0,
        "detector": {"persist_samples": 6},
    })
    body = r.get_json()
    assert r.status_code == 200
    assert body["passed"] is True and body["detections"]
    assert body["images"]["detection"].startswith("data:image/png;base64,")
    # the shown frame is the detection that matched the expectation
    assert body["matched"] is True
    assert body["detection_time"] is not None


def test_preview_case_overlays_expected_region(client):
    c, fixtures = client
    package_clip(fixtures / "clips" / "pkg.mp4")
    r = c.post("/api/preview_case", json={
        "name": "adhoc-region", "expect": "detect",
        "clip": "clips/pkg.mp4", "fps": 2.0,
        "detector": {"persist_samples": 6},
        "region": list(PKG_REGION),
    })
    body = r.get_json()
    assert r.status_code == 200
    # region overlay path runs and still returns a detection image + timing
    assert body["images"]["detection"].startswith("data:image/png;base64,")
    assert "detection_time" in body


def test_events_to_windows_filters_clamps_and_merges():
    from datetime import datetime, timezone
    from types import SimpleNamespace as E
    from package_watcher.ui.protect import _events_to_windows

    utc = timezone.utc
    start = datetime(2026, 7, 5, 15, 0, 0, tzinfo=utc)
    end = datetime(2026, 7, 5, 15, 1, 0, tzinfo=utc)  # 60s clip

    def dt(s):
        return datetime(2026, 7, 5, 15, 0, s, tzinfo=utc)

    events = [
        E(camera_id="cam1", start=dt(8), end=dt(14)),
        E(camera_id="cam2", start=dt(20), end=dt(30)),        # other camera
        E(camera_id="cam1", start=dt(12), end=dt(18)),        # overlaps first
        E(camera_id="cam1", start=dt(50), end=None),          # ongoing -> end
        E(camera_id="cam1", start=None, end=dt(40)),          # malformed
    ]
    windows = _events_to_windows(events, "cam1", start, end)
    assert windows == [(8.0, 18.0), (50.0, 60.0)]


def test_pull_returns_presence_windows(client, monkeypatch):
    from package_watcher.ui import hass, protect
    from package_watcher.config import UnifiConfig

    monkeypatch.setattr(hass, "discover_unifi_protect",
                        lambda: UnifiConfig(host="10.0.0.5", api_key="k"))
    monkeypatch.setattr(protect, "available", lambda: True)
    monkeypatch.setattr(protect, "pull_clip",
                        lambda cfg, cam, s, e, out: out)
    monkeypatch.setattr(protect, "person_windows",
                        lambda cfg, cam, s, e: [(8.0, 14.0)])
    c, _ = client
    r = c.post("/api/pull", json={
        "camera_id": "abc",
        "start": "2026-07-05T15:00:00+00:00",
        "end": "2026-07-05T15:01:00+00:00"})
    body = r.get_json()
    assert r.status_code == 200
    assert body["clip"].startswith("clips/")
    assert body["presence"] == [[8.0, 14.0]]


def test_pull_survives_presence_failure(client, monkeypatch):
    from package_watcher.ui import hass, protect
    from package_watcher.config import UnifiConfig

    monkeypatch.setattr(hass, "discover_unifi_protect",
                        lambda: UnifiConfig(host="10.0.0.5", api_key="k"))
    monkeypatch.setattr(protect, "available", lambda: True)
    monkeypatch.setattr(protect, "pull_clip",
                        lambda cfg, cam, s, e, out: out)

    def boom(cfg, cam, s, e):
        raise RuntimeError("events API down")

    monkeypatch.setattr(protect, "person_windows", boom)
    c, _ = client
    r = c.post("/api/pull", json={
        "camera_id": "abc",
        "start": "2026-07-05T15:00:00+00:00",
        "end": "2026-07-05T15:01:00+00:00"})
    body = r.get_json()
    assert r.status_code == 200          # the clip is not lost
    assert body["presence"] == []
    assert "events API down" in body["presence_error"]


def test_person_gated_case_end_to_end(client):
    """Wizard round-trip: preview an unsaved gated case, save it, run it,
    then reopen it via the case list (full detail comes back)."""
    c, fixtures = client
    delivery_clip(fixtures / "clips" / "delivery.mp4",
                  warmup_s=8, visit_s=6, tail_s=12)
    payload = {
        "name": "gated-real-delivery", "expect": "detect",
        "clip": "clips/delivery.mp4",
        "fps": 2.0,
        "detector": {"mode": "person_gated", "settle_samples": 3},
        "presence": [[8.0, 14.0]],
        "region": list(PKG_REGION),
    }
    # Verify step grades it before saving.
    prev = c.post("/api/preview_case", json=payload).get_json()
    assert prev["passed"] is True and prev["matched"] is True

    # Save, then the saved case runs and passes too.
    assert c.post("/api/save_case", json=payload).status_code == 200
    text = (fixtures / "cases.yaml").read_text()
    assert "mode: person_gated" in text and "presence:" in text
    res = {r["name"]: r for r in c.post("/api/run").get_json()}
    assert res["gated-real-delivery"]["status"] == "pass"

    # Reopening: the case list carries everything the wizard needs to edit.
    listed = {x["name"]: x for x in c.get("/api/cases").get_json()}
    saved = listed["gated-real-delivery"]
    assert saved["clip"] == "clips/delivery.mp4" and saved["present"] is True
    assert saved["detector"]["mode"] == "person_gated"
    assert saved["presence"] == [[8.0, 14.0]]
    assert saved["region"] == list(PKG_REGION)

    # The case-list preview PNG must show the SAME detection the grade was
    # based on: the green box drawn at the package, not at the first blob.
    import cv2
    import numpy as np
    from videogen import PKG

    png = c.get("/api/preview/gated-real-delivery.png?kind=detection").data
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    x, y, w, h = PKG
    pad = 6
    roi = img[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
    green = ((roi[:, :, 0] < 120) & (roi[:, :, 1] > 170) & (roi[:, :, 2] < 120))
    assert green.sum() > 20, "matching detection box not drawn at the package"


def test_cameras_use_discovered_protect(client, monkeypatch):
    from package_watcher.ui import hass, protect
    from package_watcher.config import UnifiConfig

    monkeypatch.setattr(hass, "discover_unifi_protect",
                        lambda: UnifiConfig(host="10.0.0.5", api_key="k"))
    monkeypatch.setattr(protect, "available", lambda: True)
    monkeypatch.setattr(protect, "list_cameras",
                        lambda cfg: [{"id": "abc", "name": "G4 Doorbell"}])
    c, _ = client
    body = c.get("/api/cameras").get_json()
    assert body["available"] is True
    assert body["source"] == "protect"
    assert body["discovered"] is True
    assert body["supports_pull"] is True
    assert body["cameras"][0]["name"] == "G4 Doorbell"


def test_save_new_case_appends_and_preserves(client):
    c, fixtures = client
    package_clip(fixtures / "clips" / "pkg.mp4")
    payload = {
        "name": "new real pkg", "expect": "detect",
        "clip": "clips/pkg.mp4",
        "fps": 2.0, "detector": {"persist_samples": 6},
        "region": list(PKG_REGION), "after": "6",
    }
    r = c.post("/api/save_case", json=payload)
    assert r.status_code == 200
    assert r.get_json()["saved"] == "new-real-pkg"

    text = (fixtures / "cases.yaml").read_text()
    assert "seed-empty" in text          # original survived
    assert "new-real-pkg" in text        # new one appended

    names = [x["name"] for x in c.get("/api/cases").get_json()]
    assert "new-real-pkg" in names
    # And it actually runs + passes through the harness.
    res = {r["name"]: r for r in c.post("/api/run").get_json()}
    assert res["new-real-pkg"]["status"] == "pass"


def test_save_rejects_bad_expect(client):
    c, _ = client
    r = c.post("/api/save_case", json={"name": "x", "expect": "maybe",
                                       "clip": "clips/x.mp4"})
    assert r.status_code == 400


def test_upload_clip_saved(client):
    c, fixtures = client
    data = {"file": (io.BytesIO(b"not really a video"), "my clip.mp4")}
    r = c.post("/api/upload", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    clip = r.get_json()["clip"]
    assert clip.startswith("clips/")
    assert (fixtures / clip).is_file()
