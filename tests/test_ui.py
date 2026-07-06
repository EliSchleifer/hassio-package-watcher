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

MANIFEST = """cases:
  - name: seed-empty
    scene: {scene: empty, seconds: 10}
    fps: 2.0
    expect: no_detect
"""


@pytest.fixture()
def client(tmp_path):
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
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
    payload = {
        "name": "new synthetic pkg", "expect": "detect",
        "scene": {"scene": "package", "hold_s": 14},
        "fps": 2.0, "detector": {"persist_samples": 6},
        "region": [0.4, 0.58, 0.22, 0.22], "after": "6",
    }
    r = c.post("/api/save_case", json=payload)
    assert r.status_code == 200
    assert r.get_json()["saved"] == "new-synthetic-pkg"

    text = (fixtures / "cases.yaml").read_text()
    assert "seed-empty" in text          # original survived
    assert "new-synthetic-pkg" in text   # new one appended

    names = [x["name"] for x in c.get("/api/cases").get_json()]
    assert "new-synthetic-pkg" in names
    # And it actually runs + passes through the harness.
    res = {r["name"]: r for r in c.post("/api/run").get_json()}
    assert res["new-synthetic-pkg"]["status"] == "pass"


def test_save_rejects_bad_expect(client):
    c, _ = client
    r = c.post("/api/save_case", json={"name": "x", "expect": "maybe",
                                       "scene": {"scene": "empty"}})
    assert r.status_code == 400


def test_upload_clip_saved(client):
    c, fixtures = client
    data = {"file": (io.BytesIO(b"not really a video"), "my clip.mp4")}
    r = c.post("/api/upload", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    clip = r.get_json()["clip"]
    assert clip.startswith("clips/")
    assert (fixtures / clip).is_file()
