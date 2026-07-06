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


def test_cameras_unavailable_without_unifi(client):
    c, _ = client
    body = c.get("/api/cameras").get_json()
    assert body["available"] is False


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
