from __future__ import annotations

import pytest

from package_watcher.config import load_config


VALID = """
events_dir: /data/events
cameras:
  - name: front-door
    source: rtsps://192.168.1.1:7441/token?enableSrtp
    sample_fps: 2.0
    zone:
      - [0.0, 0.5]
      - [1.0, 0.5]
      - [1.0, 1.0]
      - [0.0, 1.0]
detector:
  persist_samples: 10
unifi:
  host: 192.168.1.1
  username: watcher
  password: ${PW_TEST_PASSWORD}
  camera_map:
    front-door: "Front Door"
sinks:
  jsonl_path: /data/events.jsonl
  webhook_url: http://ha.local/api/webhook/pw
"""


def test_load_valid_config(tmp_path, monkeypatch):
    monkeypatch.setenv("PW_TEST_PASSWORD", "s3cret")
    path = tmp_path / "config.yaml"
    path.write_text(VALID)
    cfg = load_config(str(path))

    assert cfg.events_dir == "/data/events"
    cam = cfg.cameras[0]
    assert cam.name == "front-door"
    assert cam.sample_fps == 2.0
    assert cam.zone == [(0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)]
    assert cfg.detector.persist_samples == 10
    assert cfg.detector.fast_alpha == 0.15  # default preserved
    assert cfg.unifi.password == "s3cret"
    assert cfg.unifi.camera_map == {"front-door": "Front Door"}
    assert cfg.sinks.webhook_url == "http://ha.local/api/webhook/pw"


def test_missing_env_var_is_loud(tmp_path, monkeypatch):
    monkeypatch.delenv("PW_TEST_PASSWORD", raising=False)
    path = tmp_path / "config.yaml"
    path.write_text(VALID)
    with pytest.raises(KeyError, match="PW_TEST_PASSWORD"):
        load_config(str(path))


def test_no_cameras_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("cameras: []\n")
    with pytest.raises(ValueError, match="at least one camera"):
        load_config(str(path))


def test_unknown_keys_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "cameras:\n  - name: a\n    source: x.mp4\n    tyop: 1\n")
    with pytest.raises(ValueError, match="tyop"):
        load_config(str(path))
