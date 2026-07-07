"""Flask app for authoring and running fixture cases.

Workflow the UI supports:
  1. Pull a real clip from a Protect camera (scrub, mark in/out — person
     presence windows are imported automatically), upload a local video, or
     reference one already in fixtures/clips/.
  2. Label it "should detect a package" or "should NOT"; draw the expected
     region on the clip; pick the detection mode.
  3. Verify: run the detector on the unsaved case and confirm it grades the
     way you intend (annotated frame, diff mask, pass/fail).
  4. Save it into fixtures/cases.yaml — where `pytest` / `package-watcher
     test` will grade it from then on. Saved cases can be reopened, watched,
     and edited from the case list.

Single-file app with embedded templates so it has no asset build step.
"""

from __future__ import annotations

import base64
import io
import os
import threading
from datetime import datetime
from typing import Any, Optional

import yaml

from ..config import UnifiConfig
from ..harness import FixtureCase, load_cases, run_and_evaluate


def create_app(fixtures_dir: str, unifi: Optional[UnifiConfig] = None,
               reload: bool = False, verifier_cfg=None,
               zones_path: Optional[str] = None):
    try:
        from flask import (Flask, Response, jsonify, request,
                           send_from_directory)
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "the UI needs Flask; install with: pip install "
            "'package-watcher[ui]'") from exc

    import cv2

    fixtures_dir = os.path.abspath(fixtures_dir)
    clips_dir = os.path.join(fixtures_dir, "clips")
    manifest_path = os.path.join(fixtures_dir, "cases.yaml")
    # Watch zones live in the CONFIG space (zones.yaml next to config.yaml)
    # so the live watcher picks them up too; fixtures dir is the fallback
    # when the UI runs without a config file.
    zones_path = zones_path or os.path.join(fixtures_dir, "zones.yaml")
    os.makedirs(clips_dir, exist_ok=True)

    # Per-camera watch zones (normalized polygons keyed by camera id AND
    # display name — the live service resolves by name via camera_map):
    # everything outside the zone is ignored.
    def _load_zones() -> dict[str, list]:
        if not os.path.isfile(zones_path):
            return {}
        with open(zones_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _store_zone(camera_id: str, poly) -> None:
        zones = _load_zones()
        if poly:
            zones[camera_id] = poly
        else:
            zones.pop(camera_id, None)
        with open(zones_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(zones, f, sort_keys=True)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512 MB uploads

    # Second-stage semantic verification (Florence), when configured. Built
    # once; the model itself loads lazily on first use.
    from ..verify import build_verifier
    verifier = build_verifier(verifier_cfg) if verifier_cfg else None

    # Live-reload (dev): the page polls /__alive; when the process restarts
    # (Flask's reloader on file save), the boot id changes and the browser
    # reloads itself — edit -> save -> refreshed page, no manual F5.
    # The 1 Hz poll would flood the access log, so drop those lines.
    import logging as _logging

    class _AliveFilter(_logging.Filter):
        def filter(self, record):  # noqa: A003
            return "__alive" not in record.getMessage()

    _logging.getLogger("werkzeug").addFilter(_AliveFilter())
    boot_id = str(os.getpid())
    livereload = (
        "<script>let _b=null;setInterval(async()=>{try{"
        "const r=await fetch('__alive');const t=await r.text();"
        "if(_b&&_b!==t)location.reload();_b=t;}catch(e){}},1000);</script>"
    ) if reload else ""

    @app.get("/__alive")
    def _alive() -> Any:
        return Response(boot_id, mimetype="text/plain")

    def _resolve_unifi():
        """(UnifiConfig|None, discovered): explicit `unifi` block wins;
        otherwise try to reuse the HA UniFi Protect integration's creds."""
        if unifi is not None:
            return unifi, False
        from . import hass
        return hass.discover_unifi_protect(), True

    # --- pages ------------------------------------------------------------
    @app.get("/")
    def index() -> Any:
        # Under Home Assistant ingress the page is served from a token-prefixed
        # path (e.g. /api/hassio_ingress/<token>/); HA passes that prefix in
        # X-Ingress-Path. Emitting it as the document <base> lets every
        # relative fetch/img URL below resolve correctly both there and when
        # run standalone (prefix empty -> base "/").
        prefix = request.headers.get("X-Ingress-Path", "").rstrip("/")
        return (_PAGE.replace("__INGRESS_BASE__", prefix)
                     .replace("__LIVERELOAD__", livereload))

    # --- case data --------------------------------------------------------
    @app.get("/api/cases")
    def api_cases() -> Any:
        cases = load_cases(manifest_path) if os.path.isfile(manifest_path) else []
        out = []
        for c in cases:
            p = (c.clip if os.path.isabs(c.clip)
                 else os.path.join(fixtures_dir, c.clip))
            out.append({
                # Full detail so the UI can reopen a saved case for review
                # and editing, not just list it.
                "name": c.name, "expect": c.expect,
                "clip": c.clip, "description": c.description,
                "present": os.path.isfile(p),
                "fps": c.fps,
                "detector": c.detector,
                "region": list(c.region) if c.region else None,
                "presence": [list(w) for w in c.presence],
                "after": c.after, "before": c.before,
            })
        return jsonify(out)

    @app.post("/api/run")
    def api_run() -> Any:
        name = request.args.get("name")
        cases = load_cases(manifest_path)
        if name:
            cases = [c for c in cases if c.name == name]
        results = []
        for case in cases:
            if case.clip:
                p = (case.clip if os.path.isabs(case.clip)
                     else os.path.join(fixtures_dir, case.clip))
                if not os.path.isfile(p):
                    results.append({"name": case.name, "status": "skip",
                                    "reason": f"clip missing: {case.clip}"})
                    continue
            outcome = run_and_evaluate(case, fixtures_dir)
            results.append({
                "name": case.name,
                "status": "pass" if outcome.passed else "fail",
                "expect": case.expect,
                "reason": outcome.reason,
                "detections": [
                    {"t": round(d.t, 1), "bbox": [round(v, 3) for v in d.bbox_norm],
                     "confidence": round(d.confidence, 3), "triggered": d.triggered}
                    for d in outcome.result.detections],
            })
        return jsonify(results)

    @app.get("/api/preview/<name>.png")
    def api_preview(name: str) -> Any:
        cases = {c.name: c for c in load_cases(manifest_path)}
        case = cases.get(name)
        if case is None:
            return Response("no such case", status=404)
        kind = request.args.get("kind", "detection")
        # Evaluate (not just run) so this shows the SAME detection the grade
        # was based on — the one matching the expected region — keeping the
        # case-list preview consistent with the wizard's verify view.
        outcome = run_and_evaluate(case, fixtures_dir, capture_preview=True)
        res = outcome.result
        frames = res.frames_for_preview
        mi = outcome.matched_index
        img = None
        if mi is not None and mi < len(res.det_frames):
            img = {"detection": res.det_frames[mi],
                   "mask": res.det_masks[mi]}.get(kind)
        if img is None:
            img = frames.get(kind)
        if img is None:
            img = frames.get("detection")
        if img is None:
            img = frames.get("first")
        if img is None:
            return Response("no preview frame", status=404)
        if kind == "detection" and case.region is not None:
            img = img.copy()
            dh, dw = img.shape[:2]
            rx, ry, rw, rh = case.region
            cv2.rectangle(img, (int(rx * dw), int(ry * dh)),
                          (int((rx + rw) * dw), int((ry + rh) * dh)),
                          (255, 200, 0), 2)
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            return Response("encode failed", status=500)
        return Response(buf.tobytes(), mimetype="image/png")

    # --- Protect + clips --------------------------------------------------
    @app.get("/api/cameras")
    def api_cameras() -> Any:
        from . import hass, protect
        # Prefer a Protect NVR — explicitly configured, or auto-discovered from
        # the HA UniFi Protect integration — since it can pull recorded clips
        # by time range. Otherwise fall back to listing camera.* entities from
        # the HA Core API (no credentials, but no historical footage either).
        u, discovered = _resolve_unifi()
        if u is not None:
            if not protect.available():
                return jsonify({"available": False,
                                "reason": "uiprotect not installed"})
            try:
                return jsonify({"available": True, "source": "protect",
                                "discovered": discovered, "supports_pull": True,
                                "cameras": protect.list_cameras(u)})
            except Exception as exc:  # noqa: BLE001
                return jsonify({"available": False, "reason": str(exc)})
        if hass.available():
            try:
                return jsonify({"available": True, "source": "homeassistant",
                                "supports_pull": False,
                                "cameras": hass.list_cameras()})
            except Exception as exc:  # noqa: BLE001
                return jsonify({"available": False,
                                "reason": f"Home Assistant API: {exc}"})
        return jsonify({"available": False,
                        "reason": "no unifi block, no discoverable Protect "
                                  "integration, and no Home Assistant API "
                                  "(run as an add-on with homeassistant_api)"})

    @app.get("/api/snapshot")
    def api_snapshot() -> Any:
        """One historical frame at a timestamp — drives the scrubber preview.
        Served as image/jpeg so the browser can load it straight into an
        <img>, fetched on demand as the user scrubs (no video download)."""
        from . import protect
        u, _ = _resolve_unifi()
        if u is None or not protect.available():
            return Response("Protect not configured", status=400)
        camera_id = request.args.get("camera_id")
        at = request.args.get("at")
        if not camera_id:
            return Response("camera_id required", status=400)
        try:
            # No `at` = the live view (recorded footage at "now" does not
            # exist yet) — used by the zone drawer.
            dt = datetime.fromisoformat(at) if at else None
            width = int(request.args.get("width", 640))
        except ValueError as exc:
            return Response(f"bad params: {exc}", status=400)
        try:
            data = protect.snapshot_at(u, camera_id, dt, width=width)
        except Exception as exc:  # noqa: BLE001
            return Response(f"snapshot failed: {exc}", status=502)
        if not data:
            return Response("no frame at that time", status=404)
        return Response(data, mimetype="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.post("/api/pull")
    def api_pull() -> Any:
        from . import protect
        u, _ = _resolve_unifi()
        if u is None or not protect.available():
            return jsonify({"error": "Protect not configured"}), 400
        data = request.get_json(force=True)
        try:
            start = datetime.fromisoformat(data["start"])
            end = datetime.fromisoformat(data["end"])
        except (KeyError, ValueError) as exc:
            return jsonify({"error": f"bad start/end: {exc}"}), 400
        fname = _safe_name(data.get("filename") or
                           f"{data['camera_id']}-{start:%Y%m%dT%H%M%S}.mp4")
        out = os.path.join(clips_dir, fname)
        try:
            protect.pull_clip(u, data["camera_id"], start, end, out)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc)}), 500
        # Also fetch person smart-detect windows for the same range, so the
        # clip arrives pre-labeled for person-gated mode. Best-effort: a
        # failure here must not lose the clip we just pulled.
        presence: list = []
        presence_error = None
        try:
            presence = protect.person_windows(u, data["camera_id"], start, end)
        except Exception as exc:  # noqa: BLE001
            presence_error = str(exc)
        return jsonify({"clip": f"clips/{fname}", "presence": presence,
                        **({"presence_error": presence_error}
                           if presence_error else {})})

    @app.post("/api/upload")
    def api_upload() -> Any:
        if "file" not in request.files:
            return jsonify({"error": "no file"}), 400
        f = request.files["file"]
        fname = _safe_name(f.filename or "upload.mp4")
        f.save(os.path.join(clips_dir, fname))
        return jsonify({"clip": f"clips/{fname}"})

    @app.get("/clips/<path:name>")
    def serve_clip(name: str) -> Any:
        return send_from_directory(clips_dir, name)

    @app.post("/api/save_case")
    def api_save_case() -> Any:
        data = request.get_json(force=True)
        try:
            case = _case_from_form(data)
        except (ValueError, KeyError) as exc:
            return jsonify({"error": str(exc)}), 400
        try:
            _upsert_case(manifest_path, case)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc)}), 500
        return jsonify({"saved": case["name"]})

    @app.post("/api/preview_case")
    def api_preview_case() -> Any:
        """Run the detector on an *unsaved* case built from the wizard form, so
        the user can verify the clip + expectation before committing it."""
        data = request.get_json(force=True)
        try:
            case_dict = _case_from_form(data)
        except (ValueError, KeyError) as exc:
            return jsonify({"error": str(exc)}), 400
        entry = dict(case_dict)
        if "region" in entry:
            entry["region"] = tuple(entry["region"])
        if "zone" in entry:
            entry["zone"] = [tuple(pt) for pt in entry["zone"]]
        if "presence" in entry:
            entry["presence"] = [tuple(w) for w in entry["presence"]]
        case = FixtureCase(**entry)
        try:
            outcome = run_and_evaluate(case, fixtures_dir, capture_preview=True)
        except FileNotFoundError as exc:
            return jsonify({"error": f"clip not found: {exc}"}), 400
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc)}), 500
        res = outcome.result
        frames = res.frames_for_preview
        dets = res.detections
        # Prefer the detection that actually satisfied the expectation, so the
        # image matches the verdict (not just the first, noisy detection).
        mi = outcome.matched_index
        if mi is not None and mi < len(res.det_frames):
            det = res.det_frames[mi]
            mask = res.det_masks[mi]
            det_t = dets[mi].t
        else:
            det = frames.get("detection")
            if det is None:
                det = frames.get("first")
            mask = frames.get("mask")
            det_t = dets[0].t if dets else None
        # Overlay the expected region (cyan) next to the detector's own box
        # (green) so "right thing, right place?" is answerable at a glance.
        if det is not None and case.region is not None:
            det = det.copy()
            dh, dw = det.shape[:2]
            rx, ry, rw, rh = case.region
            cv2.rectangle(det, (int(rx * dw), int(ry * dh)),
                          (int((rx + rw) * dw), int((ry + rh) * dh)),
                          (255, 200, 0), 2)
        # Second stage: caption the matched candidate's clean crop, so the
        # wizard shows what the vision model thinks it is.
        verdict = None
        if verifier is not None and mi is not None and mi < len(res.det_raw):
            try:
                verdict = verifier.verify(res.det_raw[mi], res.det_bboxes[mi])
            except Exception as exc:  # noqa: BLE001
                verdict = {"error": str(exc)}
        return jsonify({
            "passed": outcome.passed,
            "reason": outcome.reason,
            "expect": case.expect,
            "detection_time": round(det_t, 1) if det_t is not None else None,
            "matched": mi is not None,
            "verification": verdict,
            "detections": [
                {"t": round(d.t, 1),
                 "bbox": [round(v, 3) for v in d.bbox_norm],
                 "confidence": round(d.confidence, 3)}
                for d in dets],
            "images": {
                "detection": _png_data_uri(det),
                "mask": _png_data_uri(mask),
            },
        })

    # --- backtest: scan a camera's recorded day every X minutes ------------
    bt = {"running": False, "progress": [0, 0], "error": None, "result": None,
          "partial": []}

    def _bt_json(res) -> dict[str, Any]:
        return {
            "camera_id": res.camera_id,
            "start": res.start.isoformat(), "end": res.end.isoformat(),
            "interval_s": res.interval_s,
            "samples_total": res.samples_total,
            "samples_skipped_person": res.samples_skipped_person,
            "samples_missing": res.samples_missing,
            "scene_flips": res.scene_flips,
            "hits": _group_hits(res.hits),
        }

    @app.get("/api/zone")
    def api_zone_get() -> Any:
        camera_id = request.args.get("camera_id", "")
        return jsonify({"camera_id": camera_id,
                        "zone": _load_zones().get(camera_id)})

    @app.get("/api/zone/protect")
    def api_zone_protect() -> Any:
        """Zones already configured on the camera in Protect, importable."""
        from . import protect
        u, _ = _resolve_unifi()
        if u is None or not protect.available():
            return jsonify({"error": "Protect not configured"}), 400
        camera_id = request.args.get("camera_id", "")
        try:
            return jsonify({"zones": protect.camera_zones(u, camera_id)})
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc)}), 502

    @app.post("/api/zone")
    def api_zone_set() -> Any:
        data = request.get_json(force=True)
        camera_id = data.get("camera_id")
        if not camera_id:
            return jsonify({"error": "camera_id required"}), 400
        rect = data.get("rect")  # [x, y, w, h] normalized, or null to clear
        poly = data.get("poly")  # or a full polygon [[x,y], ...]
        if poly:
            try:
                poly = [[float(a), float(b)] for a, b in poly]
            except (TypeError, ValueError):
                return jsonify({"error": "poly must be [[x, y], ...]"}), 400
            if len(poly) < 3:
                return jsonify({"error": "poly needs at least 3 points"}), 400
        elif rect:
            try:
                x, y, w, h = (float(v) for v in rect)
            except (TypeError, ValueError):
                return jsonify({"error": "rect must be [x, y, w, h]"}), 400
            if w <= 0 or h <= 0:
                return jsonify({"error": "rect must have positive size"}), 400
            poly = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        else:
            poly = None
        _store_zone(camera_id, poly)
        # Also key by display name so the live watcher can resolve the zone
        # through unifi.camera_map (which maps to Protect display names).
        if data.get("camera_name"):
            _store_zone(data["camera_name"], poly)
        return jsonify({"camera_id": camera_id, "zone": poly})

    @app.post("/api/backtest")
    def api_backtest() -> Any:
        from . import protect
        u, _ = _resolve_unifi()
        if u is None or not protect.available():
            return jsonify({"error": "Protect not configured"}), 400
        if bt["running"]:
            return jsonify({"error": "a backtest is already running"}), 409
        data = request.get_json(force=True)
        try:
            start = datetime.fromisoformat(data["start"])
            end = datetime.fromisoformat(data["end"])
            interval_s = float(data.get("interval_s", 600))
            camera_id = data["camera_id"]
        except (KeyError, ValueError) as exc:
            return jsonify({"error": f"bad params: {exc}"}), 400
        want_verify = bool(data.get("verify")) and verifier is not None

        def job():
            try:
                from .. import backtest as btmod
                try:
                    windows = protect.person_windows(u, camera_id, start, end)
                except Exception:  # noqa: BLE001 - windows are an optimization
                    windows = []
                raw_hits: list = []

                def stream(h):
                    raw_hits.append(h)
                    bt["partial"] = _group_hits(raw_hits)

                zone = _load_zones().get(camera_id)
                res = btmod.run_backtest(
                    btmod.protect_snapshot_fn(u, camera_id), camera_id,
                    start, end, interval_s=interval_s,
                    person_windows=windows,
                    zone=[tuple(p) for p in zone] if zone else None,
                    verifier=verifier if want_verify else None,
                    progress=lambda i, n: bt.update(progress=[i, n]),
                    on_hit=stream)
                bt["result"] = _bt_json(res)
            except Exception as exc:  # noqa: BLE001
                bt["error"] = str(exc)
            finally:
                bt["running"] = False

        bt.update(running=True, error=None, result=None, progress=[0, 0],
                  partial=[])
        threading.Thread(target=job, daemon=True, name="backtest").start()
        return jsonify({"started": True,
                        "verify": want_verify,
                        "verifier_available": verifier is not None})

    @app.get("/api/backtest/status")
    def api_backtest_status() -> Any:
        return jsonify({k: bt[k] for k in
                        ("running", "progress", "error", "result", "partial")})

    return app


# --- helpers --------------------------------------------------------------

def _group_jpg(frame, bboxes) -> Optional[str]:
    """One frame with every candidate box of a sample drawn and numbered."""
    import cv2

    img = frame.copy()
    th = max(2, img.shape[1] // 640)
    for i, (x, y, w, h) in enumerate(bboxes, 1):
        cv2.rectangle(img, (x, y), (x + w, y + h), (60, 220, 60), th)
        cv2.putText(img, str(i), (x, max(18, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 220, 60), th)
    scale = 480 / img.shape[1]
    if scale < 1:
        img = cv2.resize(img, (480, int(img.shape[0] * scale)),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return ("data:image/jpeg;base64," +
            base64.b64encode(buf.tobytes()).decode()) if ok else None


def _group_hits(hits) -> list[dict[str, Any]]:
    """One comparison can yield several blobs; present them as ONE sample
    with numbered boxes, not as separate same-timestamp events. Each group
    carries the comparison pair — the person-free 'before' snapshot and the
    annotated 'after' — so garbage candidates explain themselves (lighting
    moved, camera shifted, …)."""
    groups: dict[str, dict[str, Any]] = {}
    frames: dict[str, Any] = {}
    baselines: dict[str, Any] = {}
    bboxes: dict[str, list] = {}
    for h in hits:
        key = h.at.isoformat()
        g = groups.setdefault(key, {"at": key, "boxes": []})
        frames[key] = h.frame  # same frame for every blob of the sample
        if getattr(h, "baseline", None) is not None:
            baselines[key] = h.baseline
            g["before_at"] = (h.baseline_at.isoformat()
                              if h.baseline_at else None)
        bboxes.setdefault(key, []).append(h.bbox)
        g["boxes"].append({
            "bbox_norm": [round(v, 4) for v in h.bbox_norm],
            "confidence": round(h.confidence, 3),
            "verification": h.verification,
        })
    out = []
    for key, g in groups.items():
        g["jpg"] = _group_jpg(frames[key], bboxes[key])
        g["before_jpg"] = (_group_jpg(baselines[key], [])
                           if key in baselines else None)
        out.append(g)
    return out


def _png_data_uri(img) -> Optional[str]:
    if img is None:
        return None
    import cv2

    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return None
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def _safe_name(name: str) -> str:
    keep = "-_.() "
    cleaned = "".join(c for c in os.path.basename(name)
                      if c.isalnum() or c in keep).strip()
    return cleaned or "clip.mp4"


def _case_from_form(data: dict[str, Any]) -> dict[str, Any]:
    name = _safe_name(data.get("name", "")).replace(" ", "-")
    if not name:
        raise ValueError("name is required")
    expect = data.get("expect")
    if expect not in ("detect", "no_detect"):
        raise ValueError("expect must be 'detect' or 'no_detect'")
    case: dict[str, Any] = {"name": name, "expect": expect}
    if data.get("description"):
        case["description"] = data["description"]
    if data.get("clip"):
        case["clip"] = data["clip"]
    else:
        raise ValueError("a clip is required")
    case["fps"] = float(data.get("fps", 2.0))
    if data.get("detector"):
        case["detector"] = data["detector"]
    if data.get("region"):
        region = [float(v) for v in data["region"]]
        if len(region) != 4:
            raise ValueError("region must be [x, y, w, h]")
        case["region"] = region
    if data.get("after") not in (None, ""):
        case["after"] = float(data["after"])
    if data.get("before") not in (None, ""):
        case["before"] = float(data["before"])
    if data.get("zone"):
        case["zone"] = [[float(a), float(b)] for a, b in data["zone"]]
    if data.get("presence"):
        case["presence"] = [[float(a), float(b)] for a, b in data["presence"]]
    # Validate it constructs before we persist it.
    entry = dict(case)
    if "region" in entry:
        entry["region"] = tuple(entry["region"])
    if "zone" in entry:
        entry["zone"] = [tuple(pt) for pt in entry["zone"]]
    if "presence" in entry:
        entry["presence"] = [tuple(w) for w in entry["presence"]]
    FixtureCase(**entry)
    return case


def _upsert_case(manifest_path: str, case: dict[str, Any]) -> None:
    """Add a new case (appended as text, preserving comments) or replace an
    existing one by name (full rewrite)."""
    existing_text = ""
    doc: dict[str, Any] = {"cases": []}
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            existing_text = f.read()
        doc = yaml.safe_load(existing_text) or {"cases": []}
    names = [c.get("name") for c in doc.get("cases", [])]

    if case["name"] in names:
        # Replace in place; this reserializes and drops comments.
        doc["cases"] = [case if c.get("name") == case["name"] else c
                        for c in doc["cases"]]
        with open(manifest_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False)
        return

    # New case: append a formatted block so curated comments survive.
    block = yaml.safe_dump([case], sort_keys=False, default_flow_style=False)
    block = "\n".join("  " + line if line else line
                      for line in block.splitlines())
    text = existing_text
    if "cases:" not in text:
        text = (text + "\n" if text.strip() else "") + "cases:\n"
    if not text.endswith("\n"):
        text += "\n"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(text + block + "\n")


def serve(fixtures_dir: str, unifi: Optional[UnifiConfig],
          host: str, port: int, reload: bool = False,
          verifier_cfg=None, zones_path: Optional[str] = None) -> None:
    app = create_app(fixtures_dir, unifi, reload=reload,
                     verifier_cfg=verifier_cfg, zones_path=zones_path)
    print(f"package-watcher UI on http://{host}:{port}  "
          f"(fixtures: {os.path.abspath(fixtures_dir)})"
          + ("  [reload — edits auto-refresh the browser]" if reload else ""))
    # reload=True enables Flask's auto-restart-on-edit plus browser
    # live-reload for fast local iteration; off by default for the add-on.
    app.run(host=host, port=port, debug=reload, use_reloader=reload)


# --- embedded single-page UI ----------------------------------------------

_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<base href="__INGRESS_BASE__/">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>package-watcher · fixtures</title>
<style>
  :root { color-scheme: light dark; --line:#8884; --ok:#2a9d3f; --bad:#d13232;
          --skip:#9a7b1a; --accent:#2f6fed; }
  body { font-family: system-ui, sans-serif; margin: 0; line-height: 1.45; }
  header { padding: 12px 20px; border-bottom: 1px solid var(--line);
           display:flex; gap:16px; align-items:center; }
  header h1 { font-size: 16px; margin: 0; font-weight: 650; margin-right:auto; }
  main { padding: 20px; max-width: 900px; }
  h2 { font-size: 13px; text-transform: uppercase; letter-spacing:.05em;
       opacity:.7; margin: 0 0 10px; }
  button { font: inherit; padding: 6px 12px; border:1px solid var(--line);
           border-radius: 7px; background: transparent; cursor: pointer; }
  button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
  button.tab.active { background:var(--line); font-weight:650; }
  table { border-collapse: collapse; width: 100%; font-size: 14px; }
  td, th { text-align:left; padding: 6px 8px; border-bottom:1px solid var(--line); }
  .pill { font-size:12px; padding:2px 8px; border-radius:999px; border:1px solid var(--line); }
  .pass{ color:var(--ok); } .fail{ color:var(--bad); } .skip{ color:var(--skip); }
  .reason { font-size:12px; opacity:.75; }
  label { display:block; font-size:13px; margin:8px 0 3px; opacity:.85; }
  input, select, textarea { font: inherit; padding:6px 8px; width:100%;
    box-sizing:border-box; border:1px solid var(--line); border-radius:6px;
    background:transparent; color:inherit; }
  input[type=range]{ padding:0; }
  fieldset { border:1px solid var(--line); border-radius:8px; margin:10px 0; }
  .row { display:flex; gap:10px; align-items:center; } .row > * { flex:1; }
  .row.tight > * { flex:0 0 auto; }
  img.preview { max-width:100%; border:1px solid var(--line); border-radius:6px;
                margin-top:8px; }
  .muted { opacity:.6; font-size:12px; }
  /* wizard modal */
  .modal { position:fixed; inset:0; background:#0009; display:none;
           align-items:flex-start; justify-content:center; padding:24px;
           overflow:auto; z-index:20; }
  .sheet { background:Canvas; color:CanvasText; width:min(780px,100%);
           border:1px solid var(--line); border-radius:12px; padding:16px 18px; }
  .wizhead { display:flex; align-items:center; gap:12px; margin-bottom:6px; }
  .wizhead b { font-size:15px; }
  .steps { margin-left:auto; font-size:12px; opacity:.7; }
  .steps span[data-s].on { color:var(--accent); font-weight:700; opacity:1; }
  .frame { width:100%; background:#000; border:1px solid var(--line);
           border-radius:6px; min-height:200px; object-fit:contain; display:block; }
  .wiznav { display:flex; align-items:center; gap:12px; margin-top:14px;
            border-top:1px solid var(--line); padding-top:12px; }
  .wiznav .muted { margin-left:auto; }
  video { width:100%; border:1px solid var(--line); border-radius:6px; }
</style></head>
<body>
<header>
  <h1>📦 package-watcher fixtures</h1>
  <button class="primary" onclick="openWiz()">+ Add a case</button>
  <button onclick="runAll()">Run all</button>
  <span id="summary" class="muted"></span>
</header>
<main>
  <section>
    <h2>Fixture cases</h2>
    <table id="cases"><tbody></tbody></table>
    <div id="preview"></div>
  </section>

  <section style="margin-top:28px">
    <h2>Backtest a day</h2>
    <div class="muted">Scan a camera's recorded history: one snapshot every
      X minutes (people skipped via Protect events), each compared with the
      previous one — candidate packages are shown with where they appeared.
      Optionally the vision model captions each candidate.</div>
    <div class="row" style="margin-top:8px">
      <div><label>Camera</label>
        <select id="btCam"><option value="">— none —</option></select></div>
      <div><label>Day</label><input type="date" id="btDate" autocomplete="off"
        data-form-type="other" data-lpignore="true" data-1p-ignore></div>
      <div><label>Every</label>
        <select id="btInt">
          <option value="300">5 min</option>
          <option value="600" selected>10 min</option>
          <option value="900">15 min</option>
          <option value="1800">30 min</option>
        </select></div>
      <div><label>2nd stage</label>
        <select id="btVerify">
          <option value="1" selected>Florence verify</option>
          <option value="">CV only</option>
        </select></div>
      <div style="flex:0 0 auto"><label>&nbsp;</label>
        <button class="primary" onclick="runBacktest()">Run</button></div>
    </div>
    <div class="row tight" style="gap:8px;margin-top:6px">
      <button onclick="openZone()">🎯 Watch zone…</button>
      <span id="zoneInfo" class="muted"></span>
    </div>
    <div id="zoneUI" style="display:none;max-width:640px;margin-top:8px">
      <div class="muted">Drag a rectangle over the area to care about
        (the porch landing) — everything outside it is ignored by backtests
        on this camera.</div>
      <div style="position:relative;line-height:0;margin-top:6px">
        <img id="zimg" style="width:100%;border:1px solid var(--line);border-radius:6px">
        <canvas id="zcanvas" style="position:absolute;left:0;top:0;cursor:crosshair"></canvas>
      </div>
      <div class="row tight" style="gap:6px;margin-top:6px">
        <button class="primary" onclick="saveZone()">Save zone</button>
        <button onclick="clearZone()">Clear zone</button>
        <select id="pzones" style="width:auto"><option value="">import from Protect…</option></select>
        <button onclick="document.getElementById('zoneUI').style.display='none'">Close</button>
      </div>
    </div>
    <div id="btStatus" class="muted" style="margin-top:6px"></div>
    <div id="btResults"></div>
  </section>
</main>

<div class="modal" id="modal">
 <div class="sheet">
  <div class="wizhead">
    <b>Add a case</b>
    <span class="steps">
      <span data-s="1">1 Source</span> ›
      <span data-s="2">2 Expectation</span> ›
      <span data-s="3">3 Verify &amp; save</span>
    </span>
    <button onclick="closeWiz()">✕</button>
  </div>

  <!-- STEP 1: SOURCE -->
  <div class="wizstep" data-step="1">
    <div id="step1tabs" class="row tight" style="gap:6px;margin-bottom:8px">
      <button class="tab" data-src="scrub" onclick="srcTab('scrub')">Scrub Protect</button>
      <button class="tab" data-src="upload" onclick="srcTab('upload')">Upload file</button>
      <button class="tab" data-src="ref" onclick="srcTab('ref')">Reference a clip</button>
    </div>

    <div class="srcpane" data-src="scrub">
      <div class="row">
        <select id="camera"><option value="">— none / unavailable —</option></select>
        <button onclick="loadCameras()">Refresh</button>
      </div>
      <div id="cameraNote" class="muted"></div>
      <div class="row">
        <div><label>Start</label>
          <input type="datetime-local" id="scrubStart" onchange="startScrub()"
            autocomplete="off" data-form-type="other" data-lpignore="true"
            data-1p-ignore></div>
        <div><label>Timeline span</label>
          <select id="scrubWindow" onchange="startScrub()">
            <option value="120">2 min</option>
            <option value="300" selected>5 min</option>
            <option value="600">10 min</option>
            <option value="1800">30 min</option>
          </select></div>
      </div>
      <div id="scrubUI" style="display:none;margin-top:8px">
        <img id="frame" class="frame" alt="frame at playhead">
        <div class="muted" id="frameTime" style="text-align:center;margin:4px 0"></div>
        <input type="range" id="timeline" min="0" max="300" step="1" value="0"
               oninput="onScrub()">
        <div class="row tight" style="justify-content:center;gap:6px;margin-top:8px;flex-wrap:wrap">
          <button onclick="nudge(-30)">−30s</button>
          <button onclick="nudge(-5)">−5s</button>
          <button onclick="nudge(-1)">−1s</button>
          <button onclick="nudge(1)">+1s</button>
          <button onclick="nudge(5)">+5s</button>
          <button onclick="nudge(30)">+30s</button>
          <button onclick="markIn()">⟤ Set In</button>
          <button onclick="markOut()">Set Out ⟥</button>
        </div>
        <div class="muted" id="stripSel" style="text-align:center;margin-top:6px"></div>
        <div style="text-align:center;margin-top:6px">
          <button class="primary" onclick="makeClip()">Pull clip from selection</button>
        </div>
      </div>
    </div>

    <div class="srcpane" data-src="upload" style="display:none">
      <label>Upload a video file</label>
      <input type="file" id="upload" accept="video/*" onchange="uploadClip()">
    </div>

    <div class="srcpane" data-src="ref" style="display:none">
      <label>Reference an existing clip (relative to the fixtures dir)</label>
      <input id="clip" placeholder="clips/my-clip.mp4" onchange="refClip()">
    </div>

    <hr style="border:none;border-top:1px solid var(--line);margin:14px 0">
    <div id="clipWatch"></div>
  </div>

  <!-- STEP 2: EXPECTATION -->
  <div class="wizstep" data-step="2" style="display:none">
    <div class="row">
      <div><label>Name</label><input id="name" placeholder="real-amazon-box"></div>
      <div><label>Should…</label>
        <select id="expect">
          <option value="detect">detect a package</option>
          <option value="no_detect">NOT detect</option>
        </select></div>
    </div>
    <label>Description</label><input id="description">
    <div class="row">
      <div><label>Detection mode</label>
        <select id="mode">
          <option value="background">background — continuous watch</option>
          <option value="person_gated">person-gated — compare before/after visits</option>
        </select></div>
      <div><label>fps</label><input id="fps" value="2.0"></div>
      <div><label>persist_samples</label><input id="persist" value="6"></div>
    </div>
    <div id="presenceRow">
      <label>Person present (clip time, e.g. <code>2:19-2:36, 0:08-0:14</code>)</label>
      <input id="presence" placeholder="auto-filled from Protect when pulling a clip">
      <div class="muted">Person-gated mode ignores everything while a person is
        in frame and compares the clean scene after each visit against the
        clean scene before it.</div>
    </div>

    <div id="regionDrawer">
      <label>Where should the package be? (detect only)</label>
      <div class="muted">Scrub the clip to when the box is on the ground,
        click <b>Draw box</b>, then drag a rectangle around it. The detector
        must fire inside this region to pass.</div>
      <div class="row tight" style="gap:6px;margin:6px 0">
        <button type="button" id="drawToggle" class="tab" onclick="toggleDraw()">✏️ Draw box</button>
        <button type="button" onclick="clearRegion()">Clear</button>
        <span class="muted" id="regionReadout"></span>
      </div>
      <div id="regionWrap" style="position:relative;line-height:0">
        <video id="rvid" preload="metadata" controls
               style="width:100%;display:block;border:1px solid var(--line);border-radius:6px"></video>
        <canvas id="rcanvas" style="position:absolute;left:0;top:0;pointer-events:none"></canvas>
      </div>
      <div class="muted" id="vidTime" style="text-align:right"></div>
      <input id="region" type="hidden">
    </div>

    <label>Detection time window (optional, detect only) — the detection only
      counts if it fires inside this range of clip time</label>
    <div class="row">
      <div><label>no earlier than</label><input id="after" placeholder="e.g. 2:30"></div>
      <div><label>no later than</label><input id="before" placeholder="leave blank = any"></div>
    </div>
  </div>

  <!-- STEP 3: VERIFY & SAVE -->
  <div class="wizstep" data-step="3" style="display:none">
    <div class="row" style="justify-content:space-between">
      <div class="muted">Runs the detector on this clip + expectation so you can
        confirm it grades the way you intend before saving.</div>
      <button onclick="verify()">↻ Re-run</button>
    </div>
    <div id="verifyOut" style="margin-top:8px"></div>
  </div>

  <div class="wiznav">
    <button id="btnBack" onclick="wizBack()">Back</button>
    <button id="btnNext" class="primary" onclick="wizNext()">Next</button>
    <button id="btnSave" class="primary" onclick="saveCase()" style="display:none">Save case</button>
    <span id="wizMsg" class="muted"></span>
  </div>
 </div>
</div>

<script>
async function j(url, opts){ const r = await fetch(url, opts); return r.json(); }
function msg(id, text){ document.getElementById(id).textContent = text; }
function wizMsg(t){ msg('wizMsg', t); }

// ---- case list + run all --------------------------------------------------
let caseIndex = {};   // name -> full case detail, for reopening/editing

async function loadCases(){
  const cases = await j('api/cases');
  caseIndex = {};
  const tb = document.querySelector('#cases tbody');
  tb.innerHTML = '';
  for(const c of cases){
    caseIndex[c.name] = c;
    const mode = (c.detector && c.detector.mode) || 'background';
    const tr = document.createElement('tr');
    tr.innerHTML = `<td><b>${c.name}</b><div class="reason">${c.description||''}</div></td>
      <td><span class="pill">${c.expect}</span> <span class="pill">${mode}</span></td>
      <td class="muted">${c.present?'':'⚠ clip missing'}</td>
      <td><button onclick="editCase('${c.name}')">open</button>
          <button onclick="preview('${c.name}')">preview</button></td>
      <td id="st-${cssId(c.name)}"></td>`;
    tb.appendChild(tr);
  }
}

function editCase(name){
  const c = caseIndex[name];
  if(!c) return;
  // Reopen the wizard on a saved case: clip playable, region drawn on the
  // video, every field editable. Saving overwrites the case by name.
  wiz.clip = c.clip;
  document.getElementById('clip').value = c.clip || '';
  document.getElementById('name').value = c.name;
  document.getElementById('expect').value = c.expect;
  document.getElementById('description').value = c.description || '';
  document.getElementById('fps').value = c.fps != null ? c.fps : '2.0';
  document.getElementById('persist').value =
    (c.detector && c.detector.persist_samples) || '6';
  document.getElementById('mode').value =
    (c.detector && c.detector.mode) || 'background';
  document.getElementById('presence').value =
    (c.presence || []).map(w => `${fmtTime(w[0])}-${fmtTime(w[1])}`).join(', ');
  document.getElementById('region').value =
    c.region ? c.region.join(' ') : '';
  document.getElementById('after').value = c.after != null ? fmtTime(c.after) : '';
  document.getElementById('before').value = c.before != null ? fmtTime(c.before) : '';
  renderWatch();
  step = 2;  // straight to the clip + region view; Back reaches the source
  showStep();
  document.getElementById('modal').style.display = 'flex';
  if(!c.present) wizMsg('clip file is missing on this machine — the case is '
    + 'editable but cannot be verified here');
}
function cssId(s){ return s.replace(/[^a-z0-9]/gi,'_'); }

async function runAll(){
  msg('summary','running…');
  const res = await j('api/run', {method:'POST'});
  let pass=0, fail=0, skip=0;
  for(const r of res){
    const cell = document.getElementById('st-'+cssId(r.name));
    if(cell){ cell.innerHTML = `<span class="${r.status}">${r.status.toUpperCase()}</span>
      <div class="reason">${r.reason||''}</div>`; }
    if(r.status==='pass')pass++; else if(r.status==='fail')fail++; else skip++;
  }
  msg('summary', `${pass} passed, ${fail} failed, ${skip} skipped`);
}

async function preview(name){
  const div = document.getElementById('preview');
  const bust = 't=' + Date.now();
  const url = k => `api/preview/${encodeURIComponent(name)}.png?${bust}&kind=${k}`;
  div.innerHTML = `<h2>Preview: ${name}</h2>
    <div class="row" style="align-items:flex-start">
      <div><div class="muted">detection</div><img class="preview" src="${url('detection')}"></div>
      <div><div class="muted">diff mask</div><img class="preview" src="${url('mask')}"></div>
    </div>
    <div class="muted">Full frame with no box means nothing was detected.</div>`;
}

// ---- wizard shell ---------------------------------------------------------
let wiz = { clip:null };
let step = 1;
function openWiz(){
  // Fresh case: clear anything left over from a previous add/edit session.
  wiz.clip = null;
  for(const id of ['clip','name','description','presence','region','after','before'])
    document.getElementById(id).value = '';
  document.getElementById('expect').value = 'detect';
  document.getElementById('mode').value = 'background';
  document.getElementById('fps').value = '2.0';
  document.getElementById('persist').value = '6';
  clearRegion(); renderWatch();
  step=1;
  if(document.getElementById('camera').options.length<=1) loadCameras();
  showStep();
  document.getElementById('modal').style.display='flex';
}
function closeWiz(){ document.getElementById('modal').style.display='none'; }
function showStep(){
  document.querySelectorAll('.wizstep').forEach(el =>
    el.style.display = (+el.dataset.step===step) ? '' : 'none');
  document.querySelectorAll('.steps span[data-s]').forEach(s =>
    s.classList.toggle('on', +s.dataset.s===step));
  document.getElementById('btnBack').style.visibility = step>1 ? 'visible' : 'hidden';
  document.getElementById('btnNext').style.display = step<3 ? '' : 'none';
  document.getElementById('btnSave').style.display = step===3 ? '' : 'none';
  wizMsg('');
  if(step===2) enterStep2();
  if(step===3) verify();
}
function wizBack(){ if(step>1){ step--; showStep(); } }
function wizNext(){
  if(step===1 && !wiz.clip){
    wizMsg('pull, upload, or reference a clip first'); return; }
  if(step===2 && !document.getElementById('name').value.trim()){
    wizMsg('give the case a name'); return; }
  if(step<3){ step++; showStep(); }
}
function srcTab(which){
  document.querySelectorAll('.srcpane').forEach(p =>
    p.style.display = (p.dataset.src===which) ? '' : 'none');
  document.querySelectorAll('#step1tabs .tab').forEach(b =>
    b.classList.toggle('active', b.dataset.src===which));
}

// ---- region drawer (step 2): scrub the clip, drag a box on the frame ------
let region = null, drawMode = false, drawStart = null;

function enterStep2(){
  const drawer = document.getElementById('regionDrawer');
  const canDraw = !!wiz.clip && document.getElementById('expect').value === 'detect';
  drawer.style.display = canDraw ? '' : 'none';
  if(!canDraw) return;
  const v = document.getElementById('rvid');
  if(v.getAttribute('src') !== wiz.clip){ v.src = wiz.clip; }
  parseRegionField();
  syncCanvas();
}
function parseRegionField(){
  const parts = (document.getElementById('region').value.trim().split(/\\s+/)).map(Number);
  region = (parts.length === 4 && parts.every(n => !isNaN(n)))
    ? {x:parts[0], y:parts[1], w:parts[2], h:parts[3]} : null;
  updateRegionReadout();
}
function syncCanvas(){
  const v = document.getElementById('rvid'), c = document.getElementById('rcanvas');
  c.width = v.clientWidth || v.offsetWidth; c.height = v.clientHeight || v.offsetHeight;
  drawRegion();
}
function toggleDraw(){
  drawMode = !drawMode;
  const c = document.getElementById('rcanvas'), v = document.getElementById('rvid');
  c.style.pointerEvents = drawMode ? 'auto' : 'none';
  c.style.cursor = 'crosshair';
  document.getElementById('drawToggle').classList.toggle('active', drawMode);
  if(drawMode) v.pause();
  syncCanvas();
}
function _pt(e){
  const c = document.getElementById('rcanvas'), r = c.getBoundingClientRect();
  return {x:(e.clientX-r.left), y:(e.clientY-r.top)};
}
function _drawRect(a, b, dash){
  const c = document.getElementById('rcanvas'), ctx = c.getContext('2d');
  ctx.clearRect(0,0,c.width,c.height);
  ctx.strokeStyle = '#2a9d3f'; ctx.lineWidth = 2;
  ctx.setLineDash(dash ? [6,4] : []);
  ctx.strokeRect(Math.min(a.x,b.x), Math.min(a.y,b.y),
                 Math.abs(a.x-b.x), Math.abs(a.y-b.y));
  ctx.setLineDash([]);
}
function drawRegion(){
  const c = document.getElementById('rcanvas'), ctx = c.getContext('2d');
  ctx.clearRect(0,0,c.width,c.height);
  if(!region) return;
  _drawRect({x:region.x*c.width, y:region.y*c.height},
            {x:(region.x+region.w)*c.width, y:(region.y+region.h)*c.height}, true);
}
function commitRegion(a, b){
  const c = document.getElementById('rcanvas');
  const x = Math.min(a.x,b.x)/c.width, y = Math.min(a.y,b.y)/c.height;
  const w = Math.abs(a.x-b.x)/c.width, h = Math.abs(a.y-b.y)/c.height;
  if(w < 0.01 || h < 0.01){ return; }
  region = {x,y,w,h};
  document.getElementById('region').value =
    `${x.toFixed(3)} ${y.toFixed(3)} ${w.toFixed(3)} ${h.toFixed(3)}`;
  updateRegionReadout(); drawRegion();
}
function clearRegion(){
  region = null; document.getElementById('region').value = '';
  updateRegionReadout();
  const c = document.getElementById('rcanvas');
  if(c.getContext) c.getContext('2d').clearRect(0,0,c.width,c.height);
}
function updateRegionReadout(){
  document.getElementById('regionReadout').textContent = region
    ? `region ${region.x.toFixed(2)}, ${region.y.toFixed(2)}, ${region.w.toFixed(2)}, ${region.h.toFixed(2)}`
    : 'no region set';
}
function initDrawer(){
  const c = document.getElementById('rcanvas');
  c.addEventListener('pointerdown', e => { if(drawMode){ drawStart = _pt(e); } });
  c.addEventListener('pointermove', e => { if(drawMode && drawStart){ _drawRect(drawStart, _pt(e), false); } });
  c.addEventListener('pointerup',  e => { if(drawMode && drawStart){ commitRegion(drawStart, _pt(e)); drawStart = null; } });
  const v = document.getElementById('rvid');
  v.addEventListener('loadeddata', syncCanvas);
  // Live readout in the same clip-time units the presence/window fields use.
  v.addEventListener('timeupdate', () => {
    document.getElementById('vidTime').textContent =
      `video at ${fmtTime(v.currentTime)}`;
  });
  window.addEventListener('resize', () => { if(step===2) syncCanvas(); });
}

// ---- source: cameras ------------------------------------------------------
async function loadCameras(){
  const res = await j('api/cameras');
  const sel = document.getElementById('camera');
  const note = document.getElementById('cameraNote');
  sel.innerHTML = '';
  if(!res.available){
    sel.innerHTML = `<option value="">unavailable: ${res.reason}</option>`;
    note.textContent = ''; return;
  }
  const bt = document.getElementById('btCam');
  bt.innerHTML = '';
  for(const cam of res.cameras){
    const o = document.createElement('option');
    o.value = cam.id; o.textContent = cam.name; sel.appendChild(o);
    bt.appendChild(o.cloneNode(true));
  }
  note.textContent = (res.source === 'homeassistant')
    ? `${res.cameras.length} camera(s) from Home Assistant. Recorded-clip scrub needs a Protect NVR — for these, upload or reference a clip.`
    : `${res.cameras.length} camera(s) from Unifi Protect${res.discovered ? ' (auto-discovered from the HA integration).' : '.'}`;
}

// ---- source: Protect scrubber (preview + timeline, frames fetched async) --
let scrub = null, seekTimer = null;
function startScrub(){
  const cam = document.getElementById('camera').value;
  const startLocal = document.getElementById('scrubStart').value;
  if(!cam){ wizMsg('pick a camera'); return; }
  if(!startLocal){ return; }
  const startMs = new Date(startLocal).getTime();
  const windowS = parseFloat(document.getElementById('scrubWindow').value);
  scrub = { cam, startMs, windowS, inS:null, outS:null };
  const tl = document.getElementById('timeline');
  tl.min = 0; tl.max = windowS; tl.step = 1; tl.value = 0;
  document.getElementById('scrubUI').style.display = '';
  updSel(); seek(0);
}
function atIso(off){ return new Date(scrub.startMs + off*1000).toISOString(); }
function curOff(){ return parseFloat(document.getElementById('timeline').value); }
function fmtClock(off){ return new Date(scrub.startMs + off*1000).toLocaleTimeString(); }
function onScrub(){
  const off = curOff();
  document.getElementById('frameTime').textContent = fmtClock(off);
  if(seekTimer) clearTimeout(seekTimer);
  seekTimer = setTimeout(() => { seekTimer = null; seek(off); }, 120);  // async, latest wins
}
function seek(off){
  const img = document.getElementById('frame');
  img.src = `api/snapshot?camera_id=${encodeURIComponent(scrub.cam)}&at=${encodeURIComponent(atIso(off))}&width=640`;
  document.getElementById('frameTime').textContent = fmtClock(off);
}
function nudge(d){
  if(!scrub) return;
  const tl = document.getElementById('timeline');
  let v = Math.max(0, curOff() + d);
  if(v > parseFloat(tl.max)) tl.max = v;   // allow scrubbing past the initial span
  tl.value = v; seek(v);
}
function markIn(){ if(!scrub) return; scrub.inS = curOff(); if(scrub.outS!=null && scrub.outS<=scrub.inS) scrub.outS=null; updSel(); }
function markOut(){ if(!scrub) return; scrub.outS = curOff(); if(scrub.inS!=null && scrub.outS<=scrub.inS) scrub.inS=null; updSel(); }
function updSel(){
  const f = s => s==null ? '—' : new Date(scrub.startMs + s*1000).toLocaleTimeString();
  let t = `in ${f(scrub && scrub.inS)} · out ${f(scrub && scrub.outS)}`;
  if(scrub && scrub.inS!=null && scrub.outS!=null) t += ` · length ${Math.round(scrub.outS-scrub.inS)}s`;
  document.getElementById('stripSel').textContent = t;
}
async function makeClip(){
  if(!scrub || scrub.inS==null || scrub.outS==null){ wizMsg('set In and Out first'); return; }
  wizMsg('pulling selected clip from Protect…');
  const res = await j('api/pull', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({camera_id: scrub.cam, start: atIso(scrub.inS), end: atIso(scrub.outS)})});
  if(res.error){ wizMsg('pull failed: '+res.error); return; }
  setClip(res.clip);
  // Protect also tells us when a person was in frame — pre-label the case
  // and suggest person-gated mode, the mode that ground truth exists for.
  if(res.presence && res.presence.length){
    document.getElementById('presence').value =
      res.presence.map(w => `${fmtTime(w[0])}-${fmtTime(w[1])}`).join(', ');
    document.getElementById('mode').value = 'person_gated';
    wizMsg(`clip ready — person in frame at ${res.presence.map(w=>fmtTime(w[0])+'–'+fmtTime(w[1])).join(', ')}; watch below, then Next`);
  } else {
    wizMsg('clip ready — watch it below to confirm, then Next'
      + (res.presence_error ? ` (person events unavailable: ${res.presence_error})` : ''));
  }
}

// ---- source: upload / reference / synthetic -------------------------------
async function uploadClip(){
  const f = document.getElementById('upload').files[0];
  if(!f) return;
  wizMsg('uploading…');
  const fd = new FormData(); fd.append('file', f);
  const res = await j('api/upload', {method:'POST', body: fd});
  if(res.error){ wizMsg(res.error); return; }
  setClip(res.clip); wizMsg('uploaded — watch it below to confirm, then Next');
}
function refClip(){ const v = document.getElementById('clip').value.trim(); if(v) setClip(v); }

function setClip(path){
  wiz.clip = path;
  document.getElementById('clip').value = path;
  renderWatch();
}
function renderWatch(){
  const el = document.getElementById('clipWatch');
  if(wiz.clip){
    el.innerHTML = `<div class="muted">Source clip — watch to confirm it's the right footage:</div>
      <video controls preload="metadata" src="${encodeURI(wiz.clip)}"></video>
      <div class="muted">${wiz.clip}</div>`;
  } else {
    el.innerHTML = `<div class="muted">No source chosen yet.</div>`;
  }
}

// ---- expectation + verify + save ------------------------------------------
// Times are shown mm:ss to match the video scrubber, stored as seconds.
function parseTime(s){
  s = (s||'').trim();
  if(!s) return null;
  const parts = s.split(':').map(Number);
  if(parts.some(isNaN)) return null;
  return parts.reduce((acc, p) => acc * 60 + p, 0);
}
function fmtTime(sec){
  const m = Math.floor(sec / 60), s = sec - m * 60;
  const whole = Math.floor(s), tenth = Math.round((s - whole) * 10);
  return `${m}:${String(whole).padStart(2,'0')}` + (tenth ? `.${tenth}` : '');
}

function parsePresence(text){
  // "2:19.5-2:36, 8-14" -> [[139.5,156],[8,14]]; null when empty/invalid.
  const windows = [];
  for(const part of text.split(',')){
    const halves = part.trim().split(/\\s*[-–]\\s*/);
    if(halves.length !== 2) continue;
    const a = parseTime(halves[0]), b = parseTime(halves[1]);
    if(a != null && b != null && b > a) windows.push([a, b]);
  }
  return windows.length ? windows : null;
}

function formCase(){
  const region = document.getElementById('region').value.trim();
  const mode = document.getElementById('mode').value;
  const det = { persist_samples: parseInt(document.getElementById('persist').value||'6') };
  if(mode !== 'background') det.mode = mode;
  return {
    name: document.getElementById('name').value,
    expect: document.getElementById('expect').value,
    description: document.getElementById('description').value,
    clip: (wiz.clip || document.getElementById('clip').value) || null,
    fps: parseFloat(document.getElementById('fps').value||'2'),
    detector: det,
    region: region ? region.split(/\\s+/).map(Number) : null,
    presence: parsePresence(document.getElementById('presence').value),
    after: parseTime(document.getElementById('after').value),
    before: parseTime(document.getElementById('before').value),
  };
}

async function verify(){
  const box = document.getElementById('verifyOut');
  box.innerHTML = '<div class="muted">running detector…</div>';
  let res;
  try { res = await j('api/preview_case', {method:'POST',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify(formCase())}); }
  catch(e){ box.innerHTML = `<div class="fail">error: ${e}</div>`; return; }
  if(res.error){ box.innerHTML = `<div class="fail">error: ${res.error}</div>`; return; }
  const cls = res.passed ? 'pass' : 'fail';
  const mark = res.passed ? '✓ grades as you expect' : '✗ does NOT grade as expected';
  const all = res.detections || [];
  const n = all.length;
  const cap = res.detection_time == null
    ? 'no detection — showing first frame'
    : res.matched
      ? `frame at the matching detection (t=${res.detection_time}s)`
      : `frame at first detection (t=${res.detection_time}s)`;
  // Show the first few detections, then summarize the rest.
  const shown = all.slice(0, 5).map(d =>
    `t=${d.t}s · (${d.bbox.join(', ')}) · conf=${d.confidence}`).join('<br>');
  const more = n > 5 ? `<br>…and ${n-5} more` : '';
  const legend = res.images.detection
    ? '<div class="muted"><span style="color:var(--ok)">■ detected</span> · <span style="color:#00c8ff">■ your expected region</span></div>'
    : '';
  let vline = '';
  if(res.verification){
    const v = res.verification;
    vline = v.error
      ? `<div class="muted">🔎 verifier error: ${v.error}</div>`
      : `<div class="${v.accepted ? 'pass' : 'fail'}">🔎 model sees: “${v.caption}” → ${v.accepted ? 'accepted' : 'rejected'} (${v.label})</div>`;
  }
  box.innerHTML = `<div class="${cls}"><b>${mark}</b> — ${res.reason}</div>
    ${vline}
    ${legend}
    <div class="row" style="align-items:flex-start;margin-top:8px">
      <div><div class="muted">${cap}</div>${res.images.detection?`<img class="preview" src="${res.images.detection}">`:'<div class="muted">—</div>'}</div>
      <div><div class="muted">diff mask</div>${res.images.mask?`<img class="preview" src="${res.images.mask}">`:'<div class="muted">—</div>'}</div>
    </div>
    <div class="muted" style="margin-top:6px">${n} detection(s)${n?':<br>'+shown+more:''}</div>`;
}

async function saveCase(){
  wizMsg('saving…');
  const res = await j('api/save_case', {method:'POST',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify(formCase())});
  if(res.error){ wizMsg('save failed: '+res.error); return; }
  wizMsg('saved '+res.saved);
  await loadCases();
  closeWiz();
}

// ---- per-camera watch zone ---------------------------------------------
// Canonical state is a normalized POLYGON: drawn rectangles become 4-corner
// polys, and zones imported from Protect keep their full shape.
let zonePoly = null, zoneDrag = null, protectZones = [];

function camName(){ const s = document.getElementById('btCam');
  return s.options[s.selectedIndex] ? s.options[s.selectedIndex].text : ''; }

async function openZone(){
  const cam = document.getElementById('btCam').value;
  if(!cam){ msg('btStatus','pick a camera first'); return; }
  msg('btStatus','');
  document.getElementById('zoneUI').style.display = '';
  const img = document.getElementById('zimg');
  img.onload = () => { syncZoneCanvas(); drawZone(); };
  img.onerror = () => msg('btStatus',
    'could not fetch a live snapshot from this camera — is Protect reachable?');
  // No `at` = live view; recorded footage at "now" doesn't exist yet.
  img.src = `api/snapshot?camera_id=${encodeURIComponent(cam)}&width=640&t=${Date.now()}`;
  const res = await j(`api/zone?camera_id=${encodeURIComponent(cam)}`);
  zonePoly = res.zone || null;
  updateZoneInfo(); syncZoneCanvas(); drawZone();
  loadProtectZones(cam);
}

async function loadProtectZones(cam){
  const sel = document.getElementById('pzones');
  sel.innerHTML = '<option value="">import from Protect…</option>';
  sel.onchange = () => {
    const i = parseInt(sel.value);
    if(!isNaN(i) && protectZones[i]){
      zonePoly = protectZones[i].points;
      drawZone(); updateZoneInfo();
      msg('btStatus', `imported Protect zone “${protectZones[i].name}” — Save zone to keep it`);
    }
  };
  const res = await j(`api/zone/protect?camera_id=${encodeURIComponent(cam)}`);
  if(res.error || !res.zones || !res.zones.length){
    sel.options[0].text = 'no Protect zones on this camera'; return;
  }
  protectZones = res.zones;
  res.zones.forEach((z, i) => {
    const o = document.createElement('option');
    o.value = i; o.textContent = `${z.name} (${z.kind}, ${z.points.length} pts)`;
    sel.appendChild(o);
  });
}

function syncZoneCanvas(){
  const img = document.getElementById('zimg'), c = document.getElementById('zcanvas');
  c.width = img.clientWidth; c.height = img.clientHeight;
}
function drawZone(){
  const c = document.getElementById('zcanvas'), ctx = c.getContext('2d');
  ctx.clearRect(0,0,c.width,c.height);
  if(!zonePoly || zonePoly.length < 3) return;
  // Dim everything outside the polygon (even-odd fill), outline the zone.
  ctx.beginPath();
  ctx.rect(0,0,c.width,c.height);
  ctx.moveTo(zonePoly[0][0]*c.width, zonePoly[0][1]*c.height);
  for(const [x,y] of zonePoly.slice(1)) ctx.lineTo(x*c.width, y*c.height);
  ctx.closePath();
  ctx.fillStyle = '#0008';
  ctx.fill('evenodd');
  ctx.beginPath();
  ctx.moveTo(zonePoly[0][0]*c.width, zonePoly[0][1]*c.height);
  for(const [x,y] of zonePoly.slice(1)) ctx.lineTo(x*c.width, y*c.height);
  ctx.closePath();
  ctx.strokeStyle = '#00c8ff'; ctx.lineWidth = 2;
  ctx.stroke();
}
function initZoneDrawer(){
  const c = document.getElementById('zcanvas');
  const pt = e => { const r = c.getBoundingClientRect();
    return {x:(e.clientX-r.left)/c.width, y:(e.clientY-r.top)/c.height}; };
  c.addEventListener('pointerdown', e => { zoneDrag = pt(e); });
  c.addEventListener('pointermove', e => {
    if(!zoneDrag) return;
    const p = pt(e);
    const x0 = Math.min(zoneDrag.x,p.x), y0 = Math.min(zoneDrag.y,p.y);
    const x1 = Math.max(zoneDrag.x,p.x), y1 = Math.max(zoneDrag.y,p.y);
    zonePoly = [[x0,y0],[x1,y0],[x1,y1],[x0,y1]];
    drawZone();
  });
  c.addEventListener('pointerup', () => { zoneDrag = null; updateZoneInfo(); });
}
function updateZoneInfo(){
  document.getElementById('zoneInfo').textContent = (zonePoly && zonePoly.length >= 3)
    ? `zone: ${zonePoly.length} points`
    : 'no zone — whole frame watched';
}
async function saveZone(){
  const cam = document.getElementById('btCam').value;
  if(!zonePoly || zonePoly.length < 3){
    msg('btStatus','drag a rectangle (or import a Protect zone) first'); return; }
  const res = await j('api/zone', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({camera_id: cam, camera_name: camName(),
                          poly: zonePoly})});
  if(res.error){ msg('btStatus', res.error); return; }
  msg('btStatus','watch zone saved — everything outside it is now ignored');
  updateZoneInfo();
}
async function clearZone(){
  const cam = document.getElementById('btCam').value;
  await j('api/zone', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({camera_id: cam, camera_name: camName(), rect: null})});
  zonePoly = null; drawZone(); updateZoneInfo();
  msg('btStatus','watch zone cleared — whole frame watched');
}

// ---- backtest ---------------------------------------------------------
let btTimer = null;

async function runBacktest(){
  const cam = document.getElementById('btCam').value;
  const day = document.getElementById('btDate').value;
  if(!cam || !day){ msg('btStatus','pick a camera and a day'); return; }
  const start = new Date(day + 'T00:00:00');
  const end = new Date(start.getTime() + 24*3600*1000);
  const res = await j('api/backtest', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({camera_id: cam,
      start: start.toISOString(), end: end.toISOString(),
      interval_s: parseFloat(document.getElementById('btInt').value),
      verify: !!document.getElementById('btVerify').value})});
  if(res.error){ msg('btStatus', res.error); return; }
  if(res.verify === false && document.getElementById('btVerify').value){
    msg('btStatus','note: verifier not configured on the server — CV only');
  }
  document.getElementById('btResults').innerHTML = '';
  if(btTimer) clearInterval(btTimer);
  btTimer = setInterval(pollBt, 2000);
  pollBt();
}

async function pollBt(){
  // Any failure here must land in btStatus, never die silently — a blank
  // results area with a happy summary is undebuggable from the UI.
  try{
    const s = await j('api/backtest/status');
    if(s.running){
      msg('btStatus', `scanning… sample ${s.progress[0]}/${s.progress[1]}`
        + (s.partial.length ? ` · ${s.partial.length} sample(s) with candidates so far` : ''));
      if(s.partial.length) renderHits(s.partial);   // play forward as it scans
      return;
    }
    clearInterval(btTimer); btTimer = null;
    if(s.error){ msg('btStatus','backtest failed: '+s.error); return; }
    if(s.result) renderBt(s.result);
  }catch(e){
    msg('btStatus', 'display error: ' + (e && e.message ? e.message : e));
    console.error('backtest render', e);
  }
}

function renderBt(r){
  msg('btStatus',
    `${r.samples_total} samples · ${r.samples_skipped_person} skipped (person)`
    + ` · ${r.samples_missing} missing · ${r.scene_flips} lighting flips`
    + ` · ${r.hits.length} sample(s) with candidates`);
  if(!r.hits.length){
    document.getElementById('btResults').innerHTML =
      '<div class="muted" style="margin-top:8px">no candidate packages found</div>';
    return;
  }
  renderHits(r.hits);
}

function renderHits(groups){
  const el = document.getElementById('btResults');
  const grid = document.createElement('div');
  grid.style.cssText = 'display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:12px;margin-top:10px';
  for(const g of groups){
    // One card per SAMPLE; a comparison can yield several candidate boxes,
    // numbered to match the labels drawn on the frame.
    const rows = g.boxes.map((b, i) => {
      const v = b.verification;
      const vtxt = !v ? '<span class="muted">no 2nd stage</span>'
        : v.error ? `<span class="muted">verifier error: ${v.error}</span>`
        : `<span class="${v.accepted?'pass':'fail'}">${v.accepted?'✓':'✗'} “${v.caption}”</span>`;
      return `<div class="muted" style="margin-top:4px"><b>#${i+1}</b>
        conf ${b.confidence} · (${b.bbox_norm.join(', ')})<br>${vtxt}</div>`;
    }).join('');
    const anyAccepted = g.boxes.some(b => b.verification && b.verification.accepted);
    const card = document.createElement('div');
    card.style.cssText = 'border:1px solid var(--line);border-radius:8px;padding:8px'
      + (g.boxes.some(b=>b.verification) && !anyAccepted ? ';opacity:.55' : '');
    // Show the comparison pair: the person-free baseline this sample was
    // diffed against, then the sample with the candidate boxes drawn.
    const beforeT = g.before_at ? new Date(g.before_at).toLocaleTimeString() : '';
    const pair = g.before_jpg
      ? `<div style="display:flex;gap:4px">
           <div style="flex:1"><div class="muted">before ${beforeT}</div>
             <img src="${g.before_jpg}" style="width:100%;border-radius:5px"></div>
           <div style="flex:1"><div class="muted">after ${new Date(g.at).toLocaleTimeString()}</div>
             <img src="${g.jpg}" style="width:100%;border-radius:5px"></div>
         </div>`
      : (g.jpg ? `<img src="${g.jpg}" style="width:100%;border-radius:5px">` : '');
    card.innerHTML = `${pair}
      <div><b>${new Date(g.at).toLocaleTimeString()}</b>
        <span class="muted">${g.boxes.length} candidate box(es)</span></div>
      ${rows}`;
    grid.appendChild(card);
  }
  // Build fully, then swap — a mid-render exception must not leave the
  // results area wiped.
  el.innerHTML = '';
  el.appendChild(grid);
}

srcTab('scrub');
initDrawer();
initZoneDrawer();
loadCases();
</script>
__LIVERELOAD__
</body></html>
"""
