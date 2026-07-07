"""Flask app for authoring and running fixture cases.

Workflow the UI supports:
  1. Point at a Protect camera and pull a clip by begin/end time (or upload
     a local video, or reference one already in fixtures/clips/).
  2. Label it "should detect a package" or "should NOT", optionally draw an
     expected region and set a time window.
  3. Preview: run the detector on the clip and see what it found (annotated
     frame, baseline, diff mask) so you can confirm/tune before saving.
  4. Save it into fixtures/cases.yaml — where `pytest` / `package-watcher
     test` will grade it from then on.

Single-file app with embedded templates so it has no asset build step.
"""

from __future__ import annotations

import base64
import io
import os
from datetime import datetime
from typing import Any, Optional

import yaml

from ..config import UnifiConfig
from ..harness import FixtureCase, load_cases, run_and_evaluate, run_case


def create_app(fixtures_dir: str, unifi: Optional[UnifiConfig] = None,
               reload: bool = False):
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
    os.makedirs(clips_dir, exist_ok=True)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512 MB uploads

    # Live-reload (dev): the page polls /__alive; when the process restarts
    # (Flask's reloader on file save), the boot id changes and the browser
    # reloads itself — edit -> save -> refreshed page, no manual F5.
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
            present = True
            if c.clip:
                p = (c.clip if os.path.isabs(c.clip)
                     else os.path.join(fixtures_dir, c.clip))
                present = os.path.isfile(p)
            out.append({
                "name": c.name, "expect": c.expect,
                "source": "clip" if c.clip else "synthetic",
                "clip": c.clip, "description": c.description,
                "present": present,
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
        result = run_case(case, fixtures_dir, capture_preview=True)
        frames = result.frames_for_preview
        img = frames.get(kind)
        if img is None:
            img = frames.get("detection")
        if img is None:
            img = frames.get("first")
        if img is None:
            return Response("no preview frame", status=404)
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
        if not camera_id or not at:
            return Response("camera_id and at required", status=400)
        try:
            dt = datetime.fromisoformat(at)
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
        return jsonify({"clip": f"clips/{fname}"})

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
        case = FixtureCase(**entry)
        try:
            outcome = run_and_evaluate(case, fixtures_dir, capture_preview=True)
        except FileNotFoundError as exc:
            return jsonify({"error": f"clip not found: {exc}"}), 400
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc)}), 500
        frames = outcome.result.frames_for_preview
        det = frames.get("detection")
        if det is None:
            det = frames.get("first")
        # Overlay the expected region (cyan) next to the detector's own box
        # (green) so "right thing, right place?" is answerable at a glance.
        if det is not None and case.region is not None:
            det = det.copy()
            dh, dw = det.shape[:2]
            rx, ry, rw, rh = case.region
            cv2.rectangle(det, (int(rx * dw), int(ry * dh)),
                          (int((rx + rw) * dw), int((ry + rh) * dh)),
                          (255, 200, 0), 2)
        dets = outcome.result.detections
        return jsonify({
            "passed": outcome.passed,
            "reason": outcome.reason,
            "expect": case.expect,
            "detection_time": round(dets[0].t, 1) if dets else None,
            "detections": [
                {"t": round(d.t, 1),
                 "bbox": [round(v, 3) for v in d.bbox_norm],
                 "confidence": round(d.confidence, 3)}
                for d in dets],
            "images": {
                "detection": _png_data_uri(det),
                "mask": _png_data_uri(frames.get("mask")),
            },
        })

    return app


# --- helpers --------------------------------------------------------------

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
    elif data.get("scene"):
        case["scene"] = data["scene"]
    else:
        raise ValueError("a clip or a synthetic scene is required")
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
    # Validate it constructs before we persist it.
    entry = dict(case)
    if "region" in entry:
        entry["region"] = tuple(entry["region"])
    if "zone" in entry:
        entry["zone"] = [tuple(pt) for pt in entry["zone"]]
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
          host: str, port: int, reload: bool = False) -> None:
    app = create_app(fixtures_dir, unifi, reload=reload)
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
      <button class="tab" data-src="ref" onclick="srcTab('ref')">Reference / synthetic</button>
    </div>

    <div class="srcpane" data-src="scrub">
      <div class="row">
        <select id="camera"><option value="">— none / unavailable —</option></select>
        <button onclick="loadCameras()">Refresh</button>
      </div>
      <div id="cameraNote" class="muted"></div>
      <div class="row">
        <div><label>Start</label>
          <input type="datetime-local" id="scrubStart" onchange="startScrub()"></div>
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
      <label>…or a synthetic scene (JSON) — leave clip blank</label>
      <input id="scene" placeholder='{"scene":"package","hold_s":16}' oninput="onScene()">
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
      <div><label>fps</label><input id="fps" value="2.0"></div>
      <div><label>persist_samples</label><input id="persist" value="6"></div>
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
      <input id="region" type="hidden">
    </div>

    <div class="row">
      <div><label>After (s)</label><input id="after"></div>
      <div><label>Before (s)</label><input id="before"></div>
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
async function loadCases(){
  const cases = await j('api/cases');
  const tb = document.querySelector('#cases tbody');
  tb.innerHTML = '';
  for(const c of cases){
    const tr = document.createElement('tr');
    tr.innerHTML = `<td><b>${c.name}</b><div class="reason">${c.description||''}</div></td>
      <td><span class="pill">${c.expect}</span></td>
      <td class="muted">${c.source}${c.present?'':' ⚠ missing'}</td>
      <td><button onclick="preview('${c.name}')">preview</button></td>
      <td id="st-${cssId(c.name)}"></td>`;
    tb.appendChild(tr);
  }
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
function openWiz(){ step=1; if(document.getElementById('camera').options.length<=1) loadCameras(); showStep(); document.getElementById('modal').style.display='flex'; }
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
  if(step===1 && !wiz.clip && !sceneVal()){
    wizMsg('pull, upload, or reference a clip first (or enter a synthetic scene)'); return; }
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
  document.getElementById('rvid').addEventListener('loadeddata', syncCanvas);
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
  for(const cam of res.cameras){
    const o = document.createElement('option');
    o.value = cam.id; o.textContent = cam.name; sel.appendChild(o);
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
  setClip(res.clip); wizMsg('clip ready — watch it below to confirm, then Next');
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
function onScene(){ wiz.clip = null; renderWatch(); }
function sceneVal(){ return document.getElementById('scene').value.trim(); }

function setClip(path){
  wiz.clip = path;
  document.getElementById('clip').value = path;
  document.getElementById('scene').value = '';
  renderWatch();
}
function renderWatch(){
  const el = document.getElementById('clipWatch');
  if(wiz.clip){
    el.innerHTML = `<div class="muted">Source clip — watch to confirm it's the right footage:</div>
      <video controls preload="metadata" src="${encodeURI(wiz.clip)}"></video>
      <div class="muted">${wiz.clip}</div>`;
  } else if(sceneVal()){
    el.innerHTML = `<div class="muted">Synthetic scene — nothing to watch; verify on step 3.</div>`;
  } else {
    el.innerHTML = `<div class="muted">No source chosen yet.</div>`;
  }
}

// ---- expectation + verify + save ------------------------------------------
function formCase(){
  const region = document.getElementById('region').value.trim();
  const scene = sceneVal();
  return {
    name: document.getElementById('name').value,
    expect: document.getElementById('expect').value,
    description: document.getElementById('description').value,
    clip: (wiz.clip || document.getElementById('clip').value) || null,
    scene: scene ? JSON.parse(scene) : null,
    fps: parseFloat(document.getElementById('fps').value||'2'),
    detector: { persist_samples: parseInt(document.getElementById('persist').value||'6') },
    region: region ? region.split(/\\s+/).map(Number) : null,
    after: document.getElementById('after').value || null,
    before: document.getElementById('before').value || null,
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
  const cap = res.detection_time != null
    ? `frame at first detection (t=${res.detection_time}s)`
    : 'no detection — showing first frame';
  // Show the first few detections, then summarize the rest.
  const shown = all.slice(0, 5).map(d =>
    `t=${d.t}s · (${d.bbox.join(', ')}) · conf=${d.confidence}`).join('<br>');
  const more = n > 5 ? `<br>…and ${n-5} more` : '';
  const legend = res.images.detection
    ? '<div class="muted"><span style="color:var(--ok)">■ detected</span> · <span style="color:#00c8ff">■ your expected region</span></div>'
    : '';
  box.innerHTML = `<div class="${cls}"><b>${mark}</b> — ${res.reason}</div>
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

srcTab('scrub');
initDrawer();
loadCases();
</script>
__LIVERELOAD__
</body></html>
"""
