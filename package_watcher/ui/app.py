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

import io
import os
from datetime import datetime
from typing import Any, Optional

import yaml

from ..config import UnifiConfig
from ..harness import FixtureCase, load_cases, run_and_evaluate, run_case


def create_app(fixtures_dir: str, unifi: Optional[UnifiConfig] = None):
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

    # --- pages ------------------------------------------------------------
    @app.get("/")
    def index() -> Any:
        # Under Home Assistant ingress the page is served from a token-prefixed
        # path (e.g. /api/hassio_ingress/<token>/); HA passes that prefix in
        # X-Ingress-Path. Emitting it as the document <base> lets every
        # relative fetch/img URL below resolve correctly both there and when
        # run standalone (prefix empty -> base "/").
        prefix = request.headers.get("X-Ingress-Path", "").rstrip("/")
        return _PAGE.replace("__INGRESS_BASE__", prefix)

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
            img = frames.get("detection") or frames.get("first")
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
        # Prefer a configured Protect NVR: it can pull recorded clips by time
        # range. Otherwise fall back to discovering camera.* entities straight
        # from Home Assistant (works out of the box in the add-on, no
        # credentials), which lists the cameras but can't fetch past footage.
        if unifi is not None and protect.available():
            try:
                return jsonify({"available": True, "source": "protect",
                                "supports_pull": True,
                                "cameras": protect.list_cameras(unifi)})
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
        if unifi is not None and not protect.available():
            return jsonify({"available": False,
                            "reason": "uiprotect not installed"})
        return jsonify({"available": False,
                        "reason": "no unifi block and no Home Assistant API "
                                  "(run as an add-on with homeassistant_api)"})

    @app.post("/api/pull")
    def api_pull() -> Any:
        from . import protect
        if unifi is None or not protect.available():
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
            protect.pull_clip(unifi, data["camera_id"], start, end, out)
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

    return app


# --- helpers --------------------------------------------------------------

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
          host: str, port: int) -> None:
    app = create_app(fixtures_dir, unifi)
    print(f"package-watcher UI on http://{host}:{port}  "
          f"(fixtures: {os.path.abspath(fixtures_dir)})")
    app.run(host=host, port=port)


# --- embedded single-page UI ----------------------------------------------

_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<base href="__INGRESS_BASE__/">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>package-watcher · fixtures</title>
<style>
  :root { color-scheme: light dark; --line:#8884; --ok:#2a9d3f; --bad:#d13232;
          --skip:#9a7b1a; }
  body { font-family: system-ui, sans-serif; margin: 0; line-height: 1.45; }
  header { padding: 12px 20px; border-bottom: 1px solid var(--line);
           display:flex; gap:16px; align-items:center; }
  header h1 { font-size: 16px; margin: 0; font-weight: 650; }
  main { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; padding: 20px; }
  @media (max-width: 900px){ main{ grid-template-columns:1fr; } }
  h2 { font-size: 13px; text-transform: uppercase; letter-spacing:.05em;
       opacity:.7; margin: 0 0 10px; }
  button { font: inherit; padding: 6px 12px; border:1px solid var(--line);
           border-radius: 7px; background: transparent; cursor: pointer; }
  button.primary { background:#2f6fed; color:#fff; border-color:#2f6fed; }
  table { border-collapse: collapse; width: 100%; font-size: 14px; }
  td, th { text-align:left; padding: 6px 8px; border-bottom:1px solid var(--line); }
  .pill { font-size:12px; padding:2px 8px; border-radius:999px; border:1px solid var(--line); }
  .pass{ color:var(--ok); } .fail{ color:var(--bad); } .skip{ color:var(--skip); }
  .reason { font-size:12px; opacity:.75; }
  label { display:block; font-size:13px; margin:8px 0 3px; opacity:.85; }
  input, select, textarea { font: inherit; padding:6px 8px; width:100%;
    box-sizing:border-box; border:1px solid var(--line); border-radius:6px;
    background:transparent; }
  fieldset { border:1px solid var(--line); border-radius:8px; margin:10px 0; }
  .row { display:flex; gap:10px; } .row > * { flex:1; }
  img.preview { max-width:100%; border:1px solid var(--line); border-radius:6px;
                margin-top:8px; }
  .muted { opacity:.6; font-size:12px; }
</style></head>
<body>
<header>
  <h1>📦 package-watcher fixtures</h1>
  <button class="primary" onclick="runAll()">Run all</button>
  <span id="summary" class="muted"></span>
</header>
<main>
  <section>
    <h2>Fixture cases</h2>
    <table id="cases"><tbody></tbody></table>
    <div id="preview"></div>
  </section>
  <section>
    <h2>Add a case</h2>
    <fieldset>
      <legend>Source</legend>
      <label>Protect camera (pull a recorded clip)</label>
      <div class="row">
        <select id="camera"><option value="">— none / unavailable —</option></select>
        <button onclick="loadCameras()">Refresh</button>
      </div>
      <div id="cameraNote" class="muted"></div>
      <div class="row">
        <div><label>Begin (ISO)</label><input id="start" placeholder="2026-07-04T14:03:00"></div>
        <div><label>End (ISO)</label><input id="end" placeholder="2026-07-04T14:04:30"></div>
      </div>
      <button onclick="pullClip()">Pull clip from Protect</button>
      <label>…or upload a video file</label>
      <input type="file" id="upload" accept="video/*" onchange="uploadClip()">
      <label>…or reference an existing clip / leave blank for synthetic</label>
      <input id="clip" placeholder="clips/my-clip.mp4">
      <label>…or a synthetic scene (JSON)</label>
      <input id="scene" placeholder='{"scene":"package","hold_s":16}'>
    </fieldset>
    <fieldset>
      <legend>Expectation</legend>
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
      <label>Expected region (detect only) — x y w h, normalized 0–1</label>
      <input id="region" placeholder="0.35 0.55 0.30 0.35">
      <div class="row">
        <div><label>After (s)</label><input id="after"></div>
        <div><label>Before (s)</label><input id="before"></div>
      </div>
    </fieldset>
    <button onclick="preview()">Preview detection</button>
    <button class="primary" onclick="saveCase()">Save case</button>
    <div id="saveMsg" class="muted"></div>
  </section>
</main>
<script>
async function j(url, opts){ const r = await fetch(url, opts); return r.json(); }

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
  document.getElementById('summary').textContent = 'running…';
  const res = await j('api/run', {method:'POST'});
  let pass=0, fail=0, skip=0;
  for(const r of res){
    const cell = document.getElementById('st-'+cssId(r.name));
    if(cell){ cell.innerHTML = `<span class="${r.status}">${r.status.toUpperCase()}</span>
      <div class="reason">${r.reason||''}</div>`; }
    if(r.status==='pass')pass++; else if(r.status==='fail')fail++; else skip++;
  }
  document.getElementById('summary').textContent =
    `${pass} passed, ${fail} failed, ${skip} skipped`;
}

async function loadCameras(){
  const res = await j('api/cameras');
  const sel = document.getElementById('camera');
  const note = document.getElementById('cameraNote');
  sel.innerHTML = '';
  if(!res.available){
    sel.innerHTML = `<option value="">unavailable: ${res.reason}</option>`;
    note.textContent = '';
    return;
  }
  for(const cam of res.cameras){
    const o = document.createElement('option');
    o.value = cam.id; o.textContent = cam.name;
    sel.appendChild(o);
  }
  if(res.source === 'homeassistant'){
    note.textContent = `${res.cameras.length} camera(s) from Home Assistant. `
      + `Recorded-clip pull needs a Protect (unifi) config block — for HA `
      + `cameras, upload a file or reference an existing clip instead.`;
  } else {
    note.textContent = `${res.cameras.length} camera(s) from Unifi Protect.`;
  }
}

async function pullClip(){
  const body = { camera_id: document.getElementById('camera').value,
    start: document.getElementById('start').value,
    end: document.getElementById('end').value };
  msg('saveMsg','pulling clip from Protect…');
  const res = await j('api/pull', {method:'POST',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  if(res.error){ msg('saveMsg','pull failed: '+res.error); return; }
  document.getElementById('clip').value = res.clip;
  msg('saveMsg','pulled '+res.clip);
}

async function uploadClip(){
  const f = document.getElementById('upload').files[0];
  if(!f) return;
  const fd = new FormData(); fd.append('file', f);
  const res = await j('api/upload', {method:'POST', body: fd});
  if(res.error){ msg('saveMsg', res.error); return; }
  document.getElementById('clip').value = res.clip;
  msg('saveMsg','uploaded '+res.clip);
}

function formCase(){
  const region = document.getElementById('region').value.trim();
  const scene = document.getElementById('scene').value.trim();
  const c = {
    name: document.getElementById('name').value,
    expect: document.getElementById('expect').value,
    description: document.getElementById('description').value,
    clip: document.getElementById('clip').value || null,
    scene: scene ? JSON.parse(scene) : null,
    fps: parseFloat(document.getElementById('fps').value||'2'),
    detector: { persist_samples: parseInt(document.getElementById('persist').value||'6') },
    region: region ? region.split(/\\s+/).map(Number) : null,
    after: document.getElementById('after').value || null,
    before: document.getElementById('before').value || null,
  };
  return c;
}

async function saveCase(){
  const res = await j('api/save_case', {method:'POST',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify(formCase())});
  if(res.error){ msg('saveMsg','save failed: '+res.error); return; }
  msg('saveMsg','saved '+res.saved);
  loadCases();
}

async function preview(name){
  // If a name is given, preview that saved case. Otherwise save-then-preview
  // is overkill; previewing a saved case is the common path.
  const target = name || document.getElementById('name').value;
  if(!target){ msg('saveMsg','name a case (or click preview on a row)'); return; }
  const div = document.getElementById('preview');
  const bust = 't=' + Date.now();
  const url = k => `api/preview/${encodeURIComponent(target)}.png?${bust}&kind=${k}`;
  div.innerHTML = `<h2>Preview: ${target}</h2>
    <div class="row">
      <div><div class="muted">detection</div>
        <img class="preview" src="${url('detection')}"></div>
      <div><div class="muted">diff mask</div>
        <img class="preview" src="${url('mask')}"></div>
    </div>
    <div class="muted">If the detection image shows the full frame with no box,
      nothing was detected for this case.</div>`;
}

function msg(id, text){ document.getElementById(id).textContent = text; }
loadCases(); loadCameras();
</script>
</body></html>
"""
