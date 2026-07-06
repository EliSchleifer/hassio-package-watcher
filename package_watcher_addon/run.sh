#!/bin/sh
set -e

CONFIG_PATH=$(jq -r '.config_path // "/config/config.yaml"' /data/options.json)
FIXTURES_DIR=$(jq -r '.fixtures_path // "/config/fixtures"' /data/options.json)
UI_PORT=8099

if [ ! -f "$CONFIG_PATH" ]; then
    echo "No config found at $CONFIG_PATH — writing a starter config." >&2
    mkdir -p "$(dirname "$CONFIG_PATH")"
    cp /app/config.example.yaml "$CONFIG_PATH" 2>/dev/null || cat > "$CONFIG_PATH" <<'EOF'
events_dir: /config/events
cameras:
  - name: front-door
    source: rtsps://REPLACE_ME:7441/STREAM_TOKEN?enableSrtp
sinks:
  stdout: true
  jsonl_path: /config/events/events.jsonl
EOF
    echo "Edit $CONFIG_PATH (add-on configuration folder), add your camera(s)," >&2
    echo "then restart the add-on to begin watching." >&2
fi

mkdir -p "$FIXTURES_DIR"

# Start the detection service in the background — but only once real cameras
# are configured. On a fresh install the starter config still holds the
# REPLACE_ME placeholder; starting the watcher then would just spew RTSP
# connection errors, so we hold off and let the Web UI come up regardless.
if grep -q REPLACE_ME "$CONFIG_PATH"; then
    echo "Config still contains a placeholder camera — not starting the" >&2
    echo "watcher yet. Use the Web UI to author fixtures, edit $CONFIG_PATH," >&2
    echo "then restart." >&2
else
    echo "Starting package watcher against $CONFIG_PATH" >&2
    package-watcher run --config "$CONFIG_PATH" &
fi

# The Web UI is the ingress entry point and keeps the add-on's container
# alive. Bind to all interfaces so Home Assistant's ingress proxy can reach
# it; ingress rewrites URLs via the X-Ingress-Path header (handled in the app).
exec package-watcher ui \
    --config "$CONFIG_PATH" \
    --fixtures "$FIXTURES_DIR" \
    --host 0.0.0.0 \
    --port "$UI_PORT"
