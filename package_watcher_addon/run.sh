#!/bin/sh
set -e

CONFIG_PATH=$(jq -r '.config_path // "/config/config.yaml"' /data/options.json)

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
    echo "Edit $CONFIG_PATH (add-on configuration folder) and restart the add-on." >&2
    exit 1
fi

exec package-watcher run --config "$CONFIG_PATH"
