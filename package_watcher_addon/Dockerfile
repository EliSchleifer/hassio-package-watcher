# Home Assistant add-on wrapper around the package-watcher service.
# Uses a Debian python base (not the Alpine HA base) because opencv wheels
# need glibc; `init: false` in config.yaml means no s6 overlay is required.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 jq \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md config.example.yaml ./
COPY package_watcher ./package_watcher
RUN pip install --no-cache-dir ".[unifi,ui]"

COPY package_watcher_addon/run.sh /run.sh
RUN chmod +x /run.sh
CMD ["/run.sh"]
