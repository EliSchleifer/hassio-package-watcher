# Home Assistant add-on wrapper around the package-watcher service.
#
# HA builds a repository add-on with THIS directory as the build context, so
# the image can't COPY repo-root files (pyproject.toml, package_watcher/, ...).
# Instead we install the package straight from Git — the add-on repo is this
# same GitHub repo, so `@main` tracks the code the add-on ships with.
#
# Uses a Debian python base (not the Alpine HA base) because opencv wheels
# need glibc; `init: false` in config.yaml means no s6 overlay is required.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 jq git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "package-watcher[unifi,ui] @ git+https://github.com/EliSchleifer/hassio-package-watcher@main"

COPY run.sh /run.sh
RUN chmod +x /run.sh
CMD ["/run.sh"]
