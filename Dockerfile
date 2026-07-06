FROM python:3.12-slim

# ffmpeg supplies the RTSP demuxing/decoding OpenCV uses under the hood.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY package_watcher ./package_watcher
RUN pip install --no-cache-dir ".[unifi]"

VOLUME /data
ENV PW_CONFIG=/data/config.yaml

CMD ["sh", "-c", "package-watcher run --config $PW_CONFIG"]
