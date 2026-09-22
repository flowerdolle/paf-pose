# PAF-Pose backend: MediaPipe Tasks (CPU). Build context = backends/ (see docker-compose.yml).
#   docker build -f mediapipe/Dockerfile -t pafpose/mediapipe:0.1 .
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MEDIAPIPE_MODEL_DIR=/app/models

# mediapipe 0.10.35 dlopens libGLESv2/libEGL at import; opencv-contrib-python needs libGL / glib.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libgles2 libegl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# Versions pinned to the environment used for the ICCAS 2026 paper (conda env mediapipe_iccas).
RUN pip install \
        mediapipe==0.10.35 \
        numpy==1.26.4 \
        opencv-contrib-python==4.11.0.86

# Model files (MediaPipe Tasks, float16, version 1 = files dated Apr/May 2023 used in the paper).
RUN mkdir -p /app/models && cd /app/models \
    && curl -fsSL -o pose_landmarker_heavy.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task \
    && curl -fsSL -o hand_landmarker.task       https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task \
    && curl -fsSL -o face_landmarker.task       https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

# Static ffmpeg for frame decoding (see pafpose_backend.VideoReader).
RUN pip install --no-cache-dir imageio-ffmpeg==0.6.0
COPY backends/_common/pafpose_backend.py /app/
COPY backends/mediapipe/adapter.py backends/mediapipe/to_common.py /app/

WORKDIR /app
RUN mkdir -p /input /output && python -c "import mediapipe, cv2; print(mediapipe.__version__, cv2.__version__)"
ENTRYPOINT []
CMD ["python", "/app/adapter.py", "--help"]
