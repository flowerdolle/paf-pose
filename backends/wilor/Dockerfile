# PAF-Pose backend image: WiLoR hand estimator via the WiLoR-mini package.
# Build from the repository root with the `backends/` directory as context:
#   docker build -f backends/wilor/Dockerfile -t pafpose/wilor:0.1 backends
# (or `docker compose build wilor`).
#
# Environment mirrors the paper runs: torch 2.5.0 + CUDA 12.4, wilor_mini 1.1 @ ebec42f,
# ultralytics 8.1.34, smplx 0.1.28. Weights are NOT baked in; mount them at /weights.

FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime

ARG WILOR_MINI_REPO=https://github.com/warmshao/WiLoR-mini.git
ARG WILOR_MINI_COMMIT=ebec42f94c389070cdd7dda6fd1bf0b4a659c960
ARG CHUMPY_COMMIT=580566eafc9ac68b2614b64d6f7aaa84eebb70da

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    WILOR_ROOT=/opt/wilor \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates libgl1 libglib2.0-0 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Pinned upstream checkout (never vendored into this repository).
RUN git clone ${WILOR_MINI_REPO} /opt/wilor \
    && git -C /opt/wilor checkout --quiet ${WILOR_MINI_COMMIT}

# Dependencies pinned to the versions used for the paper; torch/torchvision come from the base image.
RUN pip install \
        "numpy==2.2.6" \
        "opencv-python==4.13.0.92" \
        "ultralytics==8.1.34" \
        "smplx==0.1.28" \
        "timm==1.0.26" \
        "einops==0.8.2" \
        "scikit-image==0.25.2" \
        "roma==1.5.6" \
        "huggingface_hub==1.10.1" \
        "tqdm==4.67.3" \
        "dill==0.4.1" \
        "chumpy @ git+https://github.com/mattloper/chumpy@${CHUMPY_COMMIT}" \
    && pip install --no-deps /opt/wilor \
    && python -c "import wilor_mini, ultralytics, smplx, chumpy, dill; print('wilor_mini ok')"

WORKDIR /app
COPY backends/_common/pafpose_backend.py /app/pafpose_backend.py
COPY backends/wilor/adapter.py backends/wilor/to_common.py /app/

# Writable caches for `docker run --user <uid>:<gid>`.
RUN mkdir -p /tmp/ultralytics /tmp/matplotlib && chmod 1777 /tmp/ultralytics /tmp/matplotlib

VOLUME ["/input", "/output", "/weights"]
CMD ["python", "/app/adapter.py", "--help"]
