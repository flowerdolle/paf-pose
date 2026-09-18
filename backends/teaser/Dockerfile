# PAF-Pose backend: TEASER (face70). Build context is backends/ :
#   docker build -f teaser/Dockerfile -t pafpose/teaser:0.1 backends/
# Upstream: https://github.com/Pixel-Talk/TEASER (Python 3.9 / torch 2.0.1 / CUDA 11.7 per its README).
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TEASER_ROOT=/opt/teaser \
    TEASER_COMMIT=c235716 \
    TORCH_HOME=/opt/cache/torch \
    HF_HOME=/opt/cache/hf \
    MPLCONFIGDIR=/tmp/mpl

RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget ca-certificates libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Upstream checkout pinned to the commit used in the paper.
RUN git clone https://github.com/Pixel-Talk/TEASER.git /opt/teaser \
    && cd /opt/teaser && git checkout -q c235716 && rm -rf .git

# Dependencies pinned from upstream requirements.txt (torch/torchvision come from the base image;
# pytorch3d, pytorch_lightning, albumentations, fvcore/iopath and gdown are training/render-only).
RUN pip install --no-cache-dir \
        numpy==1.22.4 \
        mediapipe==0.10.10 \
        omegaconf==2.3.0 \
        opencv-python-headless==4.9.0.80 \
        scikit-image==0.22.0 \
        scikit-learn==1.3.2 \
        timm==0.9.16 \
        tqdm==4.66.2 \
        chumpy==0.70

# Public detector asset (not license-gated).
RUN wget -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task \
        -O /opt/teaser/assets/face_landmarker.task

# Warm the timm backbone cache so the container never needs network at run time.
RUN python - <<'PY'
import sys; sys.path.insert(0, "/opt/teaser")
from src.teaser_encoder import TeaserEncoder
TeaserEncoder()
PY

# License-gated FLAME model is mounted at run time; point the relative asset path at /weights.
RUN rm -rf /opt/teaser/assets/FLAME2020 && ln -s /weights/FLAME2020 /opt/teaser/assets/FLAME2020 \
    && mkdir -p /opt/teaser/pretrained_models && ln -s /weights/TEASER.pt /opt/teaser/pretrained_models/TEASER.pt \
    && chmod -R a+rX /opt/teaser /opt/cache

COPY backends/_common/pafpose_backend.py /app/pafpose_backend.py
COPY backends/teaser/adapter.py backends/teaser/to_common.py /app/
RUN chmod -R a+rX /app && mkdir -p /output /tmp/mpl && chmod 1777 /output /tmp/mpl

WORKDIR /opt/teaser
ENTRYPOINT []
CMD ["python", "/app/adapter.py", "--help"]
