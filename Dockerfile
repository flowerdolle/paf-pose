# syntax=docker/dockerfile:1.6
# PAF-Pose: every backend in ONE image.
#
#   docker build -t pafpose/all:0.1 .          (or: docker compose -f docker-compose.single.yml build)
#   PAFPOSE_REGISTRY=backends/backends-single.yaml pafpose run ...
#
# The five backends need four Python versions and three torch/CUDA builds, so they cannot share
# one site-packages. Each backend is installed into its own virtualenv in its own build stage,
# and the final stage collects the virtualenvs, the pinned upstream checkouts and the adapters.
# torch wheels bundle their CUDA runtime, so cu117 / cu118 / cu121 / cu124 coexist; the host only
# needs a driver with CUDA 12.x minor-version compatibility (driver 535 verified) and the NVIDIA
# Container Toolkit. Weights are NOT baked in; mount them at /weights (see README.md).
#
# Inside the container:  pafpose-backend <sam3dbody|pear|wilor|teaser|mediapipe> <adapter args>
#
# Pinned versions, commits and dependency lists are copied verbatim from backends/*/Dockerfile,
# which remain the per-backend images and the reference for every version below.

ARG CUDA_DEVEL=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ARG CUDA_RUNTIME=nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# ---------------------------------------------------------------------------------------------
# Common build base: four interpreters + build tools. Every venv below is created from one of
# these interpreters at its Ubuntu path, so the same packages in the final stage make the copied
# venvs work unchanged.
# ---------------------------------------------------------------------------------------------
FROM ${CUDA_DEVEL} AS pybase
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl wget git build-essential ninja-build \
        libgl1 libgles2 libegl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.9  python3.9-dev  python3.9-venv  python3.9-distutils \
        python3.10 python3.10-dev python3.10-venv python3.10-distutils \
        python3.11 python3.11-dev python3.11-venv python3.11-distutils \
        python3.12 python3.12-dev python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------------------------
# WiLoR (hand): Python 3.11, torch 2.5.0 + cu124, WiLoR-mini @ ebec42f  (backends/wilor/Dockerfile)
# ---------------------------------------------------------------------------------------------
FROM pybase AS wilor
ARG WILOR_MINI_REPO=https://github.com/warmshao/WiLoR-mini.git
ARG WILOR_MINI_COMMIT=ebec42f94c389070cdd7dda6fd1bf0b4a659c960
ARG CHUMPY_COMMIT=580566eafc9ac68b2614b64d6f7aaa84eebb70da
ENV VENV=/opt/venvs/wilor
RUN python3.11 -m venv ${VENV} && ${VENV}/bin/pip install --upgrade pip setuptools wheel \
    && ${VENV}/bin/pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124
RUN git clone ${WILOR_MINI_REPO} /opt/wilor \
    && git -C /opt/wilor checkout --quiet ${WILOR_MINI_COMMIT}
RUN ${VENV}/bin/pip install \
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
    && ${VENV}/bin/pip install --no-deps /opt/wilor \
    && ${VENV}/bin/python -c "import wilor_mini, ultralytics, smplx, chumpy, dill; print('wilor_mini ok')"

# ---------------------------------------------------------------------------------------------
# SAM 3D Body via SAM-Body4D (body + hand): Python 3.12, torch 2.5.1 + cu121  (backends/sam3dbody/Dockerfile)
# ---------------------------------------------------------------------------------------------
FROM pybase AS sam3dbody
ARG SAM_BODY4D_COMMIT=21af102
ARG MOGE_COMMIT=07444410f1e33f402353b99d6ccd26bd31e469e8
ARG UTILS3D_COMMIT=3fab839f0be9931dac7c8488eb0e1600c236e183
ARG PIPELINE_COMMIT=866f059d2a05cde05e4a52211ec5051fd5f276d6
# Set to 1 to also build detectron2 (needed only for --bbox-mode detector).
ARG WITH_DETECTOR=0
ENV VENV=/opt/venvs/sam3dbody
RUN python3.12 -m venv ${VENV} && ${VENV}/bin/pip install --upgrade pip setuptools wheel \
    && ${VENV}/bin/pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
RUN git clone https://github.com/gaomingqi/sam-body4d.git /opt/sam-body4d \
    && git -C /opt/sam-body4d checkout ${SAM_BODY4D_COMMIT} && rm -rf /opt/sam-body4d/.git
RUN ${VENV}/bin/pip install \
        "numpy==2.2.6" "opencv-python==4.12.0.88" "einops==0.8.2" "roma==1.5.4" "omegaconf==2.3.0" \
        "yacs==0.1.8" "pytorch_lightning==2.6.1" "timm==1.0.24" "huggingface_hub==0.36.1" \
        "pycocotools==2.0.11" "fvcore==0.1.5.post20221221" "iopath==0.1.10" "braceexpand==0.1.7" \
        "termcolor==3.2.0" "cloudpickle==3.1.2" "psutil==7.1.3" "matplotlib==3.10.8" "pyrender==0.1.45" \
        "trimesh==4.11.1" "scipy==1.17.0" "pillow==12.0.0" "tqdm==4.67.3" "pandas" "webdataset" "decord==0.6.0" \
        "imageio[ffmpeg]==2.37.2" \
    && ${VENV}/bin/pip install --no-deps \
        "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@${UTILS3D_COMMIT}" \
        "pipeline @ git+https://github.com/EasternJournalist/pipeline.git@${PIPELINE_COMMIT}" \
        "click==8.3.1" \
    && ${VENV}/bin/pip install --no-deps "moge @ git+https://github.com/microsoft/MoGe.git@${MOGE_COMMIT}"
RUN if [ "${WITH_DETECTOR}" = "1" ]; then \
        ${VENV}/bin/pip install --no-build-isolation --no-deps \
            "git+https://github.com/facebookresearch/detectron2.git@a1ce2f956a1d2212ad672e3c47d53405c2fe4312"; \
    fi

# ---------------------------------------------------------------------------------------------
# PEAR (body + hand + face): Python 3.9, torch 2.0.1 + cu118, pytorch3d V0.7.8 built from source
# (backends/pear/Dockerfile). This is the only stage that needs nvcc, hence the devel base.
# ---------------------------------------------------------------------------------------------
FROM pybase AS pear
ENV VENV=/opt/venvs/pear TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9" FORCE_CUDA=1
RUN python3.9 -m venv ${VENV} && ${VENV}/bin/pip install --upgrade pip setuptools wheel \
    && ${VENV}/bin/pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
        --index-url https://download.pytorch.org/whl/cu118
COPY backends/pear/requirements-runtime.txt backends/pear/constraints.txt /tmp/
RUN ${VENV}/bin/pip install -c /tmp/constraints.txt -r /tmp/requirements-runtime.txt \
    && ${VENV}/bin/pip install -c /tmp/constraints.txt "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8" --no-build-isolation \
    && ${VENV}/bin/pip install -c /tmp/constraints.txt chumpy==0.70 --no-build-isolation \
    && ${VENV}/bin/python -c "import torch, pytorch3d; assert torch.__version__ == '2.0.1+cu118', torch.__version__; print('torch', torch.__version__, 'pytorch3d', pytorch3d.__version__)"
RUN git clone https://github.com/Pixel-Talk/PEAR.git /opt/pear \
    && cd /opt/pear && git checkout e1aa1f7 && rm -rf .git
# License-gated files resolve to the /weights mount at run time (dangling until mounted).
RUN mkdir -p /opt/pear/assets/FLAME/FLAME2020 \
    && ln -s /weights/smplx/SMPLX_NEUTRAL_2020.npz /opt/pear/assets/SMPLX/SMPLX_NEUTRAL_2020.npz \
    && ln -s /weights/flame/generic_model.pkl      /opt/pear/assets/SMPLX/flame_generic_model.pkl \
    && ln -s /weights/flame/generic_model.pkl      /opt/pear/assets/FLAME/FLAME2020/generic_model.pkl \
    && chmod -R a+rwX /opt/pear

# ---------------------------------------------------------------------------------------------
# TEASER (face): Python 3.10, torch 2.0.1 + cu117, TEASER @ c235716  (backends/teaser/Dockerfile)
# ---------------------------------------------------------------------------------------------
FROM pybase AS teaser
ENV VENV=/opt/venvs/teaser TORCH_HOME=/opt/cache/torch HF_HOME=/opt/cache/hf MPLCONFIGDIR=/tmp/mpl
RUN python3.10 -m venv ${VENV} && ${VENV}/bin/pip install --upgrade pip setuptools wheel \
    && ${VENV}/bin/pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 \
        --index-url https://download.pytorch.org/whl/cu117
RUN git clone https://github.com/Pixel-Talk/TEASER.git /opt/teaser \
    && cd /opt/teaser && git checkout -q c235716 && rm -rf .git
RUN ${VENV}/bin/pip install \
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
RUN cd /opt/teaser && ${VENV}/bin/python - <<'PY'
import sys; sys.path.insert(0, "/opt/teaser")
from src.teaser_encoder import TeaserEncoder
TeaserEncoder()
PY
# License-gated FLAME model is mounted at run time; point the relative asset path at /weights.
RUN rm -rf /opt/teaser/assets/FLAME2020 && ln -s /weights/FLAME2020 /opt/teaser/assets/FLAME2020 \
    && mkdir -p /opt/teaser/pretrained_models && ln -s /weights/TEASER.pt /opt/teaser/pretrained_models/TEASER.pt \
    && chmod -R a+rX /opt/teaser /opt/cache

# ---------------------------------------------------------------------------------------------
# MediaPipe Tasks (CPU fallback, body + hand + face): Python 3.10  (backends/mediapipe/Dockerfile)
# ---------------------------------------------------------------------------------------------
FROM pybase AS mediapipe
ENV VENV=/opt/venvs/mediapipe
RUN python3.10 -m venv ${VENV} && ${VENV}/bin/pip install --upgrade pip setuptools wheel \
    && ${VENV}/bin/pip install \
        mediapipe==0.10.35 \
        numpy==1.26.4 \
        opencv-contrib-python==4.11.0.86
# Model files (MediaPipe Tasks, float16, version 1 = files dated Apr/May 2023 used in the paper).
RUN mkdir -p /opt/mediapipe/models && cd /opt/mediapipe/models \
    && curl -fsSL -o pose_landmarker_heavy.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task \
    && curl -fsSL -o hand_landmarker.task       https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task \
    && curl -fsSL -o face_landmarker.task       https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
    && ${VENV}/bin/python -c "import mediapipe, cv2; print(mediapipe.__version__, cv2.__version__)"

# ---------------------------------------------------------------------------------------------
# Final image: runtime CUDA base + the four interpreters + every venv, checkout and adapter.
# ---------------------------------------------------------------------------------------------
FROM ${CUDA_RUNTIME} AS final
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    HF_HUB_OFFLINE=1 \
    TORCH_HOME=/opt/cache/torch \
    HF_HOME=/opt/cache/hf \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib \
    WILOR_ROOT=/opt/wilor \
    SAM_BODY4D_ROOT=/opt/sam-body4d \
    PEAR_ROOT=/opt/pear \
    TEASER_ROOT=/opt/teaser \
    TEASER_COMMIT=c235716 \
    MEDIAPIPE_MODEL_DIR=/opt/mediapipe/models
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates \
        libgl1 libgles2 libegl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.9 python3.9-distutils python3.10 python3.10-distutils \
        python3.11 python3.11-distutils python3.12 \
    && apt-get purge -y software-properties-common && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY --from=wilor     /opt/venvs/wilor     /opt/venvs/wilor
COPY --from=wilor     /opt/wilor           /opt/wilor
COPY --from=sam3dbody /opt/venvs/sam3dbody /opt/venvs/sam3dbody
COPY --from=sam3dbody /opt/sam-body4d      /opt/sam-body4d
COPY --from=pear      /opt/venvs/pear      /opt/venvs/pear
COPY --from=pear      /opt/pear            /opt/pear
COPY --from=teaser    /opt/venvs/teaser    /opt/venvs/teaser
COPY --from=teaser    /opt/teaser          /opt/teaser
COPY --from=teaser    /opt/cache           /opt/cache
COPY --from=mediapipe /opt/venvs/mediapipe /opt/venvs/mediapipe
COPY --from=mediapipe /opt/mediapipe       /opt/mediapipe

# Static ffmpeg for frame decoding in every venv (see pafpose_backend.VideoReader).
RUN for b in wilor sam3dbody pear teaser mediapipe; do \
        /opt/venvs/$b/bin/pip install --no-cache-dir imageio-ffmpeg==0.6.0 || exit 1; \
    done

# Adapters keep their per-backend layout: /app/<backend>/adapter.py finds /app/_common on its own.
COPY backends/_common/pafpose_backend.py                            /app/_common/pafpose_backend.py
COPY backends/wilor/adapter.py     backends/wilor/to_common.py       /app/wilor/
COPY backends/sam3dbody/adapter.py backends/sam3dbody/to_common.py   /app/sam3dbody/
COPY backends/pear/adapter.py      backends/pear/to_common.py        /app/pear/
COPY backends/teaser/adapter.py    backends/teaser/to_common.py      /app/teaser/
COPY backends/mediapipe/adapter.py backends/mediapipe/to_common.py   /app/mediapipe/
COPY backends/_single/pafpose-backend /usr/local/bin/pafpose-backend

# Byte-compile each adapter with its own interpreter, and make caches writable for --user runs.
RUN chmod 755 /usr/local/bin/pafpose-backend \
    && for b in wilor sam3dbody pear teaser mediapipe; do \
         /opt/venvs/$b/bin/python -m py_compile /app/$b/adapter.py /app/$b/to_common.py /app/_common/pafpose_backend.py || exit 1; \
       done \
    && mkdir -p /input /output /weights /tmp/ultralytics /tmp/matplotlib /tmp/mpl /tmp/hf /tmp/torch \
    && chmod 1777 /output /tmp/ultralytics /tmp/matplotlib /tmp/mpl /tmp/hf /tmp/torch \
    && chmod -R a+rX /app

VOLUME ["/input", "/output", "/weights"]
WORKDIR /app
ENTRYPOINT []
CMD ["pafpose-backend", "--help"]
