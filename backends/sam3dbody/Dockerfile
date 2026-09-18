# PAF-Pose backend: SAM 3D Body (via the SAM-Body4D implementation, frame-wise)
# Build context is `backends/`:  docker compose build sam3dbody
#
# Versions are pinned to the environment the ICCAS 2026 runs used
# (Python 3.12, torch 2.5.1+cu121, numpy 2.2.6, opencv 4.12, MoGe @ 0744441).
# Weights are NOT baked in; mount them at /weights (see README.md).

FROM python:3.12-slim-bookworm

ARG SAM_BODY4D_COMMIT=21af102
ARG MOGE_COMMIT=07444410f1e33f402353b99d6ccd26bd31e469e8
ARG UTILS3D_COMMIT=3fab839f0be9931dac7c8488eb0e1600c236e183
ARG PIPELINE_COMMIT=866f059d2a05cde05e4a52211ec5051fd5f276d6
# Set to 1 to also build detectron2 (needed only for --bbox-mode detector; requires nvcc).
ARG WITH_DETECTOR=0

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    SAM_BODY4D_ROOT=/opt/sam-body4d HF_HUB_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates build-essential libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch (CUDA 12.1 wheels bundle the CUDA runtime; the host only needs the driver + nvidia-container-toolkit)
RUN pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Upstream checkout at the pinned commit (code only; never vendored into this repository)
RUN git clone https://github.com/gaomingqi/sam-body4d.git /opt/sam-body4d \
    && git -C /opt/sam-body4d checkout ${SAM_BODY4D_COMMIT}

# Dependencies of the frame-wise SAM 3D Body path (subset of upstream pyproject; gradio/diffusers/sam3 are not needed)
RUN pip install \
        "numpy==2.2.6" "opencv-python==4.12.0.88" "einops==0.8.2" "roma==1.5.4" "omegaconf==2.3.0" \
        "yacs==0.1.8" "pytorch_lightning==2.6.1" "timm==1.0.24" "huggingface_hub==0.36.1" \
        "pycocotools==2.0.11" "fvcore==0.1.5.post20221221" "iopath==0.1.10" "braceexpand==0.1.7" \
        "termcolor==3.2.0" "cloudpickle==3.1.2" "psutil==7.1.3" "matplotlib==3.10.8" "pyrender==0.1.45" \
        "trimesh==4.11.1" "scipy==1.17.0" "pillow==12.0.0" "tqdm==4.67.3" "pandas" "webdataset" "decord==0.6.0" \
        "imageio[ffmpeg]==2.37.2" \
    && pip install --no-deps \
        "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@${UTILS3D_COMMIT}" \
        "pipeline @ git+https://github.com/EasternJournalist/pipeline.git@${PIPELINE_COMMIT}" \
        "click==8.3.1" \
    && pip install --no-deps "moge @ git+https://github.com/microsoft/MoGe.git@${MOGE_COMMIT}"

# Optional person detector (ViTDet through detectron2). Off by default: the paper used --bbox-mode full.
RUN if [ "${WITH_DETECTOR}" = "1" ]; then \
        pip install --no-build-isolation --no-deps \
            "git+https://github.com/facebookresearch/detectron2.git@a1ce2f956a1d2212ad672e3c47d53405c2fe4312"; \
    fi

COPY backends/_common/pafpose_backend.py /app/pafpose_backend.py
COPY backends/sam3dbody/adapter.py backends/sam3dbody/to_common.py /app/

WORKDIR /app
RUN python -m py_compile /app/adapter.py /app/to_common.py /app/pafpose_backend.py

CMD ["python", "/app/adapter.py", "--help"]
