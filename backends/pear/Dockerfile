# PAF-Pose backend image: PEAR (body + hands + face).
# Build context is the `backends/` directory:  docker compose build pear
#   or: docker build -f backends/pear/Dockerfile -t pafpose/pear:0.1 backends
#
# Upstream: https://github.com/Pixel-Talk/PEAR (Apache-2.0), pinned at commit e1aa1f7.
# License-gated model files (SMPL-X, FLAME) and the PEAR checkpoint are NOT baked in;
# they are mounted at /weights at run time (see README.md).
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp \
    HF_HOME=/tmp/hf \
    TORCH_HOME=/tmp/torch \
    PEAR_ROOT=/opt/pear \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9"

# Python 3.9 (PEAR was run with CPython 3.9) + build tools for pytorch3d / chumpy.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl git build-essential ninja-build \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.9 python3.9-dev python3.9-venv python3.9-distutils \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9 \
    && ln -sf /usr/bin/python3.9 /usr/local/bin/python && ln -sf /usr/bin/python3.9 /usr/local/bin/python3

# torch 2.0.1 + cu118 exactly as PEAR's requirements.txt pins it.
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
        --index-url https://download.pytorch.org/whl/cu118

COPY backends/pear/requirements-runtime.txt /tmp/requirements-runtime.txt
RUN python -m pip install -r /tmp/requirements-runtime.txt \
    && python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.8" --no-build-isolation \
    && python -m pip install chumpy==0.70 --no-build-isolation

# PEAR source at the pinned commit (assets tracked in git come with it).
RUN git clone https://github.com/Pixel-Talk/PEAR.git /opt/pear \
    && cd /opt/pear && git checkout e1aa1f7 && rm -rf .git

# License-gated files resolve to the /weights mount at run time (dangling until mounted).
RUN mkdir -p /opt/pear/assets/FLAME/FLAME2020 \
    && ln -s /weights/smplx/SMPLX_NEUTRAL_2020.npz /opt/pear/assets/SMPLX/SMPLX_NEUTRAL_2020.npz \
    && ln -s /weights/flame/generic_model.pkl      /opt/pear/assets/SMPLX/flame_generic_model.pkl \
    && ln -s /weights/flame/generic_model.pkl      /opt/pear/assets/FLAME/FLAME2020/generic_model.pkl \
    && chmod -R a+rwX /opt/pear /tmp

COPY backends/_common/pafpose_backend.py /app/pafpose_backend.py
COPY backends/pear/adapter.py backends/pear/to_common.py /app/

WORKDIR /opt/pear
ENTRYPOINT []
CMD ["python", "/app/adapter.py", "--help"]
