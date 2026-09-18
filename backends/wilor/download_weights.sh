#!/usr/bin/env bash
# Fetch WiLoR weights for the PAF-Pose wilor backend.
#
#   backends/wilor/download_weights.sh <weights-dir> [--with-mano-mirror]
#
# Downloads the publicly hosted WiLoR checkpoint, its YOLO hand detector, and the MANO mean
# parameters from the WiLoR-mini Hugging Face repository (pinned revision) into
# <weights-dir>/pretrained_models/. MANO_RIGHT.pkl is license-gated and is only printed as an
# instruction unless --with-mano-mirror is given.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <weights_root> [--with-mano-mirror]   (writes <weights_root>/wilor/pretrained_models/)" >&2
    exit 2
fi

TARGET="${1%/}/wilor/pretrained_models"
WITH_MANO_MIRROR="${2:-}"
HF_REPO="warmshao/WiLoR-mini"
HF_REVISION="b00adea9a6843bbb4c9042109c5eb29ab2a59dea"   # revision used for the paper runs
BASE_URL="https://huggingface.co/${HF_REPO}/resolve/${HF_REVISION}/pretrained_models"

mkdir -p "${TARGET}"

fetch() {
    local name="$1"
    local dest="${TARGET}/${name}"
    if [[ -s "${dest}" ]]; then
        echo "exists   ${dest}"
        return
    fi
    echo "download ${name}"
    curl -fL --retry 3 -C - -o "${dest}.part" "${BASE_URL}/${name}"
    mv "${dest}.part" "${dest}"
}

fetch wilor_final.ckpt        # ~2.4 GB, WiLoR (ViT-H) checkpoint
fetch detector.pt             # YOLO hand detector shipped with WiLoR
fetch mano_mean_params.npz    # MANO mean pose/shape parameters

if [[ "${WITH_MANO_MIRROR}" == "--with-mano-mirror" ]]; then
    fetch MANO_RIGHT.pkl
else
    cat <<MSG

MANO_RIGHT.pkl is NOT downloaded automatically: the MANO hand model is distributed under its own
license (https://mano.is.tue.mpg.de). To comply with it:
  1. register at https://mano.is.tue.mpg.de and download "Models & Code" (mano_v1_2.zip);
  2. copy mano_v1_2/models/MANO_RIGHT.pkl to ${TARGET}/MANO_RIGHT.pkl
The WiLoR-mini Hugging Face repository also mirrors this file; re-run with --with-mano-mirror to
fetch it from there if you have accepted the MANO license.
MSG
fi

echo
echo "weights directory: ${1%/}/wilor"
ls -l "${TARGET}"
