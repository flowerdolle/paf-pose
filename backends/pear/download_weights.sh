#!/usr/bin/env bash
# Prepare the /weights layout for the PEAR backend.
#   backends/pear/download_weights.sh <weights-root>
# Result:
#   <weights-root>/pear/pear/ehm_model_stage1.pt      (public, downloaded from Hugging Face)
#   <weights-root>/pear/smplx/SMPLX_NEUTRAL_2020.npz  (license-gated, manual)
#   <weights-root>/pear/flame/generic_model.pkl       (license-gated, manual)
set -euo pipefail

ROOT="${1:?usage: download_weights.sh <weights-root>}"
DEST="$ROOT/pear"
mkdir -p "$DEST/pear" "$DEST/smplx" "$DEST/flame"

CKPT="$DEST/pear/ehm_model_stage1.pt"
if [ -f "$CKPT" ]; then
  echo "[pear] checkpoint already present: $CKPT"
else
  echo "[pear] downloading BestWJH/PEAR_models/ehm_model_stage1.pt from Hugging Face ..."
  if command -v python3 >/dev/null 2>&1 && python3 -c "import huggingface_hub" 2>/dev/null; then
    python3 - "$CKPT" <<'PY'
import shutil, sys
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id="BestWJH/PEAR_models", filename="ehm_model_stage1.pt", repo_type="model")
shutil.copyfile(path, sys.argv[1])
print("saved", sys.argv[1])
PY
  else
    curl -L --fail -o "$CKPT" "https://huggingface.co/BestWJH/PEAR_models/resolve/main/ehm_model_stage1.pt"
  fi
fi

missing=0
if [ ! -f "$DEST/smplx/SMPLX_NEUTRAL_2020.npz" ]; then
  missing=1
  cat <<MSG
[pear] MISSING: $DEST/smplx/SMPLX_NEUTRAL_2020.npz
       SMPL-X is license-gated. Register at https://smpl-x.is.tue.mpg.de/ , download
       "SMPL-X v1.1 (NPZ+PKL)" and copy models/smplx/SMPLX_NEUTRAL_2020.npz to the path above.
MSG
fi
if [ ! -f "$DEST/flame/generic_model.pkl" ]; then
  missing=1
  cat <<MSG
[pear] MISSING: $DEST/flame/generic_model.pkl
       FLAME is license-gated. Register at https://flame.is.tue.mpg.de/ , download
       "FLAME 2020" and copy generic_model.pkl to the path above.
MSG
fi
if [ "$missing" -eq 0 ]; then
  echo "[pear] weights complete under $DEST"
else
  echo "[pear] add the files above, then re-run 'pafpose doctor'."
  exit 2
fi
