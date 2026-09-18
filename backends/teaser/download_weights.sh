#!/usr/bin/env bash
# Prepare the TEASER weights directory for PAF-Pose.
#   bash backends/teaser/download_weights.sh <weights_root>/teaser
# Result layout:
#   <dir>/TEASER.pt                      (public, Google Drive)
#   <dir>/FLAME2020/generic_model.pkl    (license-gated, manual)
set -euo pipefail
ROOT="${1:?usage: download_weights.sh <weights_root>   (writes <weights_root>/teaser/)}"
DEST="${ROOT%/}/teaser"
mkdir -p "$DEST/FLAME2020"

# TEASER.pt — public Google Drive file used by upstream quick_install.sh
if [ ! -f "$DEST/TEASER.pt" ]; then
  if command -v gdown >/dev/null 2>&1; then
    echo "Downloading TEASER.pt ..."
    gdown 1SjYuCfDf5ElxK3VNvcF7KPBR57EIRDSY -O "$DEST/TEASER.pt"
  else
    echo "gdown not found (pip install gdown). Download TEASER.pt manually from"
    echo "  https://drive.google.com/drive/folders/1WhTjAZIQBCZqDRziu8_ZBtMC736K9T2A"
    echo "and place it at $DEST/TEASER.pt"
  fi
else
  echo "TEASER.pt already present."
fi

# FLAME 2020 — requires registration and license agreement; cannot be fetched automatically.
if [ ! -f "$DEST/FLAME2020/generic_model.pkl" ]; then
  cat <<MSG

FLAME 2020 is license-gated. Register at https://flame.is.tue.mpg.de/ , download
FLAME2020.zip, and extract generic_model.pkl to:
  $DEST/FLAME2020/generic_model.pkl
(unzip -j FLAME2020.zip generic_model.pkl -d "$DEST/FLAME2020")
MSG
else
  echo "FLAME2020/generic_model.pkl already present."
fi

echo
echo "Expected layout:"
echo "  $DEST/TEASER.pt"
echo "  $DEST/FLAME2020/generic_model.pkl"
