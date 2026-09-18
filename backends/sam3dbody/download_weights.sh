#!/usr/bin/env bash
# Fetch SAM 3D Body backend weights into <weights_root>/sam3dbody/.
#
#   ./download_weights.sh /path/to/weights          # -> /path/to/weights/sam3dbody/...
#
# Public:  MoGe-2 (Ruicheng/moge-2-vitl-normal, model.pt)
# Gated:   SAM 3D Body (facebook/sam-3d-body-dinov3): accept the license on Hugging Face,
#          then `huggingface-cli login` or export HF_TOKEN=... and rerun this script.
# Optional: ViTDet person detector (only for --bbox-mode detector); the code downloads it
#          itself from the detectron2 model zoo when no local file is given.
set -euo pipefail

ROOT="${1:?usage: download_weights.sh <weights_root>}"
DEST="${ROOT%/}/sam3dbody"
mkdir -p "$DEST/sam-3d-body-dinov3/assets" "$DEST/moge-2-vitl-normal"

if ! python3 -c "import huggingface_hub" 2>/dev/null; then
  echo "huggingface_hub is required on the host: pip install huggingface_hub" >&2
  exit 1
fi

python3 - "$DEST" <<'PY'
import os, sys
from pathlib import Path
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError

dest = Path(sys.argv[1])
token = os.environ.get("HF_TOKEN")

def fetch(repo, filename, target):
    """Download repo/filename into dest/<repo leaf>/<filename> (hf keeps the sub-path)."""
    if target.is_file() and target.stat().st_size > 0:
        print(f"[skip] {target}")
        return True
    try:
        hf_hub_download(repo_id=repo, filename=filename, local_dir=str(dest / repo.split("/")[-1]), token=token)
        print(f"[ok]   {target}")
        return target.is_file()
    except GatedRepoError:
        print(f"[gated] {repo}/{filename}: accept the license at https://huggingface.co/{repo} and log in (huggingface-cli login or HF_TOKEN)")
    except HfHubHTTPError as exc:
        print(f"[fail] {repo}/{filename}: {exc}")
    return False

ok = True
ok &= fetch("Ruicheng/moge-2-vitl-normal", "model.pt", dest / "moge-2-vitl-normal" / "model.pt")
ok &= fetch("facebook/sam-3d-body-dinov3", "model.ckpt", dest / "sam-3d-body-dinov3" / "model.ckpt")
ok &= fetch("facebook/sam-3d-body-dinov3", "model_config.yaml", dest / "sam-3d-body-dinov3" / "model_config.yaml")
ok &= fetch("facebook/sam-3d-body-dinov3", "assets/mhr_model.pt", dest / "sam-3d-body-dinov3" / "assets" / "mhr_model.pt")

print()
print("expected layout under", dest)
for rel in ("sam-3d-body-dinov3/model.ckpt", "sam-3d-body-dinov3/model_config.yaml",
            "sam-3d-body-dinov3/assets/mhr_model.pt", "moge-2-vitl-normal/model.pt"):
    print(("  [present] " if (dest / rel).is_file() else "  [MISSING] ") + rel)
sys.exit(0 if ok else 2)
PY
