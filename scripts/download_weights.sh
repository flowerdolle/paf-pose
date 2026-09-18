#!/usr/bin/env bash
# Fetch every publicly downloadable weight file for all GPU backends into <weights_root>
# (default: ./weights) and print what still has to be obtained manually (license-gated files).
# See README.md (가중치 준비) for the step-by-step instructions.
set -uo pipefail
ROOT="${1:-weights}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT"
status=0
for backend in sam3dbody pear wilor teaser; do
  echo "================ $backend"
  bash "$HERE/backends/$backend/download_weights.sh" "$ROOT" || status=1
  echo
done
echo "================ summary"
(cd "$HERE" && pafpose doctor --weights "$ROOT" 2>/dev/null | grep -E "^(OK|MISSING) +weights") || \
  echo "(install the host CLI with 'pip install -e .' and run 'pafpose doctor --weights $ROOT' to verify)"
exit $status
