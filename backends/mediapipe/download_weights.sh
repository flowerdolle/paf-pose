#!/usr/bin/env bash
# MediaPipe needs no external weights directory: the three .task model files are downloaded
# into the image at build time (see Dockerfile). Run this script only to fetch the same files
# for a local (non-docker) run:  ./download_weights.sh <model-dir>
set -euo pipefail
BASE="https://storage.googleapis.com/mediapipe-models"
if [ $# -eq 0 ]; then
  echo "mediapipe backend: no weights directory required (models are baked into the image)."
  echo "usage for a local run: $0 <model-dir>   then: --model-dir <model-dir>"
  exit 0
fi
DIR="$1"; mkdir -p "$DIR"
curl -fsSL -o "$DIR/pose_landmarker_heavy.task" "$BASE/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
curl -fsSL -o "$DIR/hand_landmarker.task"       "$BASE/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
curl -fsSL -o "$DIR/face_landmarker.task"       "$BASE/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
echo "downloaded 3 .task files into $DIR"
