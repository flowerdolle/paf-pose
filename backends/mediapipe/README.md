# backend: mediapipe

CPU fallback backend. One container run produces **body + hand + face** from a single video:

- `body8_eye2_xyz` from MediaPipe **PoseLandmarker (heavy)** world landmarks (meters, hip-centered).
- `hands42_xyz` from MediaPipe **HandLandmarker** world landmarks (meters, per-hand centered), left/right by handedness score.
- `face70_xyz` from MediaPipe **FaceLandmarker** (FaceMesh 478 → approximate OpenPose face70), image units.

This is the MediaPipe **Tasks API** (mediapipe 0.10.35). The ICCAS 2026 paper's MediaPipe Pose / Hands /
FaceLandmarker baselines used exactly this API on CPU (mediapipe 0.10.35 no longer ships `mp.solutions`,
so Holistic is not available), and the same settings are kept here: detection / presence / tracking
confidence 0.5, `num_hands=2`, face detection on an upper-center crop (x 0.30–0.70, y 0.05–0.55 of the frame)
because the face landmarker fails on small faces in full-body frames.

## Files

| File | Role |
| --- | --- |
| `adapter.py` | single-video entry point; writes `<stem>.npz` + `<stem>.meta.json` |
| `to_common.py` | pure-numpy mappings: pose33 → body8+eye2, hand21 order, face478 → face70, crop un-normalization |
| `Dockerfile` | `python:3.10-slim` + mediapipe 0.10.35, numpy 1.26.4, opencv-contrib 4.11.0.86; `.task` files baked in |
| `download_weights.sh` | no weights dir needed; downloads the `.task` files for a local (non-docker) run |

## Model files (baked into the image at build time)

| Model | URL (float16, version 1) | md5 of the files used in the paper |
| --- | --- | --- |
| pose_landmarker_heavy.task | `mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/` | `453dec4d02ccc4d3ce812b6de84fa516` |
| hand_landmarker.task | `mediapipe-models/hand_landmarker/hand_landmarker/float16/1/` | `15318430ea3851670fe9914116a9cfad` |
| face_landmarker.task | `mediapipe-models/face_landmarker/face_landmarker/float16/1/` | `b0e7274907a1644404fef66b28dd6d85` |

## Manual container run

```bash
docker build -f backends/mediapipe/Dockerfile -t pafpose/mediapipe:0.1 .   # from the repository root
docker run --rm -v /abs/clip.mp4:/input/clip.mp4:ro -v /abs/out:/output pafpose/mediapipe:0.1 \
  python /app/adapter.py --video /input/clip.mp4 --out /output
```

Useful options: `--max-frames N`, `--presence-threshold 0.5`, `--face-crop full`, `--swap-hands` (mirrored video),
`--face-units normalized` (raw paper-style normalized face coordinates instead of pixel units), `--model-dir`.

Local run without docker (needs libGLESv2/libEGL on the host, e.g. `apt install libgles2 libegl1`):

```bash
./download_weights.sh /path/to/models
python adapter.py --video clip.mp4 --out out/ --model-dir /path/to/models
```

## Output details

- Coordinates are converted to the common plot frame (x right, y depth, z up) with `camera_to_plot`,
  as the paper's evaluation did for MediaPipe body / hands / face.
- `body_valid`: pose found, all body8 + eye2 joints finite and visibility ≥ `--presence-threshold`.
- `left_hand_valid` / `right_hand_valid`: hand detected with handedness score ≥ threshold; `hands_valid` = both.
- `face_valid`: a face was found in the crop.
- Extra arrays with a `raw_` prefix: pose33 world xyz + visibility, body/hand scores, face crop box.

## Known limitations

- **Scale is not metric-consistent across parts.** Pose world landmarks and hand world landmarks are each
  in meters but centered on their own part; face70 is in image pixel units (or normalized units). PAF-Pose
  fusion rescales hands (median bone ratio) and face (eye distance ratio) onto the body anchor, so this
  only matters if you use the raw per-part arrays directly. The body anchor itself carries MediaPipe's
  world-landmark scale, which the paper found less accurate than the mesh-based body models.
- **face70 is an approximate topology**: FaceMesh vertices were hand-picked to mimic the 68 iBUG points;
  eye centers (68, 69) are the iris centers.
- Hands are attached only via body wrists at fusion time (PoseLandmarker has no full hand), so hand scale
  is kept as MediaPipe reports it (wrist-only anchor → scale 1.0).
- Single person only: `num_poses=1`, `num_faces=1`, best-scoring hand per side.
- CPU only (`--delegate gpu` exists but needs EGL/GPU capabilities in the container and is untested).
