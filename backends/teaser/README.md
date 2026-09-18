# backend: teaser

Face-only backend. Runs [TEASER](https://github.com/Pixel-Talk/TEASER) (commit `c235716`)
per frame and writes 70 face landmarks in the common layout.

Per frame: MediaPipe FaceLandmarker on an enlarged upper-center crop (full-frame fallback)
→ similarity crop to 224×224 → `TeaserEncoder` → FLAME → `landmarks_fan_3d` (68 FAN/iBUG
landmarks) → `face70` = FAN68 + right/left eye centers, mapped to the plot frame with `[x, z, -y]`.

| file | role |
| --- | --- |
| `adapter.py` | container entry point (`--video`, `--out`, `--weights`, `--device`, `--max-frames`, crop options) |
| `to_common.py` | FAN68 → face70 + coordinate conversion (pure numpy, self-check with `python to_common.py <old npz>`) |
| `Dockerfile` | `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime` + upstream pins, TEASER checkout at `/opt/teaser` |
| `download_weights.sh` | fetches `TEASER.pt`, prints FLAME instructions |

## Output

`<stem>.npz`: `face70_xyz (T,70,3)`, `face_valid (T,)`, plus `raw_fan68_flame_xyz`, `raw_cam`,
`raw_detection_crop_xyxy`. `<stem>.meta.json`: backend, commit, timing, detection failures.

TEASER provides no per-landmark confidence, so `face_valid` is simply "face detected and
reconstruction finite". FLAME coordinates are face-local (not metric); the host fusion step
rescales the face by the inter-eye distance of the body source, so only the shape matters.

## Weights (`/weights`, host `weights/teaser/`)

```text
weights/teaser/
├── TEASER.pt                     public (Google Drive), fetched by download_weights.sh
└── FLAME2020/generic_model.pkl   license-gated: register at https://flame.is.tue.mpg.de/
```

The image symlinks `/opt/teaser/assets/FLAME2020 -> /weights/FLAME2020` and
`/opt/teaser/pretrained_models/TEASER.pt -> /weights/TEASER.pt`, so upstream's relative asset
paths resolve without writing into the image. `face_landmarker.task` (public) and the timm
backbone weights are baked into the image at build time; no network is needed at run time.

## Manual run

```bash
docker build -f backends/teaser/Dockerfile -t pafpose/teaser:0.1 .
docker run --rm --gpus all --user $(id -u):$(id -g) \
  -v /abs/clip.mp4:/input/clip.mp4:ro -v /abs/out:/output -v /abs/weights/teaser:/weights:ro \
  pafpose/teaser:0.1 python /app/adapter.py --video /input/clip.mp4 --out /output --weights /weights
```

## Limitations

- Single face per frame; the first detected face is used.
- The default detection crop (x 0.30–0.70, y 0.02–0.58 of the frame) assumes an upright,
  roughly centered upper body as in the paper's videos. For other framings pass
  `--detect-crop-mode full` or adjust `--detect-crop-*`.
- CPU fallback works but is slow (~11 FPS on an A40 in the paper setting).
- Build has not yet been executed on a Docker host (no Docker on the development machine).
