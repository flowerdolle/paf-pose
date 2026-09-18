# backend: wilor

WiLoR hand estimator (Potamias et al., "WiLoR: End-to-end 3D hand localization and reconstruction
in-the-wild") run through the pip-installable **WiLoR-mini** package, exactly as in the ICCAS 2026
paper runs. Produces the common **hand** part only: `hands42_xyz` (left21 + right21, OpenPose hand
order) with `left_hand_valid` / `right_hand_valid` masks.

## What the adapter does

`adapter.py --video X.mp4 --out DIR --weights /weights`

1. Loads `WiLorHandPose3dEstimationPipeline` from the mounted weights (no network access; the image sets
   `HF_HUB_OFFLINE=1`).
2. For every frame: WiLoR-mini's YOLO hand detector finds hand boxes (`--hand-conf`, default 0.3);
   the largest box per side is kept; the crop is enlarged by `--rescale-factor` (2.5) and passed to WiLoR.
3. `pred_keypoints_3d + pred_cam_t_full` (camera frame, meters) is converted to the plot frame
   (x right, y depth, z up) by `to_common.py`.
4. A side is valid in a frame when a hand was detected there (WiLoR has no per-joint confidence, so the
   presence threshold of the common contract is satisfied trivially by detected hands).
5. Writes `<stem>.npz` + `<stem>.meta.json`; model-specific extras are stored with a `raw_` prefix
   (local/camera hand21 per side, detector boxes, detections per frame).

Options: `--max-frames N`, `--device cpu|cuda`, `--dtype auto|float16|float32` (fp16 on GPU by default),
`--focal-length` (WiLoR-internal focal length at 256 px crop scale; default 5000 as in the paper),
`--swap-hands`.

## Weights layout (`/weights`)

```text
/weights/pretrained_models/wilor_final.ckpt       WiLoR checkpoint (~2.4 GB)
/weights/pretrained_models/detector.pt            YOLO hand detector
/weights/pretrained_models/MANO_RIGHT.pkl         MANO right-hand model (license-gated)
/weights/pretrained_models/mano_mean_params.npz   MANO mean parameters
```

This is the layout `wilor_mini` expects when given `wilor_pretrained_dir=/weights`. The adapter refuses
to start if any file is missing (it never downloads inside the container).

```bash
backends/wilor/download_weights.sh weights                      # public files -> weights/wilor/
backends/wilor/download_weights.sh weights --with-mano-mirror   # also MANO_RIGHT.pkl from the HF mirror
```

MANO must be obtained under its own license from https://mano.is.tue.mpg.de (see the script's message).

## Image

`Dockerfile` (build context: repository root): `pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime`, WiLoR-mini
pinned at commit `ebec42f9` (package version 1.1), `ultralytics==8.1.34`, `smplx==0.1.28`, chumpy pinned
at `580566ea`. These match the local environment used for the paper. The upstream repository is cloned at
build time into `/opt/wilor`; nothing from it is vendored here.

Manual run (what `pafpose run` does for you):

```bash
docker build -f backends/wilor/Dockerfile -t pafpose/wilor:0.1 .
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -v "$PWD/clip.mp4:/input/clip.mp4:ro" -v "$PWD/out:/output" -v "$PWD/weights/wilor:/weights:ro" \
  pafpose/wilor:0.1 python /app/adapter.py --video /input/clip.mp4 --out /output --weights /weights
```

## Relation to the paper

* The paper evaluated WiLoR with two crop sources. Its **best** WiLoR numbers
  (`outputs/wilor_hand_sam_bbox`, used in the fusion table) fed WiLoR hand boxes taken from SAM 3D Body.
  A standalone backend cannot depend on another backend, so this adapter uses WiLoR-mini's own YOLO
  detector (the paper's `outputs/wilor_hand` setting). Expect more missed frames on small/occluded hands
  than the SAM-box variant; accuracy on detected frames is the same model.
* The official model code lives in https://github.com/rolpotamias/WiLoR; the paper runs (and this image)
  use https://github.com/warmshao/WiLoR-mini, a packaged re-implementation of the same released weights
  (Hugging Face `warmshao/WiLoR-mini`, revision `b00adea9`).
* Validity rule and coordinate conversion reproduce the paper's `load_hand_prediction("wilor",
  hand_coordinate="default")` (verified against the paper's raw npz files with `to_common.from_paper_npz`).

## Known limitations

* Single person assumed; with several people the largest detected hand per side wins.
* fp16 inference on GPU by default (as in the paper); pass `--dtype float32` for CPU or if NaNs appear.
* The CUDA 12.4 base image needs an NVIDIA driver >= 525 (tested with 535 through CUDA minor-version
  compatibility).
