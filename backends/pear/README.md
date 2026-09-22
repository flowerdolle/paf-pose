# backend: pear

PEAR (Pixel-aligned Expressive humAn mesh Recovery, Pixel-Talk/PEAR @ `e1aa1f7`) as a
PAF-Pose backend. One container run processes one single-person `.mp4` and writes the
common-layout result for **body, hands, and face** in one pass.

| file | role |
| --- | --- |
| `adapter.py` | container entry point; re-implements the paper's skeleton-only `mesh_inference` path without Gradio |
| `to_common.py` | SMPL-X 145 keypoints -> OpenPose 67 -> body8_eye2 / hands42; name-based face68 + eye centers -> face70 |
| `Dockerfile` | CUDA 11.8 + Python 3.9 + torch 2.0.1 + pytorch3d 0.7.8, PEAR cloned at the pinned commit |
| `download_weights.sh` | fetches the public checkpoint, explains the license-gated files |
| `requirements-runtime.txt` | pinned Python dependencies for the inference path |

## Weights layout

Mounted read-only at `/weights` (host: `$PAFPOSE_WEIGHTS/pear/`):

```text
pear/
├── pear/pear_model.pt        Hugging Face BestWJH/PEAR_models (public, auto-download)
├── smplx/SMPLX_NEUTRAL_2020.npz    SMPL-X v1.1, https://smpl-x.is.tue.mpg.de/ (license required)
└── flame/generic_model.pkl         FLAME 2020,  https://flame.is.tue.mpg.de/  (license required)
```

The image contains symlinks `assets/SMPLX/SMPLX_NEUTRAL_2020.npz`, `assets/SMPLX/flame_generic_model.pkl`
and `assets/FLAME/FLAME2020/generic_model.pkl` pointing into `/weights`, so PEAR's code finds them at
its usual relative paths. All other assets PEAR needs are tracked in its repository and come with the clone.
SMPL (`SMPL_NEUTRAL.pkl`), MANO and SMPLX2SMPL are only used by PEAR's training/conversion utilities and are
not required.

```bash
backends/pear/download_weights.sh "$PAFPOSE_WEIGHTS"
```

## Manual container run

```bash
docker compose build pear
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -v /abs/clip.mp4:/input/clip.mp4:ro -v /abs/out:/output -v "$PAFPOSE_WEIGHTS/pear":/weights:ro \
  pafpose/pear:0.1 python /app/adapter.py --video /input/clip.mp4 --out /output --weights /weights
```

Options: `--max-frames N`, `--device cpu`, `--no-smooth` (skip PEAR's Savitzky-Golay parameter smoothing),
`--save-raw` (adds `raw_smplx145_xyz`, `raw_openpose67_xyz`, `raw_cameras`), `--legacy-face-frame` (see below).

## Output

`/output/<stem>.npz` with `body8_eye2_xyz`, `body_valid`, `hands42_xyz`, `hands_valid`, `left_hand_valid`,
`right_hand_valid`, `face70_xyz`, `face_valid` (plot frame: x right, y depth, z up, meters) and
`/output/<stem>.meta.json` with timing (`forward_fps` = backbone+head per frame; `reconstruction_fps` =
EHM_v2 keypoint regression per frame).

PEAR is a mesh-prior model and always produces every joint, so validity masks are simply "all values finite".

## Notes and limitations

* **Face frame.** The paper's exported PEAR face arrays (`openpose70_plot_xyz`) were never rotated into the
  plot frame while body and hands were; this adapter rotates all three parts consistently. Part-wise face70
  PA-MPJPE is unaffected (Procrustes), but whole-body numbers for PEAR-face combinations will differ slightly
  from the paper table. Pass `--legacy-face-frame` to reproduce the paper's arrays exactly.
* The OpenPose-67 index table and the face68 name table are not in upstream PEAR; they were carried over
  from the paper workspace (`SL_MST/papers/iccas2026/models/body/pear/`).
* Smoothing windows (7/5/7, polyorder 2) match PEAR's `app.py`; for clips shorter than the window the
  window is shrunk instead of failing.
* Frames are decoded with OpenCV (BGR -> RGB) instead of decord; the model input is otherwise identical
  (letterbox to 256x256, `/255`, ImageNet normalisation inside the pipeline).
* PEAR's `Ehm_Pipeline` is a Lightning module and `smplx_head.py` loads `assets/SMPLX/smpl_mean_params.npz`
  relative to the working directory, so the adapter `chdir`s into `/opt/pear`.
