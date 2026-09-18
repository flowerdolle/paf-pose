# backend: sam3dbody — SAM 3D Body (via the SAM-Body4D implementation, frame-wise)

Produces **body** (body8 + eye2) and **hand** (hands42) parts. In the paper this is the
accuracy-oriented body/hand source (`configs/accuracy.yaml`).

Naming note: the adapter does **not** run the SAM-Body4D video pipeline (SAM 3 tracking,
occlusion completion, temporal smoothing). It calls the SAM 3D Body module bundled in the
`gaomingqi/sam-body4d` checkout once per frame, exactly as the paper did. Cite/describe it as
"SAM 3D Body (via the SAM-Body4D implementation, frame-wise)".

## What the adapter does

1. Reads the video with OpenCV.
2. Per frame: person box = whole frame (`--bbox-mode full`, paper setting), MoGe-2 FoV
   estimate, SAM 3D Body `process_one_image(inference_type="full")` (body + hand decoders),
   largest box kept if several people are returned.
3. `to_common.py`: MHR70 keypoints -> OpenPose53 (body11 + left21 + right21) -> NIA body9 /
   hands, camera frame -> plot frame (x right, y depth, z up). body8 = body9[1:9] (nose
   dropped), eye2 = OpenPose eyes, `body_valid` additionally requires nose, both hands and
   both eyes to be finite (paper protocol).
4. Writes `<stem>.npz` (`body8_eye2_xyz`, `body_valid`, `hands42_xyz`, `hands_valid`,
   `left_hand_valid`, `right_hand_valid`, plus `raw_*` MHR70 / OpenPose53 / bbox / focal arrays)
   and `<stem>.meta.json` (timing, settings, weight paths).

## Weights (`/weights`, i.e. `$PAFPOSE_WEIGHTS/sam3dbody/`)

```text
sam-3d-body-dinov3/model.ckpt              gated  https://huggingface.co/facebook/sam-3d-body-dinov3
sam-3d-body-dinov3/model_config.yaml       gated  (same repo)
sam-3d-body-dinov3/assets/mhr_model.pt     gated  (same repo)
moge-2-vitl-normal/model.pt                public https://huggingface.co/Ruicheng/moge-2-vitl-normal
vitdet/model_final_f05665.pkl              optional, only with --bbox-mode detector
```

`./download_weights.sh <weights_root>` fetches the public file and the gated ones once you have
accepted the SAM 3D Body license on Hugging Face and logged in (`huggingface-cli login` or
`HF_TOKEN`). The adapter passes these paths explicitly (`load_sam_3d_body(ckpt, mhr_path=...)`,
`FOVEstimator(path=...)`), so no Hugging Face access is needed at run time (`HF_HUB_OFFLINE=1`).

## Build and run manually

```bash
docker compose build sam3dbody          # from the repository root
docker run --rm --gpus all \
  -v /abs/clip.mp4:/input/clip.mp4:ro -v /abs/out:/output -v /abs/weights/sam3dbody:/weights:ro \
  pafpose/sam3dbody:0.1 python /app/adapter.py --video /input/clip.mp4 --out /output --weights /weights
```

Options: `--bbox-mode {full,detector}`, `--person-strategy {largest,first}`, `--bbox-thr 0.6`,
`--nms-thr 0.3`, `--inference-type {full,body}`, `--no-fov`, `--max-frames N`, `--verbose-frames`.
`--bbox-mode detector` needs an image built with `--build-arg WITH_DETECTOR=1` (detectron2, nvcc).

## Environment (pinned in the Dockerfile)

Python 3.12, torch 2.5.1+cu121 / torchvision 0.20.1, numpy 2.2.6, opencv 4.12.0.88, omegaconf 2.3.0,
timm 1.0.24, pytorch_lightning 2.6.1, MoGe @ `0744441` (+ utils3d / pipeline pinned commits),
`gaomingqi/sam-body4d` @ `21af102`. Taken from the venv the ICCAS runs used
(`sam-body4d/local/envs/.venv`, `pyvenv.cfg`: Python 3.12.12).

## Known limitations

- ~1 FPS on an A40 (paper: 0.77-1.0 FPS end-to-end). It is the slowest backend.
- CUDA only: upstream `process_one_image` moves batches to CUDA.
- Single person: with `--bbox-mode full` the model sees the whole frame; if two people are visible the
  larger output box wins. Use `--bbox-mode detector` for crowded footage.
- No face output: SAM 3D Body's remaining MHR keypoints have no validated face70 mapping (see the paper
  workspace notes), so this backend declares `parts: [body, hand]` only.
- Local modifications present in the paper's checkout (keypoint-only fast path, extra metadata tables)
  are not required by this adapter; the pinned upstream commit is used unmodified.
