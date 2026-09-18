#!/usr/bin/env python3
"""PAF-Pose backend adapter: TEASER face landmarks for one video.

    python /app/adapter.py --video /input/clip.mp4 --out /output --weights /weights

Pipeline per frame (ported from the ICCAS 2026 run_teaser_face.py):
  1. MediaPipe FaceLandmarker on an enlarged upper-center crop (full-frame fallback),
  2. similarity crop of the face to 224x224,
  3. TeaserEncoder -> FLAME -> ``landmarks_fan_3d`` (68 FAN landmarks),
  4. face70 = FAN68 + two eye centers, converted to the plot frame.

Writes <stem>.npz with face70_xyz / face_valid (+ raw_fan68_flame_xyz, raw_cam,
raw_detection_crop_xyxy) and <stem>.meta.json.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

if not hasattr(np, "float_"):  # NumPy 2 compatibility for TEASER/FLAME imports
    np.float_ = np.float64
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

try:
    import pafpose_backend as pb
except ImportError:  # local testing outside the image
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
    import pafpose_backend as pb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from to_common import fan68_to_face70_plot  # noqa: E402

BACKEND = "teaser"
UPSTREAM_COMMIT = os.environ.get("TEASER_COMMIT", "c235716")
DEFAULT_TEASER_ROOT = Path(os.environ.get("TEASER_ROOT", "/opt/teaser"))


def parse_args():
    parser = pb.base_parser("TEASER face adapter (face70).")
    parser.add_argument("--legacy-face-frame", action="store_true",
                        help="use the paper's [x, z, -y] FLAME->plot transform (face upside-down relative to the body)")
    parser.add_argument("--teaser-root", type=Path, default=DEFAULT_TEASER_ROOT,
                        help="TEASER checkout; must contain src/, utils/, assets/ (default: $TEASER_ROOT or /opt/teaser)")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="TEASER checkpoint (default: <weights>/TEASER.pt)")
    parser.add_argument("--flame-model", type=Path, default=None,
                        help="FLAME generic_model.pkl (default: <weights>/FLAME2020/generic_model.pkl)")
    parser.add_argument("--crop-scale", type=float, default=1.4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--detect-crop-mode", choices=("upper-center", "full"), default="upper-center",
                        help="Run face detection on an enlarged upper-center crop first (faces in full-body "
                             "video are small for the full-frame detector).")
    parser.add_argument("--detect-crop-x-min", type=float, default=0.30)
    parser.add_argument("--detect-crop-x-max", type=float, default=0.70)
    parser.add_argument("--detect-crop-y-min", type=float, default=0.02)
    parser.add_argument("--detect-crop-y-max", type=float, default=0.58)
    parser.add_argument("--no-fallback-full-detection", action="store_true",
                        help="Do not retry detection on the full frame when the crop detection fails.")
    return parser.parse_args()


# --------------------------------------------------------------------------- runtime


def import_teaser_runtime(teaser_root: Path):
    if not teaser_root.is_dir():
        raise SystemExit(f"TEASER root does not exist: {teaser_root}")
    sys.path.insert(0, str(teaser_root))
    os.chdir(teaser_root)  # FLAME and mediapipe_utils load assets via relative 'assets/...' paths
    try:
        import cv2
        import torch
        from skimage.transform import estimate_transform, warp
    except ImportError as exc:
        raise SystemExit(f"Missing TEASER runtime dependency: {exc}") from exc
    from src.FLAME.FLAME import FLAME
    from src.teaser_encoder import TeaserEncoder
    from utils.mediapipe_utils import run_mediapipe

    return cv2, torch, estimate_transform, warp, FLAME, TeaserEncoder, run_mediapipe


def build_models(torch, FLAME, TeaserEncoder, checkpoint: Path, flame_model: Path, device: str):
    if not checkpoint.is_file():
        raise SystemExit(f"Missing TEASER checkpoint: {checkpoint}")
    if not flame_model.is_file():
        raise SystemExit(f"Missing FLAME model: {flame_model} (register at https://flame.is.tue.mpg.de/)")
    encoder = TeaserEncoder().to(device)
    state = torch.load(checkpoint, map_location=device)
    encoder.load_state_dict({k.replace("teaser_encoder.", ""): v for k, v in state.items() if "teaser_encoder" in k})
    encoder.eval()
    flame = FLAME(flame_model_path=str(flame_model)).to(device)
    flame.eval()
    return encoder, flame


def crop_transform(estimate_transform, landmarks_xy: np.ndarray, scale: float, image_size: int):
    left, right = float(landmarks_xy[:, 0].min()), float(landmarks_xy[:, 0].max())
    top, bottom = float(landmarks_xy[:, 1].min()), float(landmarks_xy[:, 1].max())
    old_size = (right - left + bottom - top) / 2.0
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    size = int(old_size * scale)
    if size <= 1:
        return None
    src = np.array([[center[0] - size / 2, center[1] - size / 2],
                    [center[0] - size / 2, center[1] + size / 2],
                    [center[0] + size / 2, center[1] - size / 2]], dtype=np.float32)
    dst = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]], dtype=np.float32)
    return estimate_transform("similarity", src, dst)


def detection_crop(frame_bgr: np.ndarray, args) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = frame_bgr.shape[:2]
    if args.detect_crop_mode == "full":
        return frame_bgr, (0, 0, w, h)
    clamp = lambda v: min(max(v, 0.0), 1.0)  # noqa: E731
    x0, x1 = int(round(w * clamp(args.detect_crop_x_min))), int(round(w * clamp(args.detect_crop_x_max)))
    y0, y1 = int(round(h * clamp(args.detect_crop_y_min))), int(round(h * clamp(args.detect_crop_y_max)))
    if x1 <= x0 or y1 <= y0:
        raise SystemExit("invalid --detect-crop bounds")
    return frame_bgr[y0:y1, x0:x1], (x0, y0, x1, y1)


# --------------------------------------------------------------------------- main


def main() -> int:
    args = parse_args()
    weights = Path(args.weights)
    checkpoint = args.checkpoint or weights / "TEASER.pt"
    flame_model = args.flame_model or weights / "FLAME2020" / "generic_model.pkl"

    video = args.video.resolve()
    out_dir = args.out.resolve()
    timer = pb.Timer()

    runtime = import_teaser_runtime(args.teaser_root.resolve())
    cv2, torch, estimate_transform, warp, FLAME, TeaserEncoder, run_mediapipe = runtime
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    if device != args.device:
        print(f"warning: {args.device} unavailable, using cpu", file=sys.stderr)

    timer.start_load()
    encoder, flame = build_models(torch, FLAME, TeaserEncoder, checkpoint, flame_model, device)
    timer.end_load()

    reader = pb.VideoReader(video, max_frames=args.max_frames)
    T = reader.reported_frames
    face70 = pb.nan_joints(T, 70)
    fan68_raw = pb.nan_joints(T, 68)
    cam = np.full((T, 3), np.nan, dtype=np.float32)
    crops = np.full((T, 4), -1, dtype=np.int32)
    face_valid = np.zeros(T, dtype=bool)
    detection_failures = 0
    decoded = 0

    with torch.no_grad():
        for index, frame_bgr in reader:
            if index >= T:  # container reported fewer frames than it holds
                break
            decoded += 1
            timer.start_frame()
            detect_image, box = detection_crop(frame_bgr, args)
            kpt = run_mediapipe(detect_image)
            if kpt is None and not args.no_fallback_full_detection and args.detect_crop_mode != "full":
                detect_image, box = frame_bgr, (0, 0, reader.width, reader.height)
                kpt = run_mediapipe(detect_image)
            if kpt is None:
                detection_failures += 1
                timer.end_frame()
                continue
            tform = crop_transform(estimate_transform, np.asarray(kpt[..., :2], dtype=np.float32), args.crop_scale, args.image_size)
            if tform is None:
                detection_failures += 1
                timer.end_frame()
                continue
            cropped = warp(detect_image, tform.inverse, output_shape=(args.image_size, args.image_size), preserve_range=True).astype(np.uint8)
            cropped = cv2.resize(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB), (args.image_size, args.image_size))
            tensor = torch.tensor(cropped).permute(2, 0, 1).unsqueeze(0).float().div_(255.0).to(device)

            outputs = encoder(tensor)
            fan68 = flame.forward(outputs)["landmarks_fan_3d"].squeeze(0).detach().cpu().numpy().astype(np.float32)
            timer.end_frame()

            fan68_raw[index] = fan68
            face70[index] = fan68_to_face70_plot(fan68, legacy_face_frame=args.legacy_face_frame)
            face_valid[index] = np.isfinite(face70[index]).all()
            crops[index] = box
            if "cam" in outputs:
                cam[index] = outputs["cam"].squeeze(0).detach().cpu().numpy().astype(np.float32)

    # Presence: TEASER gives no per-landmark confidence; detected frames count as presence 1.0,
    # so the threshold only matters for consistency with other backends.
    presence = np.where(face_valid[:, None], 1.0, 0.0).astype(np.float32)
    face_valid &= (presence >= args.presence_threshold).all(axis=1)

    if decoded < T:  # trim allocation to frames actually decoded
        face70, fan68_raw, cam, crops, face_valid = (a[:decoded] for a in (face70, fan68_raw, cam, crops, face_valid))

    npz_path, meta_path = pb.write_output(
        out_dir,
        video.stem,
        backend=BACKEND,
        upstream_commit=UPSTREAM_COMMIT,
        arrays={
            "face70_xyz": face70,
            "face_valid": face_valid,
            "raw_fan68_flame_xyz": fan68_raw,
            "raw_cam": cam,
            "raw_detection_crop_xyxy": crops,
        },
        fps_source=reader.fps,
        timing=timer,
        extra_meta={
            "video": str(video),
            "source_size": [reader.width, reader.height],
            "frames_decoded": decoded,
            "frames_valid": int(face_valid.sum()),
            "detection_failures": detection_failures,
            "device": device,
            "checkpoint": str(checkpoint),
            "flame_model": str(flame_model),
            "detect_crop_mode": args.detect_crop_mode,
            "mapping": "FLAME landmarks_fan_3d (FAN68) + eye centers -> face70; [x, z, -y] to plot frame",
            "coordinate_warning": "FLAME coordinates are face-local; scale is set by the host fusion step.",
        },
    )
    print(f"saved {npz_path} valid={int(face_valid.sum())}/{decoded} detection_failures={detection_failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
