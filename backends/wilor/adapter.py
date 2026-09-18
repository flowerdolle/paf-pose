#!/usr/bin/env python3
"""PAF-Pose backend adapter: WiLoR (via the WiLoR-mini package) on one video -> common hands42 npz.

Inside the container:

    python /app/adapter.py --video /input/clip.mp4 --out /output --weights /weights

Expected weights layout (mounted read-only at --weights):

    <weights>/pretrained_models/wilor_final.ckpt
    <weights>/pretrained_models/detector.pt
    <weights>/pretrained_models/MANO_RIGHT.pkl
    <weights>/pretrained_models/mano_mean_params.npz

Per frame, WiLoR-mini's own YOLO hand detector finds hands; the largest box per side is kept;
camera-space keypoints (pred_keypoints_3d + pred_cam_t_full) are converted to the plot frame.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

try:
    import pafpose_backend as pb
except ImportError:  # local development outside the image
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
    import pafpose_backend as pb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import to_common  # noqa: E402

BACKEND = "wilor"
UPSTREAM = "warmshao/WiLoR-mini"
UPSTREAM_COMMIT = "ebec42f94c389070cdd7dda6fd1bf0b4a659c960"
WEIGHT_FILES = ("wilor_final.ckpt", "detector.pt", "MANO_RIGHT.pkl", "mano_mean_params.npz")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = pb.base_parser("WiLoR hand backend for PAF-Pose")
    parser.add_argument(
        "--wilor-root",
        type=Path,
        default=Path(os.environ.get("WILOR_ROOT", "/opt/wilor")),
        help="WiLoR-mini checkout; added to sys.path only if the wilor_mini package is not installed",
    )
    parser.add_argument("--hand-conf", type=float, default=0.3, help="YOLO hand detector confidence")
    parser.add_argument("--rescale-factor", type=float, default=2.5, help="crop enlargement around the hand box")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16"), default="auto")
    parser.add_argument(
        "--focal-length",
        type=float,
        default=5000.0,
        help="WiLoR internal focal length at 256 px crop scale (default 5000, as in the paper runs); "
        "for a calibrated camera pass f_pixels * 256 / max(width, height)",
    )
    parser.add_argument("--swap-hands", action="store_true", help="swap detector left/right labels")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- model


def check_weights(weights: Path) -> Path:
    pretrained = weights / "pretrained_models"
    missing = [name for name in WEIGHT_FILES if not (pretrained / name).is_file()]
    if missing:
        raise SystemExit(
            f"missing WiLoR weights under {pretrained}: {missing}\n"
            "run backends/wilor/download_weights.sh <weights-dir> (MANO_RIGHT.pkl is license-gated, see README)"
        )
    return weights


def load_pipeline(args: argparse.Namespace):
    try:
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
    except ImportError:
        if args.wilor_root.is_dir():
            sys.path.insert(0, str(args.wilor_root))
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
    import torch

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "float32":
        dtype = torch.float32
    else:
        dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = WiLorHandPose3dEstimationPipeline(
        device=device,
        dtype=dtype,
        wilor_pretrained_dir=str(check_weights(args.weights)),
        focal_length=args.focal_length,
        verbose=False,
    )
    return pipe, str(device), str(dtype)


# --------------------------------------------------------------------------- per-frame logic


def bbox_area(detection: Mapping[str, object]) -> float:
    bbox = np.asarray(detection["hand_bbox"], dtype=np.float32)
    if bbox.shape != (4,) or not np.isfinite(bbox).all():
        return 0.0
    return float(max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]))


def select_hands(detections: Iterable[Mapping[str, object]], swap_hands: bool) -> dict[str, Mapping[str, object]]:
    """Keep the largest detection per side."""
    best: dict[str, tuple[float, Mapping[str, object]]] = {}
    for det in detections:
        side = "right" if int(det["is_right"]) else "left"
        if swap_hands:
            side = "left" if side == "right" else "right"
        area = bbox_area(det)
        if side not in best or area > best[side][0]:
            best[side] = (area, det)
    return {side: det for side, (_, det) in best.items()}


def process_video(reader: pb.VideoReader, pipe, args: argparse.Namespace, timer: pb.Timer) -> dict[str, np.ndarray]:
    import cv2

    local = {"left": [], "right": []}
    camera = {"left": [], "right": []}
    presence = {"left": [], "right": []}
    detected = {"left": [], "right": []}
    bboxes = {"left": [], "right": []}
    num_detections = []

    for _, frame_bgr in reader:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        timer.start_frame()
        detections = pipe.predict(rgb, hand_conf=args.hand_conf, rescale_factor=args.rescale_factor)
        timer.end_frame()
        selected = select_hands(detections, args.swap_hands)
        num_detections.append(len(detections))
        for side in ("left", "right"):
            if side in selected:
                loc, cam = to_common.detection_to_camera_hand21(selected[side])
                local[side].append(loc)
                camera[side].append(cam)
                presence[side].append(np.ones(21, dtype=np.float32))
                detected[side].append(True)
                bboxes[side].append(np.asarray(selected[side]["hand_bbox"], dtype=np.float32))
            else:
                local[side].append(np.full((21, 3), np.nan, dtype=np.float32))
                camera[side].append(np.full((21, 3), np.nan, dtype=np.float32))
                presence[side].append(np.zeros(21, dtype=np.float32))
                detected[side].append(False)
                bboxes[side].append(np.full(4, np.nan, dtype=np.float32))

    T = len(num_detections)
    if T == 0:
        raise SystemExit(f"no frames decoded from {reader.path}")

    def stack(seq, shape):
        return np.asarray(seq, dtype=np.float32).reshape((T,) + shape)

    left_cam, right_cam = stack(camera["left"], (21, 3)), stack(camera["right"], (21, 3))
    left_valid = to_common.side_valid(detected["left"], stack(presence["left"], (21,)), args.presence_threshold)
    right_valid = to_common.side_valid(detected["right"], stack(presence["right"], (21,)), args.presence_threshold)

    return {
        "hands42_xyz": to_common.hands42_from_camera_sides(left_cam, right_cam),
        "left_hand_valid": left_valid,
        "right_hand_valid": right_valid,
        # model-specific extras (camera frame, meters; NaN where not detected)
        "raw_left_hand21_local_xyz": stack(local["left"], (21, 3)),
        "raw_right_hand21_local_xyz": stack(local["right"], (21, 3)),
        "raw_left_hand21_camera_xyz": left_cam,
        "raw_right_hand21_camera_xyz": right_cam,
        "raw_left_hand_bbox_xyxy": stack(bboxes["left"], (4,)),
        "raw_right_hand_bbox_xyxy": stack(bboxes["right"], (4,)),
        "raw_detected_hands_per_frame": np.asarray(num_detections, dtype=np.int16),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    timer = pb.Timer()
    timer.start_load()
    pipe, device, dtype = load_pipeline(args)
    timer.end_load()

    reader = pb.VideoReader(args.video, max_frames=args.max_frames)
    arrays = process_video(reader, pipe, args, timer)
    npz_path, meta_path = pb.write_output(
        args.out,
        args.video.stem,
        backend=BACKEND,
        upstream_commit=UPSTREAM_COMMIT,
        arrays=arrays,
        fps_source=reader.fps,
        timing=timer,
        extra_meta={
            "upstream": UPSTREAM,
            "model": {
                "name": "WiLoR",
                "package": "wilor_mini 1.1",
                "device": device,
                "dtype": dtype,
                "hand_conf": args.hand_conf,
                "rescale_factor": args.rescale_factor,
                "focal_length_256px": args.focal_length,
                "swap_hands": args.swap_hands,
                "detector": "WiLoR-mini YOLO hand detector (paper's best WiLoR row used SAM 3D Body hand boxes instead)",
            },
            "video": {"width": reader.width, "height": reader.height, "reported_frames": reader.reported_frames},
            "frames_left_valid": int(arrays["left_hand_valid"].sum()),
            "frames_right_valid": int(arrays["right_hand_valid"].sum()),
        },
    )
    print(f"wrote {npz_path} and {meta_path}: {arrays['hands42_xyz'].shape[0]} frames, "
          f"left {int(arrays['left_hand_valid'].sum())}, right {int(arrays['right_hand_valid'].sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
