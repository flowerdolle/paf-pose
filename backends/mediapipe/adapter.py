#!/usr/bin/env python3
"""PAF-Pose backend: MediaPipe Tasks (PoseLandmarker + HandLandmarker + FaceLandmarker), CPU.

Single-video entry point executed inside the pafpose/mediapipe image:

    python /app/adapter.py --video /input/clip.mp4 --out /output

Produces body8_eye2 (pose world landmarks), hands42 (hand world landmarks) and an
approximate face70 (face mesh, image-normalized coordinates converted to pixel
units) in the common plot frame. Mirrors the settings the ICCAS 2026 paper used
for its MediaPipe Pose / Hands / FaceLandmarker baselines.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

try:
    import pafpose_backend as pb
except ImportError:  # local testing outside the image
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
    import pafpose_backend as pb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import to_common as tc  # noqa: E402

BACKEND = "mediapipe"
UPSTREAM_COMMIT = "n/a"  # pip package; version recorded in meta
DEFAULT_MODEL_DIR = Path(os.environ.get("MEDIAPIPE_MODEL_DIR", "/app/models"))
MODEL_FILES = {
    "pose": "pose_landmarker_heavy.task",
    "hand": "hand_landmarker.task",
    "face": "face_landmarker.task",
}


def parse_args():
    parser = pb.base_parser("MediaPipe Tasks body + hands + face backend (CPU).", needs_weights=False)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help=".task files directory")
    parser.add_argument("--delegate", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-presence-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--num-hands", type=int, default=2)
    parser.add_argument("--swap-hands", action="store_true", help="swap MediaPipe left/right labels (mirrored video)")
    parser.add_argument("--face-crop", choices=("upper-center", "full"), default="upper-center",
                        help="FaceLandmarker is weak on small full-body frames; crop enlarges the face (paper default)")
    parser.add_argument("--crop-x-min", type=float, default=0.30)
    parser.add_argument("--crop-x-max", type=float, default=0.70)
    parser.add_argument("--crop-y-min", type=float, default=0.05)
    parser.add_argument("--crop-y-max", type=float, default=0.55)
    parser.add_argument("--face-units", choices=("pixel", "normalized"), default="pixel",
                        help="pixel keeps the face aspect ratio; normalized reproduces the paper's raw output")
    return parser.parse_args()


# --------------------------------------------------------------------------- mediapipe setup


def create_landmarkers(mp, args):
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision

    for key, name in MODEL_FILES.items():
        path = args.model_dir / name
        if not path.is_file():
            raise SystemExit(f"missing MediaPipe {key} model {path}; see download_weights.sh")

    delegate = mp_tasks.BaseOptions.Delegate.GPU if args.delegate == "gpu" else mp_tasks.BaseOptions.Delegate.CPU

    def base(name):
        return mp_tasks.BaseOptions(model_asset_path=str(args.model_dir / name), delegate=delegate)

    pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=base(MODEL_FILES["pose"]),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=args.min_detection_confidence,
            min_pose_presence_confidence=args.min_presence_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            output_segmentation_masks=False,
        )
    )
    hand = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=base(MODEL_FILES["hand"]),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=args.num_hands,
            min_hand_detection_confidence=args.min_detection_confidence,
            min_hand_presence_confidence=args.min_presence_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )
    )
    face = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=base(MODEL_FILES["face"]),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=args.min_detection_confidence,
            min_face_presence_confidence=args.min_presence_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
    )
    return pose, hand, face


def crop_box(width: int, height: int, args) -> dict:
    if args.face_crop == "full":
        return {"x0": 0, "y0": 0, "x1": width, "y1": height, "frame_width": width, "frame_height": height}
    x0, x1 = int(round(width * args.crop_x_min)), int(round(width * args.crop_x_max))
    y0, y1 = int(round(height * args.crop_y_min)), int(round(height * args.crop_y_max))
    if x1 <= x0 or y1 <= y0:
        raise SystemExit("invalid face crop bounds")
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "frame_width": width, "frame_height": height}


def handedness(categories) -> tuple[str | None, float]:
    if not categories:
        return None, 0.0
    category = categories[0]
    name = getattr(category, "category_name", None) or getattr(category, "display_name", None)
    name = name.lower() if isinstance(name, str) else None
    return (name if name in ("left", "right") else None), float(getattr(category, "score", 0.0))


# --------------------------------------------------------------------------- main


def main() -> int:
    args = parse_args()
    import cv2
    import mediapipe as mp

    reader = pb.VideoReader(args.video, max_frames=args.max_frames)
    timer = pb.Timer()
    timer.start_load()
    pose, hand, face = create_landmarkers(mp, args)
    timer.end_load()

    crop = crop_box(reader.width, reader.height, args)
    fps = reader.fps if reader.fps > 0 else 30.0

    body_xyz, body_score = [], []
    left_xyz, right_xyz, left_score, right_score = [], [], [], []
    face_xyz, face_detected = [], []
    pose33_xyz, pose33_vis = [], []
    last_ts = -1

    try:
        for index, frame_bgr in reader:
            ts = max(int(index * 1000 / fps), last_ts + 1)
            last_ts = ts
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            face_rgb = frame_rgb[crop["y0"]:crop["y1"], crop["x0"]:crop["x1"]]

            timer.start_frame()
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
            pose_res = pose.detect_for_video(image, ts)
            hand_res = hand.detect_for_video(image, ts)
            face_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(face_rgb))
            face_res = face.detect_for_video(face_image, ts)
            timer.end_frame()

            # body: world landmarks (meters, camera-like frame) -> body8 + eye2
            world = pose_res.pose_world_landmarks[0] if pose_res.pose_world_landmarks else None
            xyz33, vis33 = tc.landmarks_to_xyz_score(world, 33, "visibility")
            pose33_xyz.append(xyz33)
            pose33_vis.append(vis33)
            b_xyz, b_score = tc.pose33_to_body8_eye2(xyz33, vis33)
            body_xyz.append(b_xyz)
            body_score.append(b_score)

            # hands: world landmarks per side, best handedness score wins
            sides = {"left": (pb.nan_joints(1, 21)[0], 0.0), "right": (pb.nan_joints(1, 21)[0], 0.0)}
            hand_lm = list(hand_res.hand_world_landmarks or [])
            handed = list(hand_res.handedness or [])
            for i, lm in enumerate(hand_lm):
                name, score = handedness(handed[i] if i < len(handed) else [])
                if name is None:
                    continue
                if args.swap_hands:
                    name = "right" if name == "left" else "left"
                if score <= sides[name][1]:
                    continue
                xyz21, _ = tc.landmarks_to_xyz_score(lm, 21, "presence", 1.0)
                sides[name] = (tc.mediapipe21_to_openpose_hand21(xyz21), score)
            left_xyz.append(sides["left"][0])
            left_score.append(sides["left"][1])
            right_xyz.append(sides["right"][0])
            right_score.append(sides["right"][1])

            # face: crop-normalized -> frame-normalized -> face70
            faces = list(face_res.face_landmarks or [])
            if faces:
                xyz478, _ = tc.landmarks_to_xyz_score(faces[0], 478, "presence", 1.0)
                xyz478 = tc.crop_normalized_to_frame_normalized(xyz478, crop)
                face70 = tc.face478_to_face70(xyz478)
                if args.face_units == "pixel":
                    face70 = tc.normalized_to_pixels(face70, reader.width, reader.height)
                face_xyz.append(face70)
                face_detected.append(True)
            else:
                face_xyz.append(pb.nan_joints(1, 70)[0])
                face_detected.append(False)
    finally:
        pose.close()
        hand.close()
        face.close()

    T = len(body_xyz)
    if T == 0:
        raise SystemExit(f"no frames decoded from {args.video}")
    thr = args.presence_threshold

    body_xyz = np.asarray(body_xyz, dtype=np.float32)
    body_score = np.asarray(body_score, dtype=np.float32)
    body_valid = pb.finite_frames(body_xyz) & (body_score >= thr).all(axis=1)

    left_xyz = np.asarray(left_xyz, dtype=np.float32)
    right_xyz = np.asarray(right_xyz, dtype=np.float32)
    left_score = np.asarray(left_score, dtype=np.float32)
    right_score = np.asarray(right_score, dtype=np.float32)
    left_valid = pb.finite_frames(left_xyz) & (left_score >= thr)
    right_valid = pb.finite_frames(right_xyz) & (right_score >= thr)

    face_xyz = np.asarray(face_xyz, dtype=np.float32)
    face_valid = np.asarray(face_detected, dtype=bool) & pb.finite_frames(face_xyz)

    arrays = {
        "body8_eye2_xyz": pb.camera_to_plot(body_xyz),
        "body_valid": body_valid,
        "hands42_xyz": pb.camera_to_plot(np.concatenate([left_xyz, right_xyz], axis=1)),
        "left_hand_valid": left_valid,
        "right_hand_valid": right_valid,
        "face70_xyz": pb.camera_to_plot(face_xyz),
        "face_valid": face_valid,
        "raw_pose33_world_xyz": np.asarray(pose33_xyz, dtype=np.float32),
        "raw_pose33_world_visibility": np.asarray(pose33_vis, dtype=np.float32),
        "raw_body8_eye2_visibility": body_score,
        "raw_left_hand_score": left_score,
        "raw_right_hand_score": right_score,
        "raw_face_crop_xyxy": np.asarray([crop["x0"], crop["y0"], crop["x1"], crop["y1"]], dtype=np.int32),
    }
    extra = {
        "model": {
            "name": "MediaPipe Tasks PoseLandmarker(heavy) + HandLandmarker + FaceLandmarker",
            "mediapipe_version": mp.__version__,
            "delegate": args.delegate,
            "model_files": {k: str(args.model_dir / v) for k, v in MODEL_FILES.items()},
            "min_detection_confidence": args.min_detection_confidence,
            "min_presence_confidence": args.min_presence_confidence,
            "min_tracking_confidence": args.min_tracking_confidence,
            "num_hands": args.num_hands,
            "swap_hands": args.swap_hands,
            "face_crop": args.face_crop,
            "face_crop_xyxy": [crop["x0"], crop["y0"], crop["x1"], crop["y1"]],
            "face_units": args.face_units,
            "presence_threshold": thr,
        },
        "frame_width": reader.width,
        "frame_height": reader.height,
        "notes": [
            "body8_eye2 and hands42 come from MediaPipe world landmarks (meters, per-part origin at the part center);",
            "face70 is an approximate topology from FaceMesh in image units, not metric; PAF-Pose fusion rescales it.",
        ],
    }
    npz, meta = pb.write_output(
        args.out, args.video.stem, backend=BACKEND, upstream_commit=UPSTREAM_COMMIT,
        arrays=arrays, fps_source=reader.fps, timing=timer, extra_meta=extra,
    )
    print(f"wrote {npz} and {meta}: {T} frames, body {int(body_valid.sum())}, "
          f"left {int(left_valid.sum())}, right {int(right_valid.sum())}, face {int(face_valid.sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
