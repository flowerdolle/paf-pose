"""Shared helpers for backend adapters. Copied into every backend image as /app/pafpose_backend.py.

This file must stay dependency-light (numpy + opencv only) and must not import the
host ``pafpose`` package, which is not installed inside backend containers.

Output contract (mirrors pafpose/schema.py on the host):

    <out>/<stem>.npz        body8_eye2_xyz (T,10,3)  body_valid (T,)
                            hands42_xyz    (T,42,3)  hands_valid (T,)  left_hand_valid (T,)  right_hand_valid (T,)
                            face70_xyz     (T,70,3)  face_valid (T,)
                            (only the parts the backend produces; extra model-specific arrays may be
                            added with a ``raw_`` prefix)
    <out>/<stem>.meta.json  backend, upstream_commit, parts, num_frames, fps_source, timing, ...

Coordinate convention of every *_xyz array: "plot" frame, x right, y depth (away
from the camera), z up, meters. Convert a camera frame (x right, y down, z
forward) with :func:`camera_to_plot`. Frames without an estimate are NaN and
False in the matching mask.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterator

import numpy as np

BODY8_EYE2_NAMES = (
    "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "mid_hip",
    "right_eye", "left_eye",
)
HAND21_NAMES = (
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)
# face70 = 68 iBUG landmarks (jaw 0-16, brows 17-26, nose 27-35, eyes 36-47, mouth 48-67)
# followed by right eye center (68) and left eye center (69).

JOINTS = {"body8_eye2_xyz": 10, "hands42_xyz": 42, "face70_xyz": 70}
PART_KEY = {"body": "body8_eye2_xyz", "hand": "hands42_xyz", "face": "face70_xyz"}
VALID_KEY = {"body8_eye2_xyz": "body_valid", "hands42_xyz": "hands_valid", "face70_xyz": "face_valid"}


# --------------------------------------------------------------------------- geometry


def camera_to_plot(points: np.ndarray) -> np.ndarray:
    """Camera frame (x right, y down, z forward) -> plot frame (x right, y depth, z up)."""
    points = np.asarray(points, dtype=np.float32)
    out = np.empty_like(points)
    out[..., 0] = points[..., 0]
    out[..., 1] = points[..., 2]
    out[..., 2] = -points[..., 1]
    return out


def nan_joints(num_frames: int, joints: int) -> np.ndarray:
    return np.full((num_frames, joints, 3), np.nan, dtype=np.float32)


def finite_frames(points: np.ndarray) -> np.ndarray:
    return np.isfinite(points).all(axis=tuple(range(1, points.ndim)))


# --------------------------------------------------------------------------- video


class VideoReader:
    """Sequential BGR frame reader with fps / frame count (opencv)."""

    def __init__(self, path: Path, max_frames: int | None = None):
        import cv2

        self.path = Path(path)
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video {self.path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        reported = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.max_frames = max_frames
        self.reported_frames = reported if max_frames is None else min(reported, max_frames)

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        index = 0
        while True:
            if self.max_frames is not None and index >= self.max_frames:
                break
            ok, frame = self.cap.read()
            if not ok:
                break
            yield index, frame
            index += 1
        self.cap.release()

    def read_all(self) -> list[np.ndarray]:
        return [frame for _, frame in self]


# --------------------------------------------------------------------------- timing


class Timer:
    """Collects model-load time and per-frame forward time."""

    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.model_load_sec = 0.0
        self.per_frame_sec: list[float] = []
        self._t = 0.0

    def start_load(self) -> None:
        self._t = time.perf_counter()

    def end_load(self) -> None:
        self.model_load_sec = time.perf_counter() - self._t

    def start_frame(self) -> None:
        self._t = time.perf_counter()

    def end_frame(self) -> None:
        self.per_frame_sec.append(time.perf_counter() - self._t)

    def as_dict(self) -> dict:
        frames = len(self.per_frame_sec)
        forward = float(sum(self.per_frame_sec))
        return {
            "total_sec": time.perf_counter() - self.started,
            "model_load_sec": self.model_load_sec,
            "forward_sec": forward,
            "per_frame_sec": [round(t, 6) for t in self.per_frame_sec],
            "forward_fps": (frames / forward) if forward > 0 else None,
        }


# --------------------------------------------------------------------------- cli / output


def base_parser(description: str, needs_weights: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--video", type=Path, required=True, help="input .mp4 (single person)")
    parser.add_argument("--out", type=Path, required=True, help="output directory for <stem>.npz and <stem>.meta.json")
    parser.add_argument("--weights", type=Path, required=needs_weights, default=None, help="weights root mounted at /weights")
    parser.add_argument("--device", default="cuda", help="torch device (cuda / cpu) where applicable")
    parser.add_argument("--max-frames", type=int, default=None, help="process only the first N frames")
    parser.add_argument("--presence-threshold", type=float, default=0.5, help="min per-joint confidence for a valid frame")
    return parser


def write_output(
    out_dir: Path,
    stem: str,
    *,
    backend: str,
    upstream_commit: str,
    arrays: dict[str, np.ndarray],
    fps_source: float,
    timing: Timer | dict,
    extra_meta: dict | None = None,
) -> tuple[Path, Path]:
    """Validate shapes lightly, derive ``hands_valid`` if only per-side masks were given, write files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = dict(arrays)

    if "hands42_xyz" in arrays and "hands_valid" not in arrays:
        if "left_hand_valid" in arrays and "right_hand_valid" in arrays:
            arrays["hands_valid"] = np.asarray(arrays["left_hand_valid"], bool) & np.asarray(arrays["right_hand_valid"], bool)
        else:
            raise ValueError("hands42_xyz needs hands_valid or both left_hand_valid/right_hand_valid")

    parts = []
    num_frames = None
    for part, key in PART_KEY.items():
        if key not in arrays:
            continue
        arr = np.asarray(arrays[key], dtype=np.float32)
        if arr.ndim != 3 or arr.shape[1:] != (JOINTS[key], 3):
            raise ValueError(f"{key} has shape {arr.shape}, expected (T, {JOINTS[key]}, 3)")
        mask_key = VALID_KEY[key]
        if mask_key not in arrays:
            raise ValueError(f"{key} needs mask {mask_key}")
        mask = np.asarray(arrays[mask_key]).astype(bool)
        if mask.shape != (arr.shape[0],):
            raise ValueError(f"{mask_key} has shape {mask.shape}, expected ({arr.shape[0]},)")
        # never mark a non-finite frame valid
        mask = mask & finite_frames(arr)
        arrays[key] = arr
        arrays[mask_key] = mask
        parts.append(part)
        if num_frames is None:
            num_frames = int(arr.shape[0])
        elif num_frames != arr.shape[0]:
            raise ValueError(f"frame count mismatch: {key} has {arr.shape[0]}, expected {num_frames}")
    if not parts:
        raise ValueError("no part arrays to write")

    npz_path = out_dir / f"{stem}.npz"
    meta_path = out_dir / f"{stem}.meta.json"
    np.savez_compressed(npz_path, **arrays)
    meta = {
        "backend": backend,
        "upstream_commit": upstream_commit,
        "parts": parts,
        "num_frames": num_frames,
        "fps_source": float(fps_source),
        "timing": timing.as_dict() if isinstance(timing, Timer) else dict(timing),
        "coordinate": "plot: x right, y depth, z up, meters",
        "layout": {
            "body8_eye2_xyz": list(BODY8_EYE2_NAMES),
            "hands42_xyz": "left21 + right21, OpenPose hand order",
            "face70_xyz": "68 iBUG landmarks + right eye center + left eye center",
        },
    }
    if extra_meta:
        meta.update(extra_meta)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return npz_path, meta_path
