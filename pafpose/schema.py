"""Common output contract between backend containers and the host.

Every backend adapter writes ``<stem>.npz`` plus ``<stem>.meta.json`` using only
the keys below. The host never inspects model-specific arrays.

Coordinate convention ("plot" frame, identical to the NIA ground truth used in
the paper): x right, y forward (depth, away from the camera), z up, in meters.
A camera-frame array (x right, y down, z forward) becomes plot frame with
``camera_to_plot``. Backends convert their native frame in ``to_common.py``.
Frames a backend could not estimate are NaN and marked ``False`` in the
matching ``*_valid`` mask.

Optional keys: ``left_hand_valid`` / ``right_hand_valid`` (T,) give per-side
validity; when absent both sides share ``hands_valid``. A backend that can
produce more parts than were selected should still write all of them: the
body backend's own ``hands42_xyz`` is used as the attachment anchor for hands.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

PARTS: tuple[str, ...] = ("body", "hand", "face")

# Joint names -------------------------------------------------------------

BODY8_EYE2_NAMES: tuple[str, ...] = (
    "neck",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "mid_hip",
    "right_eye",
    "left_eye",
)

HAND21_NAMES: tuple[str, ...] = (
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)

# hands42 = left21 followed by right21 (OpenPose hand order).
# face70 = 68 iBUG landmarks followed by right eye center and left eye center.

# Array keys --------------------------------------------------------------

PART_TO_KEY: dict[str, str] = {
    "body": "body8_eye2_xyz",
    "hand": "hands42_xyz",
    "face": "face70_xyz",
}

JOINT_COUNT: dict[str, int] = {
    "body8_eye2_xyz": 10,
    "hands42_xyz": 42,
    "face70_xyz": 70,
}

VALID_KEY: dict[str, str] = {
    "body8_eye2_xyz": "body_valid",
    "hands42_xyz": "hands_valid",
    "face70_xyz": "face_valid",
}

OPTIONAL_KEYS: tuple[str, ...] = ("left_hand_valid", "right_hand_valid")

META_REQUIRED: tuple[str, ...] = (
    "backend",          # registry name, e.g. "pear"
    "upstream_commit",  # pinned commit of the external repo, or "n/a"
    "parts",            # list of parts present in the npz
    "num_frames",
    "fps_source",       # source video fps
    "timing",           # {"total_sec": float, "per_frame_sec": [...]}
)


@dataclass(frozen=True)
class BackendOutput:
    """One backend's result for one video, as read from disk."""

    npz_path: Path
    meta_path: Path
    arrays: Mapping[str, np.ndarray]
    meta: Mapping[str, object]

    def part(self, part: str) -> np.ndarray:
        return np.asarray(self.arrays[PART_TO_KEY[part]])

    def valid(self, part: str) -> np.ndarray:
        return np.asarray(self.arrays[VALID_KEY[PART_TO_KEY[part]]]).astype(bool)

    def hand_side_valid(self, side: str) -> np.ndarray:
        """Per-side hand validity; falls back to the shared hands mask."""
        key = f"{side}_hand_valid"
        if key in self.arrays:
            return np.asarray(self.arrays[key]).astype(bool)
        return self.valid("hand")

    def has(self, part: str) -> bool:
        return PART_TO_KEY[part] in self.arrays

    @property
    def num_frames(self) -> int:
        for key in PART_TO_KEY.values():
            if key in self.arrays:
                return int(np.asarray(self.arrays[key]).shape[0])
        return 0


def camera_to_plot(points: np.ndarray) -> np.ndarray:
    """Camera frame (x right, y down, z forward) -> plot frame (x right, y depth, z up)."""
    points = np.asarray(points, dtype=np.float32)
    out = np.empty_like(points)
    out[..., 0] = points[..., 0]
    out[..., 1] = points[..., 2]
    out[..., 2] = -points[..., 1]
    return out


def output_paths(out_dir: Path, stem: str) -> tuple[Path, Path]:
    return out_dir / f"{stem}.npz", out_dir / f"{stem}.meta.json"


def load_output(npz_path: Path, meta_path: Path | None = None) -> BackendOutput:
    npz_path = Path(npz_path)
    if meta_path is None:
        meta_path = npz_path.with_suffix("").with_suffix(".meta.json")
    with np.load(npz_path) as data:
        arrays = {key: data[key] for key in data.files}
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    return BackendOutput(npz_path=npz_path, meta_path=Path(meta_path), arrays=arrays, meta=meta)


def validate(
    arrays: Mapping[str, np.ndarray],
    meta: Mapping[str, object],
    expected_parts: Iterable[str],
) -> list[str]:
    """Return human-readable problems; an empty list means the output is usable."""
    problems: list[str] = []
    expected = list(expected_parts)

    for field in META_REQUIRED:
        if field not in meta:
            problems.append(f"meta.json missing field '{field}'")

    frame_counts: dict[str, int] = {}
    for part in expected:
        if part not in PART_TO_KEY:
            problems.append(f"unknown part '{part}'")
            continue
        key = PART_TO_KEY[part]
        vkey = VALID_KEY[key]
        if key not in arrays:
            problems.append(f"missing array '{key}' for part '{part}'")
            continue
        arr = np.asarray(arrays[key])
        joints = JOINT_COUNT[key]
        if arr.ndim != 3 or arr.shape[1:] != (joints, 3):
            problems.append(f"'{key}' has shape {arr.shape}, expected (T, {joints}, 3)")
            continue
        if not np.issubdtype(arr.dtype, np.floating):
            problems.append(f"'{key}' dtype {arr.dtype} is not floating point")
        frame_counts[key] = int(arr.shape[0])
        if vkey not in arrays:
            problems.append(f"missing mask '{vkey}' for '{key}'")
            continue
        valid = np.asarray(arrays[vkey])
        if valid.shape != (arr.shape[0],):
            problems.append(f"'{vkey}' has shape {valid.shape}, expected ({arr.shape[0]},)")
            continue
        valid = valid.astype(bool)
        if valid.sum() == 0:
            problems.append(f"'{vkey}' marks no valid frame")
        finite = np.isfinite(arr).all(axis=(1, 2))
        bad = int((valid & ~finite).sum())
        if bad:
            problems.append(f"'{key}' has {bad} frames marked valid but containing non-finite values")

    for key in OPTIONAL_KEYS:
        if key in arrays and frame_counts:
            expected_len = next(iter(frame_counts.values()))
            if np.asarray(arrays[key]).shape != (expected_len,):
                problems.append(f"'{key}' has shape {np.asarray(arrays[key]).shape}, expected ({expected_len},)")

    if len(set(frame_counts.values())) > 1:
        problems.append(f"frame counts differ across parts: {frame_counts}")
    if frame_counts and "num_frames" in meta:
        declared = int(meta["num_frames"])  # type: ignore[arg-type]
        actual = next(iter(frame_counts.values()))
        if declared != actual:
            problems.append(f"meta num_frames={declared} but arrays have T={actual}")

    if "parts" in meta:
        declared_parts = set(meta["parts"])  # type: ignore[arg-type]
        missing = [p for p in expected if p not in declared_parts]
        if missing:
            problems.append(f"meta.parts {sorted(declared_parts)} does not declare expected {missing}")

    return problems


def validate_output(output: BackendOutput, expected_parts: Iterable[str]) -> list[str]:
    return validate(output.arrays, output.meta, expected_parts)
