"""WiLoR outputs -> PAF-Pose common hands42 layout (numpy only).

Ported from SL_MST/papers/iccas2026/models/hand/wilor/wilor_to_nia_hand.py and the
``load_hand_prediction(model="wilor", hand_coordinate="default")`` rule of the paper's fusion
scripts:

* joints: WiLoR's 21 MANO-style joints are already in OpenPose hand order
  (wrist, thumb, index, middle, ring, pinky chains) -> copied 1:1;
* coordinates: camera-space keypoints = ``pred_keypoints_3d + pred_cam_t_full`` (meters, camera
  frame x right / y down / z forward) converted to the plot frame with ``camera_to_plot``;
* validity per side = hand detected in that frame and every joint presence >= threshold
  (WiLoR gives no per-joint confidence, so presence is 1.0 for detected hands).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

import numpy as np

try:
    from pafpose_backend import camera_to_plot
except ImportError:  # local development outside the image
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
    from pafpose_backend import camera_to_plot

HAND21_NAMES = (
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)


def wilor21_to_openpose_hand21(points: np.ndarray) -> np.ndarray:
    """WiLoR hand21 is OpenPose-compatible; the identity mapping is kept explicit."""
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (21, 3):
        raise ValueError(f"expected WiLoR hand points (21, 3), got {points.shape}")
    return points.copy()


def detection_to_camera_hand21(detection: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    """One WiLoR-mini detection dict -> (local21, camera21) in the camera frame, meters."""
    preds = detection["wilor_preds"]
    local_xyz = wilor21_to_openpose_hand21(np.asarray(preds["pred_keypoints_3d"][0], dtype=np.float32))
    translation = np.asarray(preds["pred_cam_t_full"][0], dtype=np.float32)
    return local_xyz, local_xyz + translation[None, :]


def side_valid(frame_valid: np.ndarray, presence: np.ndarray, threshold: float) -> np.ndarray:
    frame_valid = np.asarray(frame_valid, dtype=bool)
    presence = np.asarray(presence, dtype=np.float32)
    return frame_valid & (presence >= threshold).all(axis=1)


def hands42_from_camera_sides(left_camera21: np.ndarray, right_camera21: np.ndarray) -> np.ndarray:
    """(T,21,3) camera-frame left/right -> (T,42,3) plot-frame hands42 (left21 then right21)."""
    left = np.asarray(left_camera21, dtype=np.float32)
    right = np.asarray(right_camera21, dtype=np.float32)
    if left.shape != right.shape or left.ndim != 3 or left.shape[1:] != (21, 3):
        raise ValueError(f"expected (T, 21, 3) per side, got {left.shape} and {right.shape}")
    return camera_to_plot(np.concatenate([left, right], axis=1))


def from_paper_npz(path: Path, presence_threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a legacy ``*_wilor_hand.npz`` (paper workspace) to (hands42, left_valid, right_valid)."""
    data = np.load(path, allow_pickle=True)
    hands42 = hands42_from_camera_sides(data["left_hand21_camera_xyz"], data["right_hand21_camera_xyz"])
    left_valid = side_valid(data["left_frame_valid"], data["left_hand21_presence"], presence_threshold)
    right_valid = side_valid(data["right_frame_valid"], data["right_hand21_presence"], presence_threshold)
    return hands42, left_valid, right_valid
