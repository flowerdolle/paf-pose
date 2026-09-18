"""SAM 3D Body (MHR70) -> OpenPose53 -> common PAF-Pose layout. Pure numpy.

Ported from SL_MST/papers/iccas2026/models/body/sam-body4d/mhr70_to_nia.py.

MHR70 layout (SAM 3D Body ``pred_keypoints_3d`` first 70 keypoints): body/face
points in 0..20, right hand 21..41, left hand 42..62, extra body points 63..69.
OpenPose53 = BODY_25 subset body11 [0,1,2,3,4,5,6,7,8,15,16] + left hand21 + right hand21.
NIA body9 = openpose53[:9] = nose, neck, R shoulder/elbow/wrist, L shoulder/elbow/wrist, mid hip.
"""

from __future__ import annotations

import numpy as np

# key: OpenPose BODY_25 index, value: MHR70 index (mid hip 8 is synthesized from hips 9/10).
MHR70_TO_OPENPOSE_BODY25 = {
    0: 0, 1: 69, 2: 6, 3: 8, 4: 41, 5: 5, 6: 7, 7: 62,
    9: 10, 10: 12, 11: 14, 12: 9, 13: 11, 14: 13,
    15: 2, 16: 1, 17: 4, 18: 3, 19: 15, 20: 16, 21: 17, 22: 18, 23: 19, 24: 20,
}
OPENPOSE_BODY11_INDICES = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16], dtype=np.int64)
# OpenPose hand order: wrist, thumb, index, middle, ring, pinky (base -> tip).
# MHR70 hand slices store the wrist last and each finger tip -> base.
MHR70_HAND_TO_OPENPOSE21 = np.asarray(
    [20, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16], dtype=np.int64
)
MHR70_RIGHT_HAND = slice(21, 42)
MHR70_LEFT_HAND = slice(42, 63)

OP53_BODY9 = slice(0, 9)
OP53_EYE2 = np.asarray([9, 10], dtype=np.int64)   # BODY_25 right eye (15), left eye (16)
OP53_LEFT_HAND = slice(11, 32)
OP53_RIGHT_HAND = slice(32, 53)


def camera_to_plot(points: np.ndarray) -> np.ndarray:
    """Camera frame (x right, y down, z forward) -> plot frame (x right, y depth, z up)."""
    points = np.asarray(points, dtype=np.float32)
    out = np.empty_like(points)
    out[..., 0] = points[..., 0]
    out[..., 1] = points[..., 2]
    out[..., 2] = -points[..., 1]
    return out


def mhr70_to_openpose_body25(mhr70: np.ndarray) -> np.ndarray:
    mhr70 = np.asarray(mhr70, dtype=np.float32)
    if mhr70.shape[-2] < 70 or mhr70.shape[-1] != 3:
        raise ValueError(f"expected (..., >=70, 3) MHR keypoints, got {mhr70.shape}")
    body25 = np.full(mhr70.shape[:-2] + (25, 3), np.nan, dtype=np.float32)
    for op_idx, mhr_idx in MHR70_TO_OPENPOSE_BODY25.items():
        body25[..., op_idx, :] = mhr70[..., mhr_idx, :]
    body25[..., 8, :] = (mhr70[..., 9, :] + mhr70[..., 10, :]) * 0.5
    return body25


def mhr70_to_openpose53(mhr70: np.ndarray) -> np.ndarray:
    """(..., 70, 3) -> (..., 53, 3) body11 + left hand21 + right hand21."""
    mhr70 = np.asarray(mhr70, dtype=np.float32)
    body11 = np.take(mhr70_to_openpose_body25(mhr70), OPENPOSE_BODY11_INDICES, axis=-2)
    left = np.take(mhr70[..., MHR70_LEFT_HAND, :], MHR70_HAND_TO_OPENPOSE21, axis=-2)
    right = np.take(mhr70[..., MHR70_RIGHT_HAND, :], MHR70_HAND_TO_OPENPOSE21, axis=-2)
    return np.concatenate([body11, left, right], axis=-2)


def openpose53_to_nia_body9(openpose53: np.ndarray) -> np.ndarray:
    return np.asarray(openpose53, dtype=np.float32)[..., OP53_BODY9, :]


def openpose53_to_nia_body_hands51(openpose53: np.ndarray) -> np.ndarray:
    openpose53 = np.asarray(openpose53, dtype=np.float32)
    return np.concatenate(
        [openpose53[..., OP53_BODY9, :], openpose53[..., OP53_LEFT_HAND, :], openpose53[..., OP53_RIGHT_HAND, :]],
        axis=-2,
    )


def finite_frames(points: np.ndarray) -> np.ndarray:
    return np.isfinite(points).all(axis=tuple(range(1, points.ndim)))


def mhr70_to_common(mhr70_xyz: np.ndarray, frame_valid: np.ndarray) -> dict[str, np.ndarray]:
    """(T, 70, 3) camera-frame MHR keypoints + (T,) frame mask -> common PAF-Pose arrays.

    Exactly the paper's protocol (``load_sam`` with ``sam_coordinate="plot"``):
    body8 = NIA body9[1:9], eye2 = openpose53[[9, 10]], hands = body_hands51[9:30] / [30:51],
    everything converted camera -> plot; body_valid requires the nose, both hands and
    both eyes to be finite as well.
    """
    mhr70_xyz = np.asarray(mhr70_xyz, dtype=np.float32)
    frame_valid = np.asarray(frame_valid, dtype=bool)
    if mhr70_xyz.ndim != 3:
        raise ValueError(f"expected (T, 70, 3), got {mhr70_xyz.shape}")
    if frame_valid.shape != (mhr70_xyz.shape[0],):
        raise ValueError(f"frame_valid shape {frame_valid.shape} does not match T={mhr70_xyz.shape[0]}")

    openpose53 = camera_to_plot(mhr70_to_openpose53(mhr70_xyz))
    body_hands51 = openpose53_to_nia_body_hands51(openpose53)
    body9 = body_hands51[:, :9]
    left = body_hands51[:, 9:30]
    right = body_hands51[:, 30:51]
    eye2 = openpose53[:, OP53_EYE2]

    body_valid = frame_valid & finite_frames(body9) & finite_frames(left) & finite_frames(right) & finite_frames(eye2)
    left_valid = frame_valid & finite_frames(left)
    right_valid = frame_valid & finite_frames(right)

    return {
        "body8_eye2_xyz": np.concatenate([body9[:, 1:9], eye2], axis=1).astype(np.float32),
        "body_valid": body_valid,
        "hands42_xyz": np.concatenate([left, right], axis=1).astype(np.float32),
        "left_hand_valid": left_valid,
        "right_hand_valid": right_valid,
        "hands_valid": left_valid & right_valid,
        "raw_openpose53_plot_xyz": openpose53,
    }
