"""PEAR native outputs -> PAF-Pose common layout (pure numpy).

PEAR's EHM head returns 145 SMPL-X keypoints per frame (``ehm.smplx.keypoint_names``).
The paper pipeline (SL_MST/papers/iccas2026/models/body/pear) derived:

* ``joints_3d`` (67): OpenPose BODY_25 + left hand21 + right hand21, via the index table
  ``OPENPOSE_FROM_SMPLX145`` below (ported from the local PEAR ``app.py``; not part of
  upstream commit e1aa1f7).
* ``face70``: 68 iBUG landmarks selected from the SMPL-X keypoints by name, plus two eye
  centers averaged from the six eye-contour landmarks per side (``export_pear_face70.py``).

Common layout: body8_eye2 (neck, R shoulder/elbow/wrist, L shoulder/elbow/wrist, mid hip,
R eye, L eye), hands42 (left21 + right21), face70. All arrays are converted from PEAR's
camera-like frame (x right, y down, z forward) to the plot frame (x right, y depth, z up).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# --------------------------------------------------------------------------- SMPL-X 145 -> OpenPose 67

OPENPOSE_BODY25_NAMES = (
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "MidHip", "RHip",
    "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar", "LBigToe",
    "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel",
)
OPENPOSE_HAND_NAMES = (
    "Wrist", "Thumb1", "Thumb2", "Thumb3", "Thumb4", "Index1", "Index2", "Index3", "Index4",
    "Middle1", "Middle2", "Middle3", "Middle4", "Ring1", "Ring2", "Ring3", "Ring4",
    "Pinky1", "Pinky2", "Pinky3", "Pinky4",
)
OPENPOSE67_NAMES = (
    OPENPOSE_BODY25_NAMES
    + tuple(f"LHand_{n}" for n in OPENPOSE_HAND_NAMES)
    + tuple(f"RHand_{n}" for n in OPENPOSE_HAND_NAMES)
)

# None = MidHip, synthesized as the mean of SMPL-X left_hip (1) and right_hip (2).
OPENPOSE_BODY25_FROM_SMPLX145 = (
    68, 12, 17, 19, 21, 16, 18, 20, None, 2,
    5, 8, 1, 4, 7, 24, 23, 107, 122, 124,
    132, 127, 135, 143, 138,
)
OPENPOSE_LEFT_HAND_FROM_SMPLX145 = (
    20, 37, 38, 39, 133, 25, 26, 27, 128, 28,
    29, 30, 129, 34, 35, 36, 131, 31, 32, 33, 130,
)
OPENPOSE_RIGHT_HAND_FROM_SMPLX145 = (
    21, 52, 53, 54, 144, 40, 41, 42, 139, 43,
    44, 45, 140, 49, 50, 51, 142, 46, 47, 48, 141,
)
OPENPOSE_FROM_SMPLX145 = (
    OPENPOSE_BODY25_FROM_SMPLX145 + OPENPOSE_LEFT_HAND_FROM_SMPLX145 + OPENPOSE_RIGHT_HAND_FROM_SMPLX145
)


def smplx145_to_openpose67(joints: np.ndarray) -> np.ndarray:
    """(T, 145, 3) SMPL-X keypoints -> (T, 67, 3) OpenPose body25 + hands."""
    joints = np.asarray(joints, dtype=np.float32)
    if joints.ndim != 3 or joints.shape[1] < 145 or joints.shape[2] != 3:
        raise ValueError(f"expected (T, >=145, 3), got {joints.shape}")
    columns = []
    for src in OPENPOSE_FROM_SMPLX145:
        if src is None:
            columns.append((joints[:, 1] + joints[:, 2]) * 0.5)
        else:
            columns.append(joints[:, src])
    return np.stack(columns, axis=1).astype(np.float32, copy=False)


# --------------------------------------------------------------------------- OpenPose 67 -> parts

BODY9_FROM_OPENPOSE67 = np.arange(0, 9)       # nose, neck, R sh/el/wr, L sh/el/wr, mid hip (paper "NIA body9")
BODY8_FROM_OPENPOSE67 = np.arange(1, 9)       # body9 without nose
EYE2_FROM_OPENPOSE67 = np.asarray([15, 16])   # REye, LEye
LEFT_HAND_FROM_OPENPOSE67 = slice(25, 46)
RIGHT_HAND_FROM_OPENPOSE67 = slice(46, 67)


def pear67_to_nia_body_hands51(joints: np.ndarray) -> np.ndarray:
    """Paper-compatible helper: (..., 67, 3) -> (..., 51, 3) body9 + left21 + right21."""
    joints = np.asarray(joints, dtype=np.float32)
    if joints.shape[-2:] != (67, 3):
        raise ValueError(f"expected (..., 67, 3), got {joints.shape}")
    return np.concatenate(
        [joints[..., 0:9, :], joints[..., LEFT_HAND_FROM_OPENPOSE67, :], joints[..., RIGHT_HAND_FROM_OPENPOSE67, :]],
        axis=-2,
    ).astype(np.float32, copy=False)


def openpose67_to_body8_eye2(joints67: np.ndarray) -> np.ndarray:
    joints67 = np.asarray(joints67, dtype=np.float32)
    return np.concatenate([joints67[:, BODY8_FROM_OPENPOSE67], joints67[:, EYE2_FROM_OPENPOSE67]], axis=1)


def openpose67_to_hands42(joints67: np.ndarray) -> np.ndarray:
    joints67 = np.asarray(joints67, dtype=np.float32)
    return np.concatenate([joints67[:, LEFT_HAND_FROM_OPENPOSE67], joints67[:, RIGHT_HAND_FROM_OPENPOSE67]], axis=1)


# --------------------------------------------------------------------------- SMPL-X 145 -> face70

# OpenPose/iBUG face68 order: 0:17 jaw, 17:22 right brow, 22:27 left brow, 27:36 nose,
# 36:42 right eye, 42:48 left eye, 48:60 outer mouth, 60:68 inner mouth.
FACE68_FROM_SMPLX_NAMES = (
    "right_contour_1", "right_contour_2", "right_contour_3", "right_contour_4", "right_contour_5",
    "right_contour_6", "right_contour_7", "right_contour_8", "contour_middle",
    "left_contour_8", "left_contour_7", "left_contour_6", "left_contour_5", "left_contour_4",
    "left_contour_3", "left_contour_2", "left_contour_1",
    "right_eye_brow1", "right_eye_brow2", "right_eye_brow3", "right_eye_brow4", "right_eye_brow5",
    "left_eye_brow5", "left_eye_brow4", "left_eye_brow3", "left_eye_brow2", "left_eye_brow1",
    "nose1", "nose2", "nose3", "nose4", "right_nose_2", "right_nose_1", "nose_middle", "left_nose_1", "left_nose_2",
    "right_eye1", "right_eye2", "right_eye3", "right_eye4", "right_eye5", "right_eye6",
    "left_eye4", "left_eye3", "left_eye2", "left_eye1", "left_eye6", "left_eye5",
    "right_mouth_1", "right_mouth_2", "right_mouth_3", "mouth_top", "left_mouth_3", "left_mouth_2", "left_mouth_1",
    "left_mouth_5", "left_mouth_4", "mouth_bottom", "right_mouth_4", "right_mouth_5",
    "right_lip_1", "right_lip_2", "lip_top", "left_lip_2", "left_lip_1", "left_lip_3", "lip_bottom", "right_lip_3",
)
RIGHT_EYE_FACE68 = tuple(range(36, 42))
LEFT_EYE_FACE68 = tuple(range(42, 48))


def _names(raw: Sequence[object]) -> tuple[str, ...]:
    return tuple(r.decode("utf-8") if isinstance(r, bytes) else str(r) for r in raw)


def face68_indices(smplx_joint_names: Sequence[object]) -> np.ndarray:
    names = _names(smplx_joint_names)
    lookup = {name: i for i, name in enumerate(names)}
    missing = [n for n in FACE68_FROM_SMPLX_NAMES if n not in lookup]
    if missing:
        raise ValueError(f"SMPL-X keypoint names lack face landmarks: {missing[:5]}...")
    return np.asarray([lookup[n] for n in FACE68_FROM_SMPLX_NAMES], dtype=np.int64)


def smplx145_to_face70(joints: np.ndarray, smplx_joint_names: Sequence[object]) -> tuple[np.ndarray, np.ndarray]:
    """(T, 145, 3) -> face70 (T, 70, 3) and per-landmark presence (T, 70) in {0, 1}."""
    joints = np.asarray(joints, dtype=np.float32)
    face68 = joints[:, face68_indices(smplx_joint_names)]
    face70 = np.empty((face68.shape[0], 70, 3), dtype=np.float32)
    face70[:, :68] = face68
    face70[:, 68] = np.nanmean(face68[:, RIGHT_EYE_FACE68], axis=1)
    face70[:, 69] = np.nanmean(face68[:, LEFT_EYE_FACE68], axis=1)
    presence68 = np.isfinite(face68).all(axis=-1).astype(np.float32)
    presence70 = np.empty((face68.shape[0], 70), dtype=np.float32)
    presence70[:, :68] = presence68
    presence70[:, 68] = presence68[:, RIGHT_EYE_FACE68].min(axis=1)
    presence70[:, 69] = presence68[:, LEFT_EYE_FACE68].min(axis=1)
    return face70, presence70


# --------------------------------------------------------------------------- frames


def camera_to_plot(points: np.ndarray) -> np.ndarray:
    """Camera frame (x right, y down, z forward) -> plot frame (x right, y depth, z up)."""
    points = np.asarray(points, dtype=np.float32)
    out = np.empty_like(points)
    out[..., 0] = points[..., 0]
    out[..., 1] = points[..., 2]
    out[..., 2] = -points[..., 1]
    return out


def convert(
    smplx_joints_3d: np.ndarray,
    smplx_joint_names: Sequence[object],
    presence_threshold: float = 0.5,
    legacy_face_frame: bool = False,
) -> dict[str, np.ndarray]:
    """Full conversion of a PEAR sequence to the common arrays.

    ``legacy_face_frame=True`` leaves face70 in PEAR's raw frame, reproducing the paper's
    exported ``openpose70_plot_xyz`` arrays (which were never rotated); the default rotates
    the face into the same plot frame as body and hands.
    """
    joints67 = smplx145_to_openpose67(smplx_joints_3d)
    body8_eye2 = camera_to_plot(openpose67_to_body8_eye2(joints67))
    hands42 = camera_to_plot(openpose67_to_hands42(joints67))
    face70, presence70 = smplx145_to_face70(smplx_joints_3d, smplx_joint_names)
    if not legacy_face_frame:
        face70 = camera_to_plot(face70)

    finite = lambda a: np.isfinite(a).all(axis=(1, 2))  # noqa: E731
    left_valid = finite(hands42[:, :21])
    right_valid = finite(hands42[:, 21:])
    face_valid = finite(face70) & (presence70 >= presence_threshold).all(axis=1)
    return {
        "body8_eye2_xyz": body8_eye2,
        "body_valid": finite(body8_eye2),
        "hands42_xyz": hands42,
        "left_hand_valid": left_valid,
        "right_hand_valid": right_valid,
        "hands_valid": left_valid & right_valid,
        "face70_xyz": face70,
        "face_valid": face_valid,
    }
