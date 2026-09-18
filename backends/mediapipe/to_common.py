"""Pure-numpy mappings from MediaPipe Tasks landmarks to the PAF-Pose common layout.

Ported from the ICCAS 2026 paper workspace:
  models/body/mediapipe/mediapipe_to_nia_body.py  (pose33 -> NIA body9)
  models/hand/mediapipe/mediapipe_to_nia_hand.py  (hand21 order, identity)
  models/face/mediapipe/mediapipe_to_openpose_face.py (face478 -> approximate OpenPose face70)
"""

from __future__ import annotations

from typing import Mapping, Tuple

import numpy as np

MEDIAPIPE_POSE_NAMES = (
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
    "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
)
MP = {name: index for index, name in enumerate(MEDIAPIPE_POSE_NAMES)}

NIA_BODY9_NAMES = (
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "mid_hip",
)
NIA_BODY9_MAPPING = {
    "nose": ("copy", (MP["nose"],)),
    "neck": ("midpoint", (MP["left_shoulder"], MP["right_shoulder"])),
    "right_shoulder": ("copy", (MP["right_shoulder"],)),
    "right_elbow": ("copy", (MP["right_elbow"],)),
    "right_wrist": ("copy", (MP["right_wrist"],)),
    "left_shoulder": ("copy", (MP["left_shoulder"],)),
    "left_elbow": ("copy", (MP["left_elbow"],)),
    "left_wrist": ("copy", (MP["left_wrist"],)),
    "mid_hip": ("midpoint", (MP["left_hip"], MP["right_hip"])),
}

# eye2 order in the common layout: right eye, left eye (pose landmarks 5 and 2).
POSE_EYE2_INDICES = (MP["right_eye"], MP["left_eye"])

# MediaPipe hand landmark order already matches OpenPose hand21 (wrist, thumb..pinky).
HAND21_NAMES = (
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)

# Approximate OpenPose face70 from FaceMesh 478 (468 mesh + 10 iris). 68 iBUG points + two iris centers.
FACE478_TO_OPENPOSE70 = np.asarray(
    [
        234, 93, 132, 58, 172, 136, 150, 149, 152, 378, 379, 365, 397, 288, 361, 323, 454,  # 0-16 jaw
        70, 63, 105, 66, 107, 336, 296, 334, 293, 300,                                     # 17-26 brows
        168, 6, 197, 4, 98, 97, 2, 326, 327,                                               # 27-35 nose
        33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,                         # 36-47 eyes
        61, 185, 40, 0, 270, 409, 291, 375, 321, 17, 91, 146,                              # 48-59 outer mouth
        78, 191, 80, 13, 310, 415, 308, 324,                                               # 60-67 inner mouth
        468, 473,                                                                          # 68-69 iris centers
    ],
    dtype=np.int64,
)


# --------------------------------------------------------------------------- landmark lists


def landmarks_to_xyz_score(landmark_list, count: int, score_attr: str = "visibility", default_score: float = 1.0):
    """MediaPipe landmark list -> ((count,3) xyz, (count,) score). Missing -> NaN / 0."""
    xyz = np.full((count, 3), np.nan, dtype=np.float32)
    score = np.zeros((count,), dtype=np.float32)
    if landmark_list is None:
        return xyz, score
    landmarks = landmark_list if isinstance(landmark_list, (list, tuple)) else getattr(landmark_list, "landmark", [])
    for index, landmark in enumerate(landmarks[:count]):
        xyz[index] = (float(landmark.x), float(landmark.y), float(landmark.z))
        value = getattr(landmark, score_attr, None)
        score[index] = float(value) if value is not None else float(default_score)
    return xyz, score


# --------------------------------------------------------------------------- body


def pose33_to_body9(xyz33: np.ndarray, score33: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
    xyz33 = np.asarray(xyz33, dtype=np.float32)
    if xyz33.shape != (33, 3):
        raise ValueError(f"expected (33, 3), got {xyz33.shape}")
    score33 = np.ones((33,), dtype=np.float32) if score33 is None else np.asarray(score33, dtype=np.float32)
    out = np.full((9, 3), np.nan, dtype=np.float32)
    out_score = np.zeros((9,), dtype=np.float32)
    for i, name in enumerate(NIA_BODY9_NAMES):
        mode, idx = NIA_BODY9_MAPPING[name]
        if mode == "copy":
            out[i] = xyz33[idx[0]]
            out_score[i] = score33[idx[0]]
        else:
            pts = xyz33[list(idx)]
            if np.isfinite(pts).all():
                out[i] = pts.mean(axis=0)
                out_score[i] = min(float(score33[idx[0]]), float(score33[idx[1]]))
    return out, out_score


def pose33_to_body8_eye2(xyz33: np.ndarray, score33: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """(33,3) pose -> (10,3) body8 (NIA body9 without nose) + eye2 (right eye, left eye)."""
    body9, body9_score = pose33_to_body9(xyz33, score33)
    score33 = np.ones((33,), dtype=np.float32) if score33 is None else np.asarray(score33, dtype=np.float32)
    idx = list(POSE_EYE2_INDICES)
    xyz = np.concatenate([body9[1:9], np.asarray(xyz33, dtype=np.float32)[idx]], axis=0)
    score = np.concatenate([body9_score[1:9], score33[idx]], axis=0)
    return xyz, score


# --------------------------------------------------------------------------- hands


def mediapipe21_to_openpose_hand21(xyz21: np.ndarray) -> np.ndarray:
    xyz21 = np.asarray(xyz21, dtype=np.float32)
    if xyz21.shape != (21, 3):
        raise ValueError(f"expected (21, 3), got {xyz21.shape}")
    return xyz21.copy()


# --------------------------------------------------------------------------- face


def face478_to_face70(xyz478: np.ndarray) -> np.ndarray:
    xyz478 = np.asarray(xyz478, dtype=np.float32)
    if xyz478.shape != (478, 3):
        raise ValueError(f"expected (478, 3), got {xyz478.shape}")
    return xyz478[FACE478_TO_OPENPOSE70].copy()


def crop_normalized_to_frame_normalized(points: np.ndarray, crop: Mapping[str, int]) -> np.ndarray:
    """Normalized coords inside a crop -> normalized coords of the full frame (z scaled by crop/frame width)."""
    points = np.asarray(points, dtype=np.float32)
    out = points.copy()
    frame_w = float(crop["frame_width"])
    frame_h = float(crop["frame_height"])
    crop_w = float(crop["x1"] - crop["x0"])
    crop_h = float(crop["y1"] - crop["y0"])
    out[..., 0] = (float(crop["x0"]) + points[..., 0] * crop_w) / frame_w
    out[..., 1] = (float(crop["y0"]) + points[..., 1] * crop_h) / frame_h
    out[..., 2] = points[..., 2] * (crop_w / frame_w)
    return out


def normalized_to_pixels(points: np.ndarray, frame_width: int, frame_height: int) -> np.ndarray:
    """Full-frame normalized xyz -> pixel units (x*W, y*H, z*W) so the face keeps its aspect ratio."""
    points = np.asarray(points, dtype=np.float32)
    out = points.copy()
    out[..., 0] *= float(frame_width)
    out[..., 1] *= float(frame_height)
    out[..., 2] *= float(frame_width)
    return out
