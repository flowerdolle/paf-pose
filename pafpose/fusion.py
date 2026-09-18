"""Part-aware fusion: attach hand and face sources to a body anchor.

Ported from SL_MST/papers/iccas2026/models/hand/fusion (ICCAS 2026 paper) with
the ground-truth dependency removed.

* Each hand is wrist-centered, scaled once per video by the median bone-length
  ratio to the body source's own hand (when the body backend produced hands),
  and translated to the body source's wrist. Without body-source hands the
  hand keeps its own scale.
* The face is eye-midpoint-centered, scaled once per video by the median
  inter-eye-distance ratio to the body source's eyes, and translated to the
  body source's eye midpoint.
* No rotation alignment, learned refinement, or temporal smoothing.

Layouts (see :mod:`pafpose.schema`): body8_eye2 (10), hands42 (left21+right21),
face70 (68 landmarks + right eye center + left eye center).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import schema

# body8 indices: 0 neck, 1 R shoulder, 2 R elbow, 3 R wrist, 4 L shoulder, 5 L elbow, 6 L wrist, 7 mid hip
BODY8_EDGES = np.asarray([(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7)], dtype=np.int64)
RIGHT_WRIST, LEFT_WRIST = 3, 6
EYE2_SLICE = slice(8, 10)

HAND21_EDGES = np.asarray(
    [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
     (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)],
    dtype=np.int64,
)
LEFT_HAND, RIGHT_HAND = slice(0, 21), slice(21, 42)
FACE_EYES = [68, 69]  # right eye center, left eye center inside face70

FUSED_KEYS = ("body8_xyz", "eye2_xyz", "hands42_xyz", "face70_xyz", "wholebody120_xyz")


@dataclass(frozen=True)
class FusionConfig:
    hand_scale: str = "anchor-median-bone"   # or "none"
    face_scale: str = "anchor-eye-distance"  # or "none"


@dataclass
class FusionResult:
    body8: np.ndarray          # (T, 8, 3)
    eye2: np.ndarray           # (T, 2, 3)
    hands42: np.ndarray        # (T, 42, 3) attached
    face70: np.ndarray         # (T, 70, 3) attached
    body_valid: np.ndarray     # (T,)
    left_valid: np.ndarray     # (T,)
    right_valid: np.ndarray    # (T,)
    face_valid: np.ndarray     # (T,)
    scales: dict[str, float] = field(default_factory=dict)
    config: FusionConfig = field(default_factory=FusionConfig)

    @property
    def num_frames(self) -> int:
        return int(self.body8.shape[0])

    @property
    def valid(self) -> np.ndarray:
        """Complete-case mask: body, both hands, and face all valid."""
        return self.body_valid & self.left_valid & self.right_valid & self.face_valid

    @property
    def wholebody120(self) -> np.ndarray:
        """body8 + hands42 + face70; eye2 is used for attachment only."""
        return np.concatenate([self.body8, self.hands42, self.face70], axis=1)

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "body8_xyz": self.body8,
            "eye2_xyz": self.eye2,
            "hands42_xyz": self.hands42,
            "face70_xyz": self.face70,
            "wholebody120_xyz": self.wholebody120,
            "valid": self.valid,
            "body_valid": self.body_valid,
            "left_hand_valid": self.left_valid,
            "right_hand_valid": self.right_valid,
            "face_valid": self.face_valid,
        }

    def save(self, out_dir: Path, extra_meta: dict | None = None) -> tuple[Path, Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = out_dir / "fused.npz"
        json_path = out_dir / "fusion.json"
        np.savez_compressed(npz_path, **self.arrays())
        meta = {
            "num_frames": self.num_frames,
            "frames_complete": int(self.valid.sum()),
            "frames_body_valid": int(self.body_valid.sum()),
            "frames_left_hand_valid": int(self.left_valid.sum()),
            "frames_right_hand_valid": int(self.right_valid.sum()),
            "frames_face_valid": int(self.face_valid.sum()),
            "scales": self.scales,
            "config": asdict(self.config),
            "layout": {
                "wholebody120_xyz": "body8 + hands42(left21, right21) + face70",
                "body8": list(schema.BODY8_EYE2_NAMES[:8]),
                "eye2": list(schema.BODY8_EYE2_NAMES[8:]),
            },
        }
        if extra_meta:
            meta.update(extra_meta)
        json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return npz_path, json_path


# --------------------------------------------------------------------------- primitives


def bone_lengths(sequence: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.linalg.norm(sequence[:, edges[:, 0], :] - sequence[:, edges[:, 1], :], axis=-1)


def median_bone_scale(reference: np.ndarray, pred: np.ndarray, edges: np.ndarray) -> float:
    """median(reference bone lengths) / median(pred bone lengths); 1.0 when undefined."""
    ref_lengths = bone_lengths(reference, edges)
    pred_lengths = bone_lengths(pred, edges)
    ref_valid = ref_lengths[np.isfinite(ref_lengths) & (ref_lengths > 1e-8)]
    pred_valid = pred_lengths[np.isfinite(pred_lengths) & (pred_lengths > 1e-8)]
    if ref_valid.size == 0 or pred_valid.size == 0:
        return 1.0
    ref_median = float(np.nanmedian(ref_valid))
    pred_median = float(np.nanmedian(pred_valid))
    if not np.isfinite(ref_median) or not np.isfinite(pred_median) or pred_median <= 1e-8:
        return 1.0
    return ref_median / pred_median


def finite_frames(points: np.ndarray) -> np.ndarray:
    return np.isfinite(points).all(axis=tuple(range(1, points.ndim)))


def attach_hand(
    hand21: np.ndarray,
    anchor_hand21: np.ndarray,
    valid: np.ndarray,
    scale_mode: str = "anchor-median-bone",
) -> tuple[np.ndarray, float]:
    """Wrist-center ``hand21``, scale it to the anchor hand, move it onto the anchor wrist.

    ``anchor_hand21`` may be a real hand from the body backend or the wrist
    repeated 21 times (then its bone lengths are zero and no scaling happens).
    Frames outside ``valid`` become NaN.
    """
    fused = np.full_like(anchor_hand21, np.nan, dtype=np.float32)
    valid = valid & finite_frames(hand21) & finite_frames(anchor_hand21)
    if not np.any(valid):
        return fused, 1.0
    centered = hand21 - hand21[:, 0:1, :]
    scale = 1.0
    if scale_mode == "anchor-median-bone":
        scale = median_bone_scale(anchor_hand21[valid] - anchor_hand21[valid, 0:1, :], centered[valid], HAND21_EDGES)
    attached = centered * np.float32(scale) + anchor_hand21[:, 0:1, :]
    fused[valid] = attached[valid]
    return fused, float(scale)


def attach_face(
    face70: np.ndarray,
    anchor_eye2: np.ndarray,
    valid: np.ndarray,
    scale_mode: str = "anchor-eye-distance",
) -> tuple[np.ndarray, float]:
    """Eye-midpoint-center ``face70``, scale by eye distance ratio, move onto the anchor eye midpoint."""
    fused = np.full_like(face70, np.nan, dtype=np.float32)
    valid = valid & finite_frames(face70) & finite_frames(anchor_eye2)
    if not np.any(valid):
        return fused, 1.0
    pred_eye2 = face70[:, FACE_EYES, :]
    pred_center = np.nanmean(pred_eye2, axis=1, keepdims=True)
    anchor_center = np.nanmean(anchor_eye2, axis=1, keepdims=True)
    centered = face70 - pred_center
    scale = 1.0
    if scale_mode == "anchor-eye-distance":
        pred_dist = np.linalg.norm(pred_eye2[:, 0, :] - pred_eye2[:, 1, :], axis=-1)
        anchor_dist = np.linalg.norm(anchor_eye2[:, 0, :] - anchor_eye2[:, 1, :], axis=-1)
        ratio = anchor_dist[valid] / np.maximum(pred_dist[valid], 1e-8)
        ratio = ratio[np.isfinite(ratio) & (ratio > 1e-8)]
        scale = float(np.nanmedian(ratio)) if ratio.size else 1.0
    fused[valid] = centered[valid] * np.float32(scale) + anchor_center[valid]
    return fused, float(scale)


def wrist_only_anchor(body8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(left, right) hand anchors made of the wrist repeated 21 times."""
    left = np.repeat(body8[:, LEFT_WRIST:LEFT_WRIST + 1, :], 21, axis=1)
    right = np.repeat(body8[:, RIGHT_WRIST:RIGHT_WRIST + 1, :], 21, axis=1)
    return left.astype(np.float32, copy=False), right.astype(np.float32, copy=False)


# --------------------------------------------------------------------------- top level


def fuse_arrays(
    body8_eye2: np.ndarray,
    body_valid: np.ndarray,
    hands42: np.ndarray,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
    face70: np.ndarray,
    face_valid: np.ndarray,
    anchor_hands42: np.ndarray | None = None,
    config: FusionConfig | None = None,
) -> FusionResult:
    config = config or FusionConfig()
    frame_count = min(a.shape[0] for a in (body8_eye2, body_valid, hands42, left_valid, right_valid, face70, face_valid))
    if anchor_hands42 is not None:
        frame_count = min(frame_count, anchor_hands42.shape[0])

    body8_eye2 = np.asarray(body8_eye2, dtype=np.float32)[:frame_count]
    body8 = body8_eye2[:, :8]
    eye2 = body8_eye2[:, EYE2_SLICE]
    body_valid = np.asarray(body_valid, dtype=bool)[:frame_count] & finite_frames(body8_eye2)
    hands42 = np.asarray(hands42, dtype=np.float32)[:frame_count]
    face70 = np.asarray(face70, dtype=np.float32)[:frame_count]
    left_valid = np.asarray(left_valid, dtype=bool)[:frame_count]
    right_valid = np.asarray(right_valid, dtype=bool)[:frame_count]
    face_valid = np.asarray(face_valid, dtype=bool)[:frame_count]

    if anchor_hands42 is not None:
        anchor_hands42 = np.asarray(anchor_hands42, dtype=np.float32)[:frame_count]
        anchor_left, anchor_right = anchor_hands42[:, LEFT_HAND], anchor_hands42[:, RIGHT_HAND]
    else:
        anchor_left, anchor_right = wrist_only_anchor(body8)

    left, left_scale = attach_hand(hands42[:, LEFT_HAND], anchor_left, left_valid, config.hand_scale)
    right, right_scale = attach_hand(hands42[:, RIGHT_HAND], anchor_right, right_valid, config.hand_scale)
    face, face_scale = attach_face(face70, eye2, face_valid, config.face_scale)

    return FusionResult(
        body8=body8,
        eye2=eye2,
        hands42=np.concatenate([left, right], axis=1),
        face70=face,
        body_valid=body_valid,
        left_valid=left_valid & finite_frames(left),
        right_valid=right_valid & finite_frames(right),
        face_valid=face_valid & finite_frames(face),
        scales={"left_hand": left_scale, "right_hand": right_scale, "face": face_scale},
        config=config,
    )


def fuse_outputs(
    body: schema.BackendOutput,
    hand: schema.BackendOutput,
    face: schema.BackendOutput,
    config: FusionConfig | None = None,
) -> FusionResult:
    """Fuse three backend outputs (which may be the same object)."""
    anchor_hands = body.part("hand") if body.has("hand") else None
    return fuse_arrays(
        body8_eye2=body.part("body"),
        body_valid=body.valid("body"),
        hands42=hand.part("hand"),
        left_valid=hand.hand_side_valid("left"),
        right_valid=hand.hand_side_valid("right"),
        face70=face.part("face"),
        face_valid=face.valid("face"),
        anchor_hands42=anchor_hands,
        config=config,
    )
