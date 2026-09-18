"""MPJPE, PA-MPJPE, and pooled summaries. Used only when ground truth is available.

Protocol (ICCAS 2026 paper): for the whole-body set both prediction and ground
truth are root-aligned at the neck, the prediction is scaled once per video by
the median body8 bone-length ratio, then per-frame MPJPE and PA-MPJPE are
computed. Part-wise PA-MPJPE is computed on the raw part. Aggregates pool all
evaluated frames across videos before taking the mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from .fusion import BODY8_EDGES, FusionResult, finite_frames, median_bone_scale

# NIA 121-joint ground-truth layout: body9 (nose, neck, R sh/el/wr, L sh/el/wr, mid hip),
# left21, right21, face70 (68 landmarks + right eye, left eye).
GT_BODY8 = np.arange(1, 9)
GT_LEFT = slice(9, 30)
GT_RIGHT = slice(30, 51)
GT_FACE70 = slice(51, 121)
GT_EYE2 = np.asarray([119, 120])


@dataclass(frozen=True)
class GroundTruth:
    body8: np.ndarray
    eye2: np.ndarray
    hands42: np.ndarray
    face70: np.ndarray

    @classmethod
    def from_nia121(cls, gt: np.ndarray) -> "GroundTruth":
        gt = np.asarray(gt, dtype=np.float32)
        return cls(
            body8=gt[:, GT_BODY8],
            eye2=gt[:, GT_EYE2],
            hands42=np.concatenate([gt[:, GT_LEFT], gt[:, GT_RIGHT]], axis=1),
            face70=gt[:, GT_FACE70],
        )

    @property
    def wholebody120(self) -> np.ndarray:
        return np.concatenate([self.body8, self.hands42, self.face70], axis=1)


# --------------------------------------------------------------------------- primitives


def root_align(points: np.ndarray, root_index: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    return points - np.take(points, [root_index], axis=-2)


def mpjpe(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Per-frame mean joint error, (T,)."""
    return np.linalg.norm(pred - gt, axis=-1).mean(axis=-1)


def procrustes_align(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Similarity-transform ``pred`` (J, 3) onto ``target`` (J, 3)."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mu_pred = pred.mean(axis=0)
    mu_target = target.mean(axis=0)
    x = pred - mu_pred
    y = target - mu_target
    if np.linalg.norm(x) < 1e-8 or np.linalg.norm(y) < 1e-8:
        return pred.astype(np.float32)
    u, s, vt = np.linalg.svd(x.T @ y)
    r = u @ vt
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        s[-1] *= -1
        r = u @ vt
    scale = float(np.sum(s) / np.sum(x**2))
    return (scale * (x @ r) + mu_target).astype(np.float32)


def pa_mpjpe(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Per-frame Procrustes-aligned MPJPE, (T,)."""
    errors = np.full((gt.shape[0],), np.nan, dtype=np.float32)
    for t in range(gt.shape[0]):
        aligned = procrustes_align(pred[t], gt[t])
        errors[t] = np.linalg.norm(aligned - gt[t], axis=-1).mean()
    return errors


def summarize(errors: np.ndarray, unit_scale: float = 1000.0) -> dict[str, float]:
    errors = np.asarray(errors, dtype=np.float64)
    errors = errors[np.isfinite(errors)]
    if errors.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "p95": float("nan"), "std": float("nan"), "n": 0}
    errors = errors * float(unit_scale)
    return {
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "p90": float(np.percentile(errors, 90)),
        "p95": float(np.percentile(errors, 95)),
        "std": float(np.std(errors)),
        "n": int(errors.size),
    }


# --------------------------------------------------------------------------- protocol


def body8_metric_scale(gt_body8: np.ndarray, pred_body8: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 1.0
    return median_bone_scale(root_align(gt_body8[mask], 0), root_align(pred_body8[mask], 0), BODY8_EDGES)


def evaluate_joint_set(
    gt: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    gt_body8: np.ndarray,
    pred_body8: np.ndarray,
    root_index: int = 0,
    scale_by_body8: bool = True,
) -> dict[str, np.ndarray | float | int]:
    gt_eval = root_align(gt, root_index)
    pred_eval = root_align(pred, root_index)
    scale = body8_metric_scale(gt_body8, pred_body8, mask) if scale_by_body8 else 1.0
    pred_eval = pred_eval * np.float32(scale)
    final = mask & finite_frames(gt_eval) & finite_frames(pred_eval)
    if not np.any(final):
        empty = np.asarray([], dtype=np.float32)
        return {"frames": 0, "scale": scale, "mpjpe": empty, "pa_mpjpe": empty}
    return {
        "frames": int(final.sum()),
        "scale": float(scale),
        "mpjpe": mpjpe(gt_eval[final], pred_eval[final]),
        "pa_mpjpe": pa_mpjpe(gt_eval[final], pred_eval[final]),
    }


def part_pa(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> np.ndarray:
    final = mask & finite_frames(gt) & finite_frames(pred)
    if not np.any(final):
        return np.asarray([], dtype=np.float32)
    return pa_mpjpe(gt[final], pred[final])


ERROR_NAMES = ("wb_mpjpe", "wb_pa_mpjpe", "body_eye10_pa", "hands42_pa", "face70_pa")


def evaluate_fusion(gt: GroundTruth, fused: FusionResult) -> dict[str, np.ndarray]:
    """Per-frame error arrays for one video (complete-case frames only)."""
    frame_count = min(gt.body8.shape[0], fused.num_frames)
    mask = fused.valid[:frame_count]
    gt_body8, pred_body8 = gt.body8[:frame_count], fused.body8[:frame_count]
    wb = evaluate_joint_set(gt.wholebody120[:frame_count], fused.wholebody120[:frame_count], mask, gt_body8, pred_body8)
    gt_body_eye10 = np.concatenate([gt_body8, gt.eye2[:frame_count]], axis=1)
    pred_body_eye10 = np.concatenate([pred_body8, fused.eye2[:frame_count]], axis=1)
    return {
        "wb_mpjpe": wb["mpjpe"],
        "wb_pa_mpjpe": wb["pa_mpjpe"],
        "body_eye10_pa": part_pa(gt_body_eye10, pred_body_eye10, mask),
        "hands42_pa": part_pa(gt.hands42[:frame_count], fused.hands42[:frame_count], mask),
        "face70_pa": part_pa(gt.face70[:frame_count], fused.face70[:frame_count], mask),
    }


def aggregate(per_video: Iterable[Mapping[str, np.ndarray]], unit_scale: float = 1000.0) -> dict[str, dict[str, float]]:
    """Pool per-frame errors across videos, then summarize each metric."""
    rows = list(per_video)
    out: dict[str, dict[str, float]] = {}
    for name in ERROR_NAMES:
        values = [np.asarray(r[name]) for r in rows if name in r and np.asarray(r[name]).size]
        out[name] = summarize(np.concatenate(values) if values else np.asarray([]), unit_scale)
    return out
