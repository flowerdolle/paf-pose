"""TEASER / FLAME native landmarks -> common face70 layout (pure numpy).

TEASER's FLAME head returns ``landmarks_fan_3d`` with 68 iBUG/FAN landmarks in a
face-local, camera-like frame (x right, y down, z toward the camera). The paper
appends two eye centers (mean of the six eye-contour points per eye) and maps
the frame to the "plot" convention with ``[x, z, -y]``.

Note: FLAME coordinates are face-local reconstruction coordinates, not metric
world coordinates. They are rescaled by the host fusion step (inter-eye
distance ratio to the body source), so only the shape matters here.
"""

from __future__ import annotations

import numpy as np

RIGHT_EYE = slice(36, 42)  # iBUG right-eye contour (subject's right)
LEFT_EYE = slice(42, 48)


def fan68_to_face70(fan68_xyz: np.ndarray) -> np.ndarray:
    """(..., 68, 3) -> (..., 70, 3): append right and left eye centers."""
    fan68_xyz = np.asarray(fan68_xyz, dtype=np.float32)
    if fan68_xyz.shape[-2:] != (68, 3):
        raise ValueError(f"expected (..., 68, 3) FAN landmarks, got {fan68_xyz.shape}")
    out = np.full(fan68_xyz.shape[:-2] + (70, 3), np.nan, dtype=np.float32)
    out[..., :68, :] = fan68_xyz
    out[..., 68, :] = np.nanmean(fan68_xyz[..., RIGHT_EYE, :], axis=-2)
    out[..., 69, :] = np.nanmean(fan68_xyz[..., LEFT_EYE, :], axis=-2)
    return out


def flame_to_plot(points: np.ndarray) -> np.ndarray:
    """FLAME frame (x right, y up, z toward the camera) -> plot frame (x right, y depth, z up): [x, -z, y].

    The paper's export used [x, z, -y], which leaves the face rotated 180 degrees about x
    relative to the body; see backends/teaser/README.md. Use :func:`flame_to_plot_legacy`
    to reproduce that output.
    """
    points = np.asarray(points, dtype=np.float32)
    out = np.empty_like(points)
    out[..., 0] = points[..., 0]
    out[..., 1] = -points[..., 2]
    out[..., 2] = points[..., 1]
    return out


def flame_to_plot_legacy(points: np.ndarray) -> np.ndarray:
    """The paper's original transform [x, z, -y] (face upside-down relative to the body)."""
    points = np.asarray(points, dtype=np.float32)
    out = np.empty_like(points)
    out[..., 0] = points[..., 0]
    out[..., 1] = points[..., 2]
    out[..., 2] = -points[..., 1]
    return out


def fan68_to_face70_plot(fan68_xyz: np.ndarray, legacy_face_frame: bool = False) -> np.ndarray:
    """Full conversion used by the adapter: FAN68 (FLAME frame) -> face70 in plot frame."""
    transform = flame_to_plot_legacy if legacy_face_frame else flame_to_plot
    return transform(fan68_to_face70(fan68_xyz))


if __name__ == "__main__":  # self-check against a paper-era raw output
    import argparse

    p = argparse.ArgumentParser(description="Verify the mapping against a run_teaser_face.py npz.")
    p.add_argument("npz")
    a = p.parse_args()
    d = np.load(a.npz, allow_pickle=True)
    ours = fan68_to_face70_plot(d["fan68_flame_xyz"], legacy_face_frame=True)  # paper files used the legacy frame
    ref = d["openpose70_plot_xyz"]
    both = np.isfinite(ours).all(axis=(1, 2)) & np.isfinite(ref).all(axis=(1, 2))
    print("frames", ours.shape[0], "compared", int(both.sum()), "max_abs_diff", float(np.abs(ours[both] - ref[both]).max()))
