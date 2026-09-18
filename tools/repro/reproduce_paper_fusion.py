#!/usr/bin/env python3
"""Reproduce rows of the ICCAS 2026 whole-body fusion table with pafpose.fusion / pafpose.metrics.

Reads the paper's raw per-model outputs (SL_MST/papers/iccas2026/outputs), converts them to the
common layout in-line, fuses with the GT-free fusion module, evaluates against the NIA GT tensor,
and compares against the numbers reported in the paper.

    .venv/bin/python tools/repro/reproduce_paper_fusion.py --hand sam
    .venv/bin/python tools/repro/reproduce_paper_fusion.py --hand wilor
    .venv/bin/python tools/repro/reproduce_paper_fusion.py --face teaser

``--face-frame paper`` (default) uses the face arrays exactly as the paper exported them.
``--face-frame corrected`` rotates them into the body's plot frame first: the paper's PEAR and
TEASER face exports were left in a frame rotated relative to the body (PEAR: camera frame;
TEASER: 180 degrees about x), which inflates whole-body PA-MPJPE.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pafpose import fusion, metrics  # noqa: E402
from pafpose.schema import camera_to_plot  # noqa: E402

VIEWS = ("D", "F", "L", "U", "R")
STEM_RE = re.compile(r"NIA_SL_WORD(?P<word>\d+)_REAL(?P<signer>\d+)_(?P<view>[DFLUR])")
PRESENCE = 0.5

# Paper table (100 videos: words 1-20, one subject, five views). Values in mm.
EXPECTED = {
    ("sam", "sam", "pear"): {"wb_mpjpe": 67.161, "wb_pa_mpjpe": 46.330, "hands42_pa": 11.586, "face70_pa": 8.295},
    ("sam", "wilor", "pear"): {"wb_mpjpe": 68.554, "wb_pa_mpjpe": 46.629, "hands42_pa": 13.632, "face70_pa": 8.295},
    ("sam", "sam", "teaser"): {"wb_mpjpe": 79.892, "wb_pa_mpjpe": 65.994, "hands42_pa": 11.579, "face70_pa": 9.818},
    ("sam", "wilor", "teaser"): {"wb_mpjpe": 81.289, "wb_pa_mpjpe": 65.666, "hands42_pa": 13.621, "face70_pa": 9.818},
}

# Same rows with the face rotated into the body frame (computed 2026-09-16 with this script).
EXPECTED_CORRECTED = {
    ("sam", "sam", "pear"): {"wb_pa_mpjpe": 18.842, "wb_mpjpe": 63.796},
    ("sam", "sam", "teaser"): {"wb_pa_mpjpe": 20.026, "wb_mpjpe": 64.731},
}

FACE_FRAME_FIX = {
    "pear": camera_to_plot,                                            # export was in PEAR's camera frame
    "teaser": lambda f: f * np.asarray([1, -1, -1], dtype=np.float32),  # export was rotated 180 deg about x
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--iccas-root", type=Path, default=Path("/nfs2/sangvv0n/SL_MST/papers/iccas2026"))
    p.add_argument("--body", choices=("sam",), default="sam")
    p.add_argument("--hand", choices=("sam", "wilor"), default="sam")
    p.add_argument("--face", choices=("pear", "teaser"), default="pear")
    p.add_argument("--face-frame", choices=("paper", "corrected"), default="paper")
    p.add_argument("--word-start", type=int, default=1)
    p.add_argument("--word-end", type=int, default=20)
    p.add_argument("--views", nargs="+", default=list(VIEWS))
    p.add_argument("--tolerance-mm", type=float, default=0.01)
    return p.parse_args()


# ---------------------------------------------------------------- converters (stage-3 adapters will own these)


def load_sam(path: Path):
    d = np.load(path)
    bh51 = camera_to_plot(np.asarray(d["nia_body_hands51_xyz"], dtype=np.float32))
    op53 = camera_to_plot(np.asarray(d["openpose53_xyz"], dtype=np.float32))
    body9, left, right, eye2 = bh51[:, :9], bh51[:, 9:30], bh51[:, 30:51], op53[:, [9, 10]]
    valid = np.asarray(d["frame_valid"], dtype=bool)
    for part in (body9, left, right, eye2):
        valid &= fusion.finite_frames(part)
    body8_eye2 = np.concatenate([body9[:, 1:9], eye2], axis=1)
    hands42 = np.concatenate([left, right], axis=1)
    return body8_eye2, valid, hands42


def load_wilor(path: Path):
    d = np.load(path, allow_pickle=True)
    left = camera_to_plot(np.asarray(d["left_hand21_camera_xyz"], dtype=np.float32))
    right = camera_to_plot(np.asarray(d["right_hand21_camera_xyz"], dtype=np.float32))
    lp = np.asarray(d["left_hand21_presence"], dtype=np.float32)
    rp = np.asarray(d["right_hand21_presence"], dtype=np.float32)
    lv = np.asarray(d["left_frame_valid"], dtype=bool) & (lp >= PRESENCE).all(axis=1)
    rv = np.asarray(d["right_frame_valid"], dtype=bool) & (rp >= PRESENCE).all(axis=1)
    return np.concatenate([left, right], axis=1), lv, rv


def load_face(path: Path):
    """PEAR and TEASER face exports share the same keys."""
    d = np.load(path, allow_pickle=True)
    face = np.asarray(d["openpose70_plot_xyz"], dtype=np.float32)
    presence = np.asarray(d["openpose70_presence"], dtype=np.float32)
    valid = np.asarray(d["frame_valid"], dtype=bool) & fusion.finite_frames(face) & (presence >= PRESENCE).all(axis=1)
    return face, valid


def main() -> int:
    args = parse_args()
    root = args.iccas_root
    sam_dir = root / "outputs" / "sam_body4d_hand_bbox" / "WORD_01_FUDLR_0001_0200_full"
    wilor_dir = root / "outputs" / "wilor_hand_sam_bbox" / "WORD_01_FUDLR_0001_0200_full"
    face_dirs = {
        "pear": (root / "outputs" / "pear_face" / "WORD_01_FUDLR_0001_0020_full", "{stem}_pear_face70.npz"),
        "teaser": (root / "outputs" / "teaser_face" / "WORD_01_FUDLR_0001_0020_full", "{stem}_teaser_face.npz"),
    }
    face_dir, face_pattern = face_dirs[args.face]
    gt = np.load(root / "datasets" / "word_01_scale_corrected.npy", mmap_mode="r")
    lengths = np.load(root / "datasets" / "word_01_scale_corrected_frame_lengths.npy", mmap_mode="r")

    stems = []
    for path in sorted(sam_dir.glob("*_sam_body4d_body.npz")):
        m = STEM_RE.search(path.name)
        if not m:
            continue
        word, view = int(m.group("word")), m.group("view")
        if args.word_start <= word <= args.word_end and view in args.views:
            stems.append((m.group(0), word, int(m.group("signer")) - 1, view))

    per_video = []
    for stem, word, signer, view in stems:
        body8_eye2, body_valid, sam_hands = load_sam(sam_dir / f"{stem}_sam_body4d_body.npz")
        if args.hand == "sam":
            hands42, lv, rv = sam_hands, body_valid.copy(), body_valid.copy()
        else:
            hands42, lv, rv = load_wilor(wilor_dir / f"{stem}_wilor_hand.npz")
        face70, fv = load_face(face_dir / face_pattern.format(stem=stem))
        if args.face_frame == "corrected":
            face70 = FACE_FRAME_FIX[args.face](face70)

        length = int(lengths[word - 1, signer, VIEWS.index(view)])
        gt121 = np.asarray(gt[word - 1, signer, VIEWS.index(view), :length, :121], dtype=np.float32)
        T = min(length, body8_eye2.shape[0], hands42.shape[0], face70.shape[0])
        fused = fusion.fuse_arrays(
            body8_eye2[:T], body_valid[:T], hands42[:T], lv[:T], rv[:T], face70[:T], fv[:T],
            anchor_hands42=sam_hands[:T],
        )
        per_video.append(metrics.evaluate_fusion(metrics.GroundTruth.from_nia121(gt121[:T]), fused))

    agg = metrics.aggregate(per_video)
    combo = (args.body, args.hand, args.face)
    expected = (EXPECTED if args.face_frame == "paper" else EXPECTED_CORRECTED).get(combo, {})
    ref_label = "paper" if args.face_frame == "paper" else "corrected"
    print(f"combination body={args.body} hand={args.hand} face={args.face} face_frame={args.face_frame}  videos={len(per_video)}")
    print(f"{'metric':14} {'pafpose':>10} {ref_label:>10} {'diff':>8}")
    ok = True
    for name in metrics.ERROR_NAMES:
        value = agg[name]["mean"]
        ref = expected.get(name)
        diff = "" if ref is None else f"{value - ref:+.3f}"
        flag = "" if ref is None or abs(value - ref) <= args.tolerance_mm else "  <-- MISMATCH"
        ok &= not flag
        print(f"{name:14} {value:10.3f} {'' if ref is None else f'{ref:10.3f}':>10} {diff:>8}{flag}")
    print(f"frames evaluated: {agg['wb_pa_mpjpe']['n']}")
    print("RESULT:", "match" if ok else "mismatch")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
