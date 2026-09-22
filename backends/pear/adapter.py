#!/usr/bin/env python3
"""PAF-Pose backend adapter: PEAR (Pixel-aligned Expressive humAn mesh Recovery).

Runs PEAR's EHM pipeline on one single-person .mp4 and writes the common-layout
npz + meta.json (body8_eye2, hands42, face70). Re-implements the skeleton-only path
of the paper's ``mesh_inference`` without importing PEAR's Gradio ``app.py``:

    per frame : pad/resize to 256, ViT backbone + SMPL-X transformer head -> body/FLAME params, camera
    sequence  : Savitzky-Golay smoothing of the parameter sequences (window 7 body / 5 face / 7 camera)
    per frame : EHM_v2 forward -> 145 SMPL-X keypoints -> common layout (to_common.py)

Inside the image PEAR lives at /opt/pear (upstream commit e1aa1f7) and the license-gated
model files are symlinked from /weights (see README.md).

    python adapter.py --video /input/clip.mp4 --out /output --weights /weights
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

try:
    import pafpose_backend as pb
except ImportError:  # local development: backends/_common next to this backend dir
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
    import pafpose_backend as pb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import to_common  # noqa: E402

BACKEND = "pear"
UPSTREAM_COMMIT = "e1aa1f7"
DEFAULT_PEAR_ROOT = Path(os.environ.get("PEAR_ROOT", "/opt/pear"))
CHECKPOINT_RELATIVE = Path("pear") / "pear_model.pt"          # name on Hugging Face since 2026-09
CHECKPOINT_LEGACY = Path("pear") / "ehm_model_stage1.pt"      # name used by the paper runs
INPUT_SIZE = 256
SMOOTH_WINDOW_BODY = 7
SMOOTH_WINDOW_FACE = 5
SMOOTH_WINDOW_CAMERA = 7
SMOOTH_POLYORDER = 2
BODY_FIELDS = ("global_pose", "body_pose", "left_hand_pose", "right_hand_pose", "hand_scale", "head_scale", "exp", "shape")
FLAME_FIELDS = ("eye_pose_params", "pose_params", "jaw_params", "eyelid_params", "expression_params", "shape_params")


def parse_args():
    parser = pb.base_parser("PEAR body + hands + face adapter for PAF-Pose")
    parser.add_argument("--pear-root", type=Path, default=DEFAULT_PEAR_ROOT, help="PEAR repository root (assets/, configs/, models/)")
    parser.add_argument("--checkpoint", type=Path, default=None, help="EHM checkpoint; default <weights>/pear/pear_model.pt (or the legacy ehm_model_stage1.pt)")
    parser.add_argument("--no-smooth", action="store_true", help="disable the temporal Savitzky-Golay smoothing PEAR applies")
    parser.add_argument("--legacy-face-frame", action="store_true", help="leave face70 in PEAR's raw frame (paper export behaviour)")
    parser.add_argument("--save-raw", action="store_true", help="also store raw_smplx145_xyz / raw_openpose67_xyz / raw_cameras")
    return parser.parse_args()


def pad_and_resize(img: np.ndarray, target_size: int) -> np.ndarray:
    """Letterbox to a square canvas (ported from PEAR app.py)."""
    import cv2

    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    x0 = (target_size - new_w) // 2
    y0 = (target_size - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def smooth_sequence(seq: np.ndarray, window: int, polyorder: int = SMOOTH_POLYORDER) -> np.ndarray:
    """Savitzky-Golay along time; shrinks the window for short clips instead of failing."""
    from scipy.signal import savgol_filter

    T = seq.shape[0]
    window = min(window, T if T % 2 == 1 else T - 1)
    if window <= polyorder:
        return seq
    return savgol_filter(seq, window_length=window, polyorder=polyorder, axis=0, mode="interp")


def load_pear(pear_root: Path, checkpoint: Path, device):
    """Import PEAR modules from ``pear_root`` and build the EHM pipeline + body model."""
    import torch

    pear_root = pear_root.resolve()
    if not (pear_root / "configs" / "infer.yaml").is_file():
        raise FileNotFoundError(f"PEAR root {pear_root} has no configs/infer.yaml")
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"PEAR checkpoint not found: {checkpoint}\n"
            "Run backends/pear/download_weights.sh <weights-root> (HF BestWJH/PEAR_models/pear_model.pt)."
        )
    # PEAR resolves assets/ and configs/ relative to the working directory.
    os.chdir(pear_root)
    sys.path.insert(0, str(pear_root))

    from models.modules.ehm import EHM_v2
    from models.pipeline.ehm_pipeline import Ehm_Pipeline
    from utils.general_utils import ConfigDict, add_extra_cfgs

    cfg = add_extra_cfgs(ConfigDict(model_config_path=os.path.join("configs", "infer.yaml")))
    pipeline = Ehm_Pipeline(cfg)
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    pipeline.backbone.load_state_dict(state["backbone"], strict=False)
    pipeline.head.load_state_dict(state["head"], strict=False)
    ehm = EHM_v2("assets/FLAME", "assets/SMPLX")
    pipeline = pipeline.to(device).eval()
    ehm = ehm.to(device).eval()
    return pipeline, ehm


def main() -> int:
    args = parse_args()
    import torch

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    checkpoint = args.checkpoint or (args.weights / CHECKPOINT_RELATIVE)
    if args.checkpoint is None and not checkpoint.is_file() and (args.weights / CHECKPOINT_LEGACY).is_file():
        checkpoint = args.weights / CHECKPOINT_LEGACY
    stem = args.video.stem
    timer = pb.Timer()

    timer.start_load()
    pipeline, ehm = load_pear(args.pear_root, checkpoint, device)
    timer.end_load()

    reader = pb.VideoReader(args.video, max_frames=args.max_frames)
    body_seq: dict[str, list] = {k: [] for k in BODY_FIELDS}
    flame_seq: dict[str, list] = {k: [] for k in FLAME_FIELDS}
    cam_seq: list = []

    import cv2

    with torch.no_grad():
        for _, frame_bgr in reader:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            patch = pad_and_resize(rgb, INPUT_SIZE)
            x = torch.from_numpy(patch).to(device).float().div_(255.0).permute(2, 0, 1).unsqueeze(0)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timer.start_frame()
            out = pipeline(x)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timer.end_frame()
            for k in BODY_FIELDS:
                body_seq[k].append(out["body_param"][k].detach().cpu())
            for k in FLAME_FIELDS:
                flame_seq[k].append(out["flame_param"][k].detach().cpu())
            cam_seq.append(out["pd_cam"].detach().cpu())

    num_frames = len(cam_seq)
    if num_frames == 0:
        raise RuntimeError(f"no frames decoded from {args.video}")

    def stack(seq: list, window: int) -> torch.Tensor:
        arr = torch.cat(seq, dim=0).numpy()
        if not args.no_smooth:
            arr = smooth_sequence(arr, window)
        return torch.as_tensor(np.ascontiguousarray(arr), dtype=torch.float32, device=device)

    body = {k: stack(v, SMOOTH_WINDOW_BODY) for k, v in body_seq.items()}
    flame = {k: stack(v, SMOOTH_WINDOW_FACE) for k, v in flame_seq.items()}
    cameras = stack(cam_seq, SMOOTH_WINDOW_CAMERA)

    import time

    recon_start = time.perf_counter()
    smplx145 = np.full((num_frames, 145, 3), np.nan, dtype=np.float32)
    with torch.no_grad():
        for t in range(num_frames):
            body_dict = {k: body[k][t:t + 1] for k in BODY_FIELDS}
            body_dict.update({"eye_pose": None, "jaw_pose": None, "joints_offset": None})
            flame_dict = {k: flame[k][t:t + 1] for k in FLAME_FIELDS}
            pred = ehm(body_dict, flame_dict, pose_type="aa")
            joints = pred["joints"][0].detach().cpu().numpy()
            smplx145[t] = joints[:145]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    recon_sec = time.perf_counter() - recon_start
    smplx_names = list(ehm.smplx.keypoint_names[:145])

    arrays = to_common.convert(
        smplx145, smplx_names, presence_threshold=args.presence_threshold, legacy_face_frame=args.legacy_face_frame
    )
    if args.save_raw:
        arrays["raw_smplx145_xyz"] = smplx145
        arrays["raw_openpose67_xyz"] = to_common.smplx145_to_openpose67(smplx145)
        arrays["raw_cameras"] = cameras.detach().cpu().numpy()

    npz_path, meta_path = pb.write_output(
        args.out,
        stem,
        backend=BACKEND,
        upstream_commit=UPSTREAM_COMMIT,
        arrays=arrays,
        fps_source=reader.fps,
        timing=timer,
        extra_meta={
            "video": str(args.video),
            "device": str(device),
            "checkpoint": str(checkpoint),
            "input_size": INPUT_SIZE,
            "smoothing": None if args.no_smooth else {
                "body_window": SMOOTH_WINDOW_BODY, "face_window": SMOOTH_WINDOW_FACE,
                "camera_window": SMOOTH_WINDOW_CAMERA, "polyorder": SMOOTH_POLYORDER,
            },
            "reconstruction_sec": recon_sec,
            "reconstruction_fps": (num_frames / recon_sec) if recon_sec > 0 else None,
            "face_frame": "raw (legacy)" if args.legacy_face_frame else "plot",
            "joint_source": "EHM_v2 SMPL-X keypoints -> OpenPose67 index table + name-based face68",
        },
    )
    t = timer.as_dict()
    print(f"[pear] {stem}: {num_frames} frames, forward {t['forward_fps']:.2f} fps, "
          f"reconstruction {num_frames / recon_sec if recon_sec > 0 else float('nan'):.2f} fps -> {npz_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
