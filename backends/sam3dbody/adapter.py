#!/usr/bin/env python3
"""PAF-Pose backend adapter: SAM 3D Body (via the SAM-Body4D implementation, frame-wise).

Runs SAM 3D Body on every frame of one video and writes the common PAF-Pose
output (body8 + eye2, hands42). Ported from
SL_MST/papers/iccas2026/models/body/sam-body4d/run_sam_body4d_body.py with the
NIA-specific batching removed.

    python /app/adapter.py --video /input/clip.mp4 --out /output --weights /weights

Expected weights layout (mounted at --weights):

    sam-3d-body-dinov3/model.ckpt
    sam-3d-body-dinov3/model_config.yaml
    sam-3d-body-dinov3/assets/mhr_model.pt
    moge-2-vitl-normal/model.pt
    vitdet/model_final_f05665.pkl          (optional, only for --bbox-mode detector)
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
try:
    import pafpose_backend as pb
except ImportError:  # local testing outside the image
    sys.path.insert(0, str(_HERE.parents[1] / "_common"))
    import pafpose_backend as pb

sys.path.insert(0, str(_HERE))
import to_common  # noqa: E402

BACKEND = "sam3dbody"
UPSTREAM_COMMIT = "21af102"
DEFAULT_ROOT = os.environ.get("SAM_BODY4D_ROOT", "/opt/sam-body4d")

WEIGHT_FILES = {
    "ckpt": "sam-3d-body-dinov3/model.ckpt",
    "config": "sam-3d-body-dinov3/model_config.yaml",
    "mhr": "sam-3d-body-dinov3/assets/mhr_model.pt",
    "fov": "moge-2-vitl-normal/model.pt",
}
DETECTOR_FILE = "vitdet/model_final_f05665.pkl"


def parse_args():
    parser = pb.base_parser("SAM 3D Body (SAM-Body4D, frame-wise) -> PAF-Pose common output")
    parser.add_argument("--sam-body4d-root", type=Path, default=Path(DEFAULT_ROOT),
                        help="gaomingqi/sam-body4d checkout (env SAM_BODY4D_ROOT)")
    parser.add_argument("--bbox-mode", choices=("full", "detector"), default="full",
                        help="full: whole frame is the person box (paper setting). detector: ViTDet person detector")
    parser.add_argument("--person-strategy", choices=("largest", "first"), default="largest")
    parser.add_argument("--bbox-thr", type=float, default=0.6)
    parser.add_argument("--nms-thr", type=float, default=0.3)
    parser.add_argument("--inference-type", choices=("full", "body"), default="full",
                        help="full = body + hand decoders (paper setting)")
    parser.add_argument("--no-fov", action="store_true", help="skip the MoGe FoV estimator (default FoV)")
    parser.add_argument("--verbose-frames", action="store_true", help="do not silence per-frame model logs")
    return parser.parse_args()


def add_sam_body4d_paths(root: Path) -> None:
    root = root.expanduser().resolve()
    if not (root / "models" / "sam_3d_body").is_dir():
        raise FileNotFoundError(f"SAM-Body4D checkout not found at {root} (expected models/sam_3d_body)")
    for path in (root / "models" / "diffusion_vas", root / "models" / "sam_3d_body", root):
        sys.path.insert(0, str(path))  # repo root ends up first: its `utils` package must win


def resolve_weights(weights: Path, need_detector: bool) -> dict[str, Path]:
    paths = {name: weights / rel for name, rel in WEIGHT_FILES.items()}
    if need_detector:
        paths["detector"] = weights / DETECTOR_FILE
    missing = [str(p) for name, p in paths.items() if name != "detector" and not p.is_file()]
    if missing:
        raise FileNotFoundError("missing weights:\n  " + "\n  ".join(missing) + "\nsee backends/sam3dbody/README.md")
    return paths


def build_estimator(args, paths: dict[str, Path]):
    import torch
    from models.sam_3d_body.sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type != "cuda":
        print("WARNING: SAM-Body4D's process_one_image expects CUDA; CPU runs are unsupported upstream.", file=sys.stderr)

    model, model_cfg = load_sam_3d_body(str(paths["ckpt"]), device=device, mhr_path=str(paths["mhr"]))

    fov_estimator = None
    if not args.no_fov:
        from models.sam_3d_body.tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(name="moge2", device=device, path=str(paths["fov"]))

    human_detector = None
    if args.bbox_mode == "detector":
        from models.sam_3d_body.tools.build_detector import HumanDetector

        detector_path = str(paths["detector"]) if paths["detector"].is_file() else ""
        human_detector = HumanDetector(name="vitdet", device=device, path=detector_path)

    return SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=None,
        fov_estimator=fov_estimator,
    )


def choose_person(outputs, strategy: str):
    if not outputs:
        return None
    if strategy == "first":
        return outputs[0]

    def area(row) -> float:
        bbox = np.asarray(row.get("bbox", []), dtype=np.float32).reshape(-1)
        if bbox.size < 4:
            return -1.0
        return float(max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]))

    return max(outputs, key=area)


def assemble_arrays(mhr70_xyz: np.ndarray, frame_valid: np.ndarray, bbox_xyxy: np.ndarray, focal: np.ndarray) -> dict:
    """Stacked per-frame model outputs -> arrays to write. Separate so it can be tested offline."""
    arrays = to_common.mhr70_to_common(mhr70_xyz, frame_valid)
    arrays["raw_mhr70_camera_xyz"] = np.asarray(mhr70_xyz, dtype=np.float32)
    arrays["raw_frame_valid"] = np.asarray(frame_valid, dtype=bool)
    arrays["raw_bbox_xyxy"] = np.asarray(bbox_xyxy, dtype=np.float32)
    arrays["raw_focal_length"] = np.asarray(focal, dtype=np.float32)
    return arrays


def main() -> int:
    args = parse_args()
    add_sam_body4d_paths(args.sam_body4d_root)
    paths = resolve_weights(args.weights, need_detector=args.bbox_mode == "detector")

    import cv2

    timer = pb.Timer()
    timer.start_load()
    estimator = build_estimator(args, paths)
    timer.end_load()

    reader = pb.VideoReader(args.video, max_frames=args.max_frames)
    mhr70_list, valid_list, bbox_list, focal_list = [], [], [], []
    for index, frame_bgr in reader:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        bboxes = None
        if args.bbox_mode == "full":
            h, w = frame_rgb.shape[:2]
            bboxes = np.asarray([[0, 0, w, h]], dtype=np.float32)

        timer.start_frame()
        sink = io.StringIO() if not args.verbose_frames else None
        with (contextlib.redirect_stdout(sink) if sink is not None else contextlib.nullcontext()):
            outputs = estimator.process_one_image(
                frame_rgb,
                bboxes=bboxes,
                bbox_thr=args.bbox_thr,
                nms_thr=args.nms_thr,
                use_mask=False,
                inference_type=args.inference_type,
            )
        timer.end_frame()

        person = choose_person(outputs, args.person_strategy)
        if person is None or "pred_keypoints_3d" not in person:
            mhr70_list.append(np.full((70, 3), np.nan, dtype=np.float32))
            valid_list.append(False)
            bbox_list.append(np.full((4,), np.nan, dtype=np.float32))
            focal_list.append(np.nan)
            continue
        mhr = np.asarray(person["pred_keypoints_3d"], dtype=np.float32)[:70]
        mhr70_list.append(mhr)
        valid_list.append(bool(np.isfinite(to_common.openpose53_to_nia_body9(to_common.mhr70_to_openpose53(mhr))).all()))
        bbox = np.asarray(person.get("bbox", np.full((4,), np.nan)), dtype=np.float32).reshape(-1)[:4]
        bbox_list.append(bbox if bbox.size == 4 else np.full((4,), np.nan, dtype=np.float32))
        focal_arr = np.asarray(person.get("focal_length", np.nan), dtype=np.float32).reshape(-1)
        focal_list.append(float(focal_arr[0]) if focal_arr.size else float("nan"))

    if not mhr70_list:
        raise RuntimeError(f"no frames decoded from {args.video}")

    arrays = assemble_arrays(
        np.stack(mhr70_list), np.asarray(valid_list, dtype=bool), np.stack(bbox_list), np.asarray(focal_list)
    )
    npz_path, meta_path = pb.write_output(
        args.out,
        args.video.stem,
        backend=BACKEND,
        upstream_commit=UPSTREAM_COMMIT,
        arrays=arrays,
        fps_source=reader.fps,
        timing=timer,
        extra_meta={
            "model": {
                "name": "SAM 3D Body (via the SAM-Body4D implementation, frame-wise)",
                "sam_body4d_root": str(args.sam_body4d_root),
                "bbox_mode": args.bbox_mode,
                "person_strategy": args.person_strategy,
                "bbox_thr": args.bbox_thr,
                "nms_thr": args.nms_thr,
                "inference_type": args.inference_type,
                "fov_estimator": None if args.no_fov else "moge2",
                "weights": {k: str(v) for k, v in paths.items()},
            },
            "video": {"path": str(args.video), "width": reader.width, "height": reader.height},
        },
    )
    valid = int(arrays["body_valid"].sum())
    print(f"[{BACKEND}] wrote {npz_path} ({len(mhr70_list)} frames, {valid} body-valid) and {meta_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
