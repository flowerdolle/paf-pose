"""Command line entry point for PAF-Pose."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from . import __version__, fusion, registry as reg, runner, schema, visualize
from .schema import PARTS


# --------------------------------------------------------------------------- run


def cmd_run(args: argparse.Namespace) -> int:
    registry = reg.Registry.load(args.registry)
    try:
        selection = registry.resolve(body=args.body, hand=args.hand, face=args.face, preset=args.preset)
        videos = runner.collect_videos(Path(args.video))
    except (reg.RegistryError, runner.RunnerError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    out_root = Path(args.out)
    weights_root = Path(args.weights) if args.weights else runner.default_weights_root()
    groups = selection.grouped()

    print(f"pafpose {__version__}")
    print(f"selection: body={selection.body} hand={selection.hand} face={selection.face}")
    print(f"videos: {len(videos)}  backends to run per video: {list(groups)}")
    if args.dry_run:
        print("dry run: commands are printed, nothing is executed")

    failures = 0
    for video in videos:
        video_out = out_root / video.stem
        print(f"\n== {video.name}")
        results: dict[str, runner.RunResult] = {}
        for backend_name, parts in groups.items():
            backend = registry.get(backend_name)
            spec = runner.RunSpec(
                backend=backend,
                video=video,
                out_dir=video_out / backend_name,
                weights_root=weights_root,
                parts=tuple(parts),
                use_gpu=not args.cpu,
            )
            try:
                result = runner.run_backend(spec, dry_run=args.dry_run, log_dir=video_out / "logs")
            except runner.RunnerError as exc:
                print(f"  [{backend_name}] error: {exc}", file=sys.stderr)
                failures += 1
                if not args.keep_going:
                    return 1
                continue
            results[backend_name] = result
            if args.dry_run:
                print(f"  [{backend_name}] {shlex.join(result.argv)}")
                continue
            status = "ok" if result.ok else "FAILED"
            print(f"  [{backend_name}] {status} ({result.elapsed_sec:.1f}s) parts={list(parts)}")
            for problem in result.problems:
                print(f"      - {problem}", file=sys.stderr)
            if not result.ok:
                failures += 1
                if not args.keep_going:
                    return 1

        if args.dry_run:
            continue
        video_out.mkdir(parents=True, exist_ok=True)
        (video_out / "selection.json").write_text(
            json.dumps({"video": str(video), "selection": selection.as_dict()}, indent=2), encoding="utf-8"
        )
        if all(r.ok for r in results.values()) and len(results) == len(groups):
            outputs = {name: r.output for name, r in results.items()}
            fused = fusion.fuse_outputs(
                body=outputs[selection.body], hand=outputs[selection.hand], face=outputs[selection.face]
            )
            npz_path, _ = fused.save(video_out, extra_meta={"selection": selection.as_dict(), "video": str(video.resolve())})
            print(f"  fused: {npz_path}  complete frames {int(fused.valid.sum())}/{fused.num_frames}")
            if args.preview:
                preview = visualize.render_preview(video_out, video, video_out / f"preview.{args.preview}")
                print(f"  preview: {preview}")

    return 1 if failures else 0


# --------------------------------------------------------------------------- fuse


def cmd_fuse(args: argparse.Namespace) -> int:
    """Fuse already-produced backend outputs without running containers."""
    try:
        body = schema.load_output(Path(args.body_npz))
        hand = schema.load_output(Path(args.hand_npz)) if args.hand_npz else body
        face = schema.load_output(Path(args.face_npz)) if args.face_npz else body
        for output, part in ((body, "body"), (hand, "hand"), (face, "face")):
            problems = schema.validate_output(output, [part])
            if problems:
                print(f"error: {output.npz_path} is not a valid {part} output:", file=sys.stderr)
                for problem in problems:
                    print(f"  - {problem}", file=sys.stderr)
                return 2
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    config = fusion.FusionConfig(
        hand_scale="none" if args.no_hand_scale else "anchor-median-bone",
        face_scale="none" if args.no_face_scale else "anchor-eye-distance",
    )
    fused = fusion.fuse_outputs(body=body, hand=hand, face=face, config=config)
    npz_path, json_path = fused.save(
        Path(args.out),
        extra_meta={
            "sources": {"body": str(body.npz_path), "hand": str(hand.npz_path), "face": str(face.npz_path)},
        },
    )
    print(f"fused: {npz_path}")
    print(f"frames: {fused.num_frames} complete {int(fused.valid.sum())} "
          f"(body {int(fused.body_valid.sum())}, left {int(fused.left_valid.sum())}, "
          f"right {int(fused.right_valid.sum())}, face {int(fused.face_valid.sum())})")
    print(f"scales: {fused.scales}")
    return 0


# --------------------------------------------------------------------------- visualize


def cmd_visualize(args: argparse.Namespace) -> int:
    result_dir = Path(args.result)
    out_path = Path(args.out) if args.out else result_dir / f"preview.{args.format}"
    options = visualize.RenderOptions(
        elev=args.elev, azim=args.azim, panel_height=args.height, stride=args.stride,
        max_frames=args.max_frames, fps=args.fps, gif_width=args.gif_width,
    )
    try:
        written = visualize.render_preview(result_dir, Path(args.video) if args.video else None, out_path, options)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"preview: {written}")
    return 0


# --------------------------------------------------------------------------- doctor


def _run_quiet(argv: list[str]) -> tuple[bool, str]:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    return proc.returncode == 0, (proc.stdout or proc.stderr).strip().splitlines()[0] if (proc.stdout or proc.stderr).strip() else ""


def cmd_doctor(args: argparse.Namespace) -> int:
    rows: list[tuple[str, bool, str]] = []
    critical_failure = False

    try:
        registry = reg.Registry.load(args.registry)
        rows.append(("registry", True, f"{len(registry)} backends from {registry.source}"))
    except reg.RegistryError as exc:
        rows.append(("registry", False, str(exc)))
        registry = None
        critical_failure = True

    for name in reg.list_presets():
        try:
            preset = reg.load_preset(name)
            if registry is not None:
                registry.resolve(preset=name)
            rows.append((f"preset {name}", True, " ".join(f"{k}={v}" for k, v in preset.items())))
        except reg.RegistryError as exc:
            rows.append((f"preset {name}", False, str(exc)))
            critical_failure = True

    docker = runner.docker_available()
    rows.append(("docker binary", docker, shutil.which("docker") or "not on PATH"))
    daemon_ok = False
    if docker:
        daemon_ok, detail = _run_quiet(["docker", "info", "--format", "{{.ServerVersion}}"])
        rows.append(("docker daemon", daemon_ok, f"server {detail}" if daemon_ok else detail))
    critical_failure = critical_failure or not docker or not daemon_ok

    nvidia_ok, detail = _run_quiet(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    rows.append(("nvidia-smi", nvidia_ok, detail if nvidia_ok else "not available (GPU backends will not run)"))
    if docker and daemon_ok:
        toolkit_ok, detail = _run_quiet(["docker", "info", "--format", "{{json .Runtimes}}"])
        has_nvidia = toolkit_ok and "nvidia" in detail
        rows.append(("nvidia container runtime", has_nvidia, "found" if has_nvidia else "not registered with docker"))

    weights_root = Path(args.weights) if args.weights else runner.default_weights_root()
    rows.append(("weights root", weights_root.is_dir(), str(weights_root)))

    if registry is not None:
        for backend in registry:
            if docker and daemon_ok:
                img_ok, _ = _run_quiet(["docker", "image", "inspect", backend.image])
                rows.append((f"image {backend.name}", img_ok, backend.image if img_ok else f"{backend.image} not built"))
            if backend.needs_weights:
                missing = runner.missing_weights(backend, weights_root)
                rows.append(
                    (
                        f"weights {backend.name}",
                        not missing,
                        "complete" if not missing else f"missing {missing} under {runner.backend_weights_dir(backend, weights_root)}",
                    )
                )

    width = max(len(r[0]) for r in rows)
    for label, ok, detail in rows:
        print(f"{'OK     ' if ok else 'MISSING'} {label.ljust(width)}  {detail}")
    print()
    if critical_failure:
        print("doctor: pafpose run cannot work on this machine yet (see MISSING rows above)")
        return 1
    print("doctor: host is ready; build missing images with `docker compose build` and place weights as listed")
    return 0


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pafpose", description="Part-Aware Fusion Whole-Body 3D Pose Estimator")
    parser.add_argument("--version", action="version", version=f"pafpose {__version__}")
    parser.add_argument("--registry", default=None, help="backends.yaml path (default: bundled)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run selected backends on a video or folder and fuse the parts")
    p_run.add_argument("--video", required=True, help="one .mp4 file or a folder of .mp4 files")
    p_run.add_argument("--out", required=True, help="output root; one sub-folder per video")
    p_run.add_argument("--preset", default=None, help=f"one of {reg.list_presets()} or a yaml path")
    for part in PARTS:
        p_run.add_argument(f"--{part}", default=None, help=f"backend for {part} (overrides preset)")
    p_run.add_argument("--weights", default=None, help="weights root (default: $PAFPOSE_WEIGHTS or ./weights)")
    p_run.add_argument("--cpu", action="store_true", help="do not pass --gpus all to docker")
    p_run.add_argument("--dry-run", action="store_true", help="print docker commands without running")
    p_run.add_argument("--keep-going", action="store_true", help="continue with other videos after a failure")
    p_run.add_argument("--preview", choices=("mp4", "gif"), default=None, help="also render a side-by-side preview per video")
    p_run.set_defaults(func=cmd_run)

    p_fuse = sub.add_parser("fuse", help="fuse existing per-backend outputs without running containers")
    p_fuse.add_argument("--body-npz", required=True, help="common-schema npz providing body8_eye2_xyz")
    p_fuse.add_argument("--hand-npz", default=None, help="npz providing hands42_xyz (default: same as body)")
    p_fuse.add_argument("--face-npz", default=None, help="npz providing face70_xyz (default: same as body)")
    p_fuse.add_argument("--out", required=True, help="directory for fused.npz and fusion.json")
    p_fuse.add_argument("--no-hand-scale", action="store_true", help="keep each hand's own scale")
    p_fuse.add_argument("--no-face-scale", action="store_true", help="keep the face's own scale")
    p_fuse.set_defaults(func=cmd_fuse)

    p_vis = sub.add_parser("visualize", help="render a side-by-side preview: 3D skeleton (left) and source video (right)")
    p_vis.add_argument("--result", required=True, help="result folder of one video (contains fused.npz)")
    p_vis.add_argument("--video", default=None, help="source video (default: path recorded in fusion.json)")
    p_vis.add_argument("--out", default=None, help="output file; default <result>/preview.<format>")
    p_vis.add_argument("--format", choices=("mp4", "gif"), default="mp4")
    p_vis.add_argument("--height", type=int, default=480, help="panel height in pixels")
    p_vis.add_argument("--stride", type=int, default=1, help="render every n-th frame")
    p_vis.add_argument("--max-frames", type=int, default=None)
    p_vis.add_argument("--fps", type=float, default=None, help="output fps (default: source fps / stride)")
    p_vis.add_argument("--gif-width", type=int, default=800, help="total gif width in pixels")
    p_vis.add_argument("--elev", type=float, default=10.0, help="3D view elevation")
    p_vis.add_argument("--azim", type=float, default=-90.0, help="3D view azimuth (-90 = frontal)")
    p_vis.set_defaults(func=cmd_visualize)

    p_doc = sub.add_parser("doctor", help="check docker, GPU, images, weights, registry, presets")
    p_doc.add_argument("--weights", default=None)
    p_doc.set_defaults(func=cmd_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    # OpenCV's ffmpeg backend logs swscaler warnings (interlaced flags, colour space) for many
    # camera files; they do not affect decoding. Keep only fatal ffmpeg messages unless the user
    # set the variable themselves.
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
