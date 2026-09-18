"""Assemble and execute ``docker run`` for one backend on one video, then read its output."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import schema
from .registry import Backend

VIDEO_SUFFIXES = (".mp4",)

CONTAINER_INPUT = "/input"
CONTAINER_OUTPUT = "/output"
CONTAINER_WEIGHTS = "/weights"


class RunnerError(RuntimeError):
    pass


def default_weights_root() -> Path:
    return Path(os.environ.get("PAFPOSE_WEIGHTS") or Path.cwd() / "weights")


def collect_videos(source: Path) -> list[Path]:
    """One mp4 file, or every mp4 directly inside a folder (sorted)."""
    source = Path(source)
    if source.is_file():
        if source.suffix.lower() not in VIDEO_SUFFIXES:
            raise RunnerError(f"{source} is not an .mp4 file")
        return [source]
    if source.is_dir():
        videos = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)
        if not videos:
            raise RunnerError(f"no .mp4 files found in {source}")
        return videos
    raise RunnerError(f"input not found: {source}")


def docker_available() -> bool:
    return shutil.which("docker") is not None


def backend_weights_dir(backend: Backend, weights_root: Path) -> Path:
    return Path(weights_root) / backend.name


def missing_weights(backend: Backend, weights_root: Path) -> list[str]:
    """Declared weight entries that do not exist under the backend's weights dir."""
    base = backend_weights_dir(backend, weights_root)
    return [entry for entry in backend.weights if not (base / entry.rstrip("/")).exists()]


@dataclass(frozen=True)
class RunSpec:
    backend: Backend
    video: Path
    out_dir: Path
    weights_root: Path
    parts: tuple[str, ...]
    use_gpu: bool = True
    run_as_current_user: bool = True
    extra_docker_args: tuple[str, ...] = ()

    @property
    def stem(self) -> str:
        return self.video.stem


def build_docker_command(spec: RunSpec) -> list[str]:
    """Return argv for docker run. Paths are made absolute; nothing is executed."""
    video = Path(spec.video).resolve()
    out_dir = Path(spec.out_dir).resolve()
    argv: list[str] = ["docker", "run", "--rm"]
    if spec.backend.gpu and spec.use_gpu:
        argv += ["--gpus", "all"]
    if spec.run_as_current_user and hasattr(os, "getuid"):
        argv += ["--user", f"{os.getuid()}:{os.getgid()}"]
    container_video = f"{CONTAINER_INPUT}/{video.name}"
    argv += ["-v", f"{video}:{container_video}:ro"]
    argv += ["-v", f"{out_dir}:{CONTAINER_OUTPUT}"]
    substitutions = {"video": container_video, "out": CONTAINER_OUTPUT}
    if spec.backend.needs_weights:
        weights_dir = backend_weights_dir(spec.backend, spec.weights_root).resolve()
        argv += ["-v", f"{weights_dir}:{CONTAINER_WEIGHTS}:ro"]
        substitutions["weights"] = CONTAINER_WEIGHTS
    argv += list(spec.extra_docker_args)
    argv.append(spec.backend.image)
    argv += shlex.split(spec.backend.command.format(**substitutions))
    return argv


@dataclass
class RunResult:
    spec: RunSpec
    argv: list[str]
    returncode: int | None
    elapsed_sec: float
    output: schema.BackendOutput | None = None
    problems: list[str] = field(default_factory=list)
    log_path: Path | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.output is not None and not self.problems


def run_backend(spec: RunSpec, dry_run: bool = False, log_dir: Path | None = None) -> RunResult:
    argv = build_docker_command(spec)
    if dry_run:
        return RunResult(spec=spec, argv=argv, returncode=None, elapsed_sec=0.0)

    if not docker_available():
        raise RunnerError("docker is not on PATH; install Docker and NVIDIA Container Toolkit, or use --dry-run")
    if spec.backend.needs_weights:
        missing = missing_weights(spec.backend, spec.weights_root)
        if missing:
            raise RunnerError(
                f"backend '{spec.backend.name}' is missing weights {missing} under "
                f"{backend_weights_dir(spec.backend, spec.weights_root)}"
            )

    out_dir = Path(spec.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{spec.stem}.{spec.backend.name}.log"

    started = time.perf_counter()
    proc = subprocess.run(argv, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    if log_path is not None:
        log_path.write_text(
            f"$ {shlex.join(argv)}\n\n[stdout]\n{proc.stdout}\n[stderr]\n{proc.stderr}\n", encoding="utf-8"
        )

    result = RunResult(spec=spec, argv=argv, returncode=proc.returncode, elapsed_sec=elapsed, log_path=log_path)
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-5:]
        result.problems.append(f"container exited with {proc.returncode}: " + " | ".join(tail))
        return result

    npz_path, meta_path = schema.output_paths(out_dir, spec.stem)
    if not npz_path.is_file() or not meta_path.is_file():
        result.problems.append(f"adapter did not write {npz_path.name} and {meta_path.name} in {out_dir}")
        return result
    output = schema.load_output(npz_path, meta_path)
    result.output = output
    result.problems.extend(schema.validate_output(output, spec.parts))
    return result
