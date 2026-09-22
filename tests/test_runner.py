import json
from pathlib import Path

import numpy as np
import pytest

from pafpose import registry as reg, runner


@pytest.fixture
def registry():
    return reg.Registry.load()


def _touch_video(tmp_path: Path, name: str = "clip.mp4") -> Path:
    video = tmp_path / name
    video.write_bytes(b"\x00")
    return video


def test_collect_videos_file_and_folder(tmp_path: Path):
    a = _touch_video(tmp_path, "b.mp4")
    b = _touch_video(tmp_path, "a.mp4")
    (tmp_path / "notes.txt").write_text("x")
    assert runner.collect_videos(a) == [a]
    assert runner.collect_videos(tmp_path) == [b, a]
    with pytest.raises(runner.RunnerError):
        runner.collect_videos(tmp_path / "notes.txt")
    with pytest.raises(runner.RunnerError):
        runner.collect_videos(tmp_path / "missing")


def test_docker_command_gpu_backend(registry, tmp_path: Path):
    video = _touch_video(tmp_path)
    spec = runner.RunSpec(
        backend=registry.get("pear"),
        video=video,
        out_dir=tmp_path / "out" / "pear",
        weights_root=tmp_path / "weights",
        parts=("body", "hand", "face"),
    )
    argv = runner.build_docker_command(spec)
    assert argv[:3] == ["docker", "run", "--rm"]
    assert "--gpus" in argv and argv[argv.index("--gpus") + 1] == "all"
    assert f"{video.resolve()}:/input/clip.mp4:ro" in argv
    assert f"{(tmp_path / 'out' / 'pear').resolve()}:/output" in argv
    assert f"{(tmp_path / 'weights' / 'pear').resolve()}:/weights:ro" in argv
    assert argv.index("pafpose/pear:0.1") < argv.index("python")
    assert argv[-6:] == ["--video", "/input/clip.mp4", "--out", "/output", "--weights", "/weights"]


def test_docker_command_cpu_backend_without_weights(registry, tmp_path: Path):
    video = _touch_video(tmp_path)
    spec = runner.RunSpec(
        backend=registry.get("mediapipe"),
        video=video,
        out_dir=tmp_path / "out",
        weights_root=tmp_path / "weights",
        parts=("body",),
    )
    argv = runner.build_docker_command(spec)
    assert "--gpus" not in argv
    assert not any(":/weights" in a for a in argv)
    assert "{weights}" not in " ".join(argv)


def test_dry_run_does_not_touch_disk(registry, tmp_path: Path):
    video = _touch_video(tmp_path)
    spec = runner.RunSpec(
        backend=registry.get("teaser"), video=video, out_dir=tmp_path / "out", weights_root=tmp_path, parts=("face",)
    )
    result = runner.run_backend(spec, dry_run=True)
    assert result.returncode is None and result.argv[0] == "docker"
    assert not (tmp_path / "out").exists()


def test_missing_weights_detected(registry, tmp_path: Path):
    backend = registry.get("teaser")
    assert runner.missing_weights(backend, tmp_path) == list(backend.weights)
    base = tmp_path / "teaser"
    for entry in backend.weights:
        target = base / entry
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")
    assert runner.missing_weights(backend, tmp_path) == []


def test_run_backend_reads_and_validates_output(registry, tmp_path: Path, monkeypatch):
    """Simulate a container run by faking subprocess.run and writing a valid npz."""
    backend = registry.get("mediapipe")
    video = _touch_video(tmp_path)
    out_dir = tmp_path / "out"

    def fake_run(argv, capture_output, text):
        out_dir.mkdir(parents=True, exist_ok=True)
        T = 4
        arrays = {
            "body8_eye2_xyz": np.zeros((T, 10, 3), dtype=np.float32),
            "body_valid": np.ones(T, dtype=bool),
        }
        np.savez_compressed(out_dir / "clip.npz", **arrays)
        (out_dir / "clip.meta.json").write_text(
            json.dumps(
                {
                    "backend": "mediapipe",
                    "upstream_commit": "n/a",
                    "parts": ["body"],
                    "num_frames": T,
                    "fps_source": 30.0,
                    "timing": {"total_sec": 0.1, "per_frame_sec": [0.025] * T},
                }
            )
        )

        class P:
            returncode = 0
            stdout = "done"
            stderr = ""

        return P()

    monkeypatch.setattr(runner, "docker_available", lambda: True)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    spec = runner.RunSpec(backend=backend, video=video, out_dir=out_dir, weights_root=tmp_path, parts=("body",))
    result = runner.run_backend(spec, log_dir=tmp_path / "logs")
    assert result.ok, result.problems
    assert result.output is not None and result.output.num_frames == 4
    assert result.log_path is not None and result.log_path.is_file()
