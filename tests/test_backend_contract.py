"""The container-side helper (backends/_common) and the host schema must agree."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from pafpose import fusion, schema

HELPER = Path(__file__).resolve().parents[1] / "backends" / "_common" / "pafpose_backend.py"


@pytest.fixture(scope="module")
def pb():
    spec = importlib.util.spec_from_file_location("pafpose_backend", HELPER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pafpose_backend"] = module
    spec.loader.exec_module(module)
    return module


def test_constants_match_host_schema(pb):
    assert tuple(pb.BODY8_EYE2_NAMES) == schema.BODY8_EYE2_NAMES
    assert tuple(pb.HAND21_NAMES) == schema.HAND21_NAMES
    assert pb.PART_KEY == schema.PART_TO_KEY
    assert pb.VALID_KEY == schema.VALID_KEY
    assert pb.JOINTS == schema.JOINT_COUNT
    pts = np.random.rand(3, 4, 3).astype(np.float32)
    assert np.array_equal(pb.camera_to_plot(pts), schema.camera_to_plot(pts))


def test_write_output_passes_host_validation_and_fuses(pb, tmp_path):
    T = 5
    timer = pb.Timer()
    for _ in range(T):
        timer.start_frame(); timer.end_frame()
    arrays = {
        "body8_eye2_xyz": np.random.rand(T, 10, 3),
        "body_valid": np.ones(T, bool),
        "hands42_xyz": np.random.rand(T, 42, 3),
        "left_hand_valid": np.array([1, 1, 0, 1, 1], bool),
        "right_hand_valid": np.ones(T, bool),
        "face70_xyz": np.random.rand(T, 70, 3),
        "face_valid": np.ones(T, bool),
        "raw_anything": np.zeros(3),
    }
    arrays["face70_xyz"][4] = np.nan  # marked valid by the adapter but non-finite: helper must fix the mask
    npz, meta = pb.write_output(tmp_path, "clip", backend="x", upstream_commit="abc", arrays=arrays, fps_source=30.0, timing=timer)
    out = schema.load_output(npz, meta)
    assert schema.validate_output(out, ["body", "hand", "face"]) == []
    assert out.meta["parts"] == ["body", "hand", "face"]
    assert out.meta["timing"]["forward_fps"] > 0
    assert out.valid("hand").tolist() == [True, True, False, True, True]
    assert out.valid("face").tolist() == [True, True, True, True, False]
    fused = fusion.fuse_outputs(out, out, out)
    assert fused.valid.tolist() == [True, True, False, True, False]


def test_write_output_rejects_bad_shapes(pb, tmp_path):
    with pytest.raises(ValueError, match="expected \\(T, 42, 3\\)"):
        pb.write_output(tmp_path, "c", backend="x", upstream_commit="a",
                        arrays={"hands42_xyz": np.zeros((2, 21, 3)), "hands_valid": np.ones(2, bool)},
                        fps_source=30, timing={"total_sec": 0, "per_frame_sec": []})
    with pytest.raises(ValueError, match="no part arrays"):
        pb.write_output(tmp_path, "c", backend="x", upstream_commit="a", arrays={"raw_x": np.zeros(1)},
                        fps_source=30, timing={"total_sec": 0, "per_frame_sec": []})
