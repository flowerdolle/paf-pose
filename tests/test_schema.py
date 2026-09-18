import numpy as np

from pafpose import schema


def _good(T: int = 5):
    arrays = {
        "body8_eye2_xyz": np.random.rand(T, 10, 3).astype(np.float32),
        "body_valid": np.ones(T, dtype=bool),
        "hands42_xyz": np.random.rand(T, 42, 3).astype(np.float32),
        "hands_valid": np.ones(T, dtype=bool),
        "face70_xyz": np.random.rand(T, 70, 3).astype(np.float32),
        "face_valid": np.ones(T, dtype=bool),
    }
    meta = {
        "backend": "pear",
        "upstream_commit": "e1aa1f7",
        "parts": ["body", "hand", "face"],
        "num_frames": T,
        "fps_source": 30.0,
        "timing": {"total_sec": 1.0, "per_frame_sec": [0.2] * T},
    }
    return arrays, meta


def test_valid_output_has_no_problems():
    arrays, meta = _good()
    assert schema.validate(arrays, meta, ["body", "hand", "face"]) == []


def test_only_expected_parts_are_checked():
    arrays, meta = _good()
    del arrays["face70_xyz"], arrays["face_valid"]
    assert schema.validate(arrays, meta, ["body", "hand"]) == []
    assert any("face70_xyz" in p for p in schema.validate(arrays, meta, ["face"]))


def test_shape_and_mask_problems():
    arrays, meta = _good()
    arrays["hands42_xyz"] = np.zeros((5, 21, 3), dtype=np.float32)
    problems = schema.validate(arrays, meta, ["hand"])
    assert any("expected (T, 42, 3)" in p for p in problems)

    arrays, meta = _good()
    arrays["body_valid"] = np.zeros(5, dtype=bool)
    assert any("no valid frame" in p for p in schema.validate(arrays, meta, ["body"]))

    arrays, meta = _good()
    arrays["face70_xyz"][2] = np.nan
    assert any("non-finite" in p for p in schema.validate(arrays, meta, ["face"]))


def test_nan_frames_are_fine_when_marked_invalid():
    arrays, meta = _good()
    arrays["face70_xyz"][2] = np.nan
    arrays["face_valid"][2] = False
    assert schema.validate(arrays, meta, ["face"]) == []


def test_frame_count_and_meta_mismatch():
    arrays, meta = _good()
    arrays["face70_xyz"] = arrays["face70_xyz"][:3]
    arrays["face_valid"] = arrays["face_valid"][:3]
    problems = schema.validate(arrays, meta, ["body", "face"])
    assert any("frame counts differ" in p for p in problems)

    arrays, meta = _good()
    meta["num_frames"] = 99
    assert any("num_frames=99" in p for p in schema.validate(arrays, meta, ["body"]))

    arrays, meta = _good()
    del meta["timing"]
    assert any("missing field 'timing'" in p for p in schema.validate(arrays, meta, ["body"]))


def test_roundtrip_through_disk(tmp_path):
    arrays, meta = _good()
    npz, mj = schema.output_paths(tmp_path, "clip")
    np.savez_compressed(npz, **arrays)
    mj.write_text(__import__("json").dumps(meta))
    out = schema.load_output(npz)
    assert out.meta_path == mj
    assert out.part("hand").shape == (5, 42, 3)
    assert out.valid("hand").dtype == bool
    assert schema.validate_output(out, ["body", "hand", "face"]) == []
