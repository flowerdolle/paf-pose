import numpy as np

from pafpose import fusion


def _hand(scale=1.0, offset=(0.0, 0.0, 0.0), T=3):
    base = np.zeros((21, 3), dtype=np.float32)
    for j in range(1, 21):
        base[j] = [0.01 * j, 0.0, 0.002 * j]
    hand = np.repeat(base[None], T, axis=0) * scale + np.asarray(offset, dtype=np.float32)
    return hand


def test_attach_hand_translates_and_scales_to_anchor():
    anchor = _hand(scale=2.0, offset=(1.0, 2.0, 3.0))
    pred = _hand(scale=1.0, offset=(-5.0, 0.0, 9.0))
    valid = np.ones(3, dtype=bool)
    out, scale = fusion.attach_hand(pred, anchor, valid)
    assert np.isclose(scale, 2.0)
    assert np.allclose(out, anchor, atol=1e-6)


def test_attach_hand_wrist_only_anchor_keeps_scale():
    body8 = np.zeros((3, 8, 3), dtype=np.float32)
    body8[:, fusion.LEFT_WRIST] = [0.5, 0.0, 1.0]
    left_anchor, _ = fusion.wrist_only_anchor(body8)
    pred = _hand(scale=3.0, offset=(7.0, 7.0, 7.0))
    out, scale = fusion.attach_hand(pred, left_anchor, np.ones(3, dtype=bool))
    assert scale == 1.0
    assert np.allclose(out[:, 0], [0.5, 0.0, 1.0])
    assert np.allclose(out - out[:, 0:1], pred - pred[:, 0:1])


def test_attach_hand_invalid_frames_become_nan():
    anchor = _hand()
    pred = _hand()
    valid = np.asarray([True, False, True])
    out, _ = fusion.attach_hand(pred, anchor, valid)
    assert np.isnan(out[1]).all() and np.isfinite(out[0]).all()


def test_attach_face_scales_by_eye_distance_and_centers_on_eyes():
    T = 2
    face = np.random.rand(T, 70, 3).astype(np.float32)
    face[:, 68] = [0.0, 0.0, 0.0]
    face[:, 69] = [0.1, 0.0, 0.0]
    anchor_eye2 = np.zeros((T, 2, 3), dtype=np.float32)
    anchor_eye2[:, 0] = [1.0, 1.0, 1.0]
    anchor_eye2[:, 1] = [1.3, 1.0, 1.0]
    out, scale = fusion.attach_face(face, anchor_eye2, np.ones(T, dtype=bool))
    assert np.isclose(scale, 3.0)
    assert np.allclose(out[:, [68, 69]].mean(axis=1), anchor_eye2.mean(axis=1), atol=1e-5)
    assert np.allclose(np.linalg.norm(out[:, 68] - out[:, 69], axis=-1), 0.3, atol=1e-5)


def test_fuse_arrays_shapes_masks_and_save(tmp_path):
    T = 4
    body = np.random.rand(T, 10, 3).astype(np.float32)
    hands = np.concatenate([_hand(T=T), _hand(T=T, offset=(1, 0, 0))], axis=1)
    face = np.random.rand(T, 70, 3).astype(np.float32)
    body_valid = np.ones(T, dtype=bool)
    left_valid = np.asarray([True, True, False, True])
    right_valid = np.ones(T, dtype=bool)
    face_valid = np.asarray([True, False, True, True])
    result = fusion.fuse_arrays(body, body_valid, hands, left_valid, right_valid, face, face_valid)
    assert result.wholebody120.shape == (T, 120, 3)
    assert result.valid.tolist() == [True, False, False, True]
    assert np.isnan(result.hands42[2, :21]).all() and np.isfinite(result.hands42[2, 21:]).all()
    npz, js = result.save(tmp_path)
    data = np.load(npz)
    assert data["wholebody120_xyz"].shape == (T, 120, 3)
    assert js.is_file()


def test_fuse_arrays_truncates_to_shortest_source():
    body = np.random.rand(6, 10, 3).astype(np.float32)
    hands = np.random.rand(4, 42, 3).astype(np.float32)
    face = np.random.rand(5, 70, 3).astype(np.float32)
    ones = lambda n: np.ones(n, dtype=bool)  # noqa: E731
    result = fusion.fuse_arrays(body, ones(6), hands, ones(4), ones(4), face, ones(5))
    assert result.num_frames == 4
