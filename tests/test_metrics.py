import numpy as np

from pafpose import fusion, metrics


def _rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def test_pa_mpjpe_is_invariant_to_similarity_transform():
    gt = np.random.rand(5, 20, 3)
    pred = 2.5 * (gt @ _rotation(0.7).T) + np.asarray([1.0, -2.0, 3.0])
    assert np.allclose(metrics.pa_mpjpe(gt, pred), 0.0, atol=1e-5)
    assert metrics.mpjpe(gt, pred).min() > 0.1


def test_summarize_scales_units_and_ignores_nan():
    errors = np.asarray([0.01, 0.02, np.nan, 0.03])
    s = metrics.summarize(errors, unit_scale=1000.0)
    assert s["n"] == 3 and np.isclose(s["mean"], 20.0) and np.isclose(s["median"], 20.0)
    assert np.isnan(metrics.summarize(np.asarray([]))["mean"])


def test_evaluate_fusion_perfect_prediction_has_zero_error():
    T = 3
    gt121 = np.random.rand(T, 121, 3).astype(np.float32)
    gt = metrics.GroundTruth.from_nia121(gt121)
    fused = fusion.FusionResult(
        body8=gt.body8.copy(),
        eye2=gt.eye2.copy(),
        hands42=gt.hands42.copy(),
        face70=gt.face70.copy(),
        body_valid=np.ones(T, dtype=bool),
        left_valid=np.ones(T, dtype=bool),
        right_valid=np.ones(T, dtype=bool),
        face_valid=np.ones(T, dtype=bool),
    )
    errors = metrics.evaluate_fusion(gt, fused)
    for name in metrics.ERROR_NAMES:
        assert errors[name].shape == (T,)
        assert np.allclose(errors[name], 0.0, atol=1e-5)
    agg = metrics.aggregate([errors, errors])
    assert agg["wb_pa_mpjpe"]["n"] == 2 * T


def test_evaluate_fusion_respects_complete_case_mask():
    T = 4
    gt = metrics.GroundTruth.from_nia121(np.random.rand(T, 121, 3).astype(np.float32))
    fused = fusion.FusionResult(
        body8=gt.body8.copy(), eye2=gt.eye2.copy(), hands42=gt.hands42.copy(), face70=gt.face70.copy(),
        body_valid=np.ones(T, dtype=bool), left_valid=np.asarray([True, False, True, True]),
        right_valid=np.ones(T, dtype=bool), face_valid=np.asarray([True, True, True, False]),
    )
    errors = metrics.evaluate_fusion(gt, fused)
    assert errors["wb_pa_mpjpe"].shape == (2,)
