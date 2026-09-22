import json

import numpy as np
import pytest

from pafpose import fusion, visualize


def _synthetic_result(tmp_path, T=6):
    body = np.random.rand(T, 10, 3).astype(np.float32)
    hands = np.random.rand(T, 42, 3).astype(np.float32)
    face = np.random.rand(T, 70, 3).astype(np.float32)
    ones = np.ones(T, dtype=bool)
    left_valid = ones.copy(); left_valid[2] = False
    fused = fusion.fuse_arrays(body, ones, hands, left_valid, ones, face, ones)
    result = tmp_path / "clip"
    fused.save(result, extra_meta={"video": str(tmp_path / "clip.mp4"), "fps_source": 10.0, "selection": {"body": "a", "hand": "b", "face": "c"}})
    return result


def test_face_edges_are_valid_indices():
    assert visualize.FACE_EDGES.min() >= 0 and visualize.FACE_EDGES.max() < 68


def test_render_preview_mp4_and_gif(tmp_path):
    result = _synthetic_result(tmp_path)
    cv2 = pytest.importorskip("cv2")
    out = visualize.render_preview(result, result / "preview.mp4", visualize.RenderOptions(panel_height=96))
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 6
    assert abs(cap.get(cv2.CAP_PROP_FPS) - 10.0) < 0.1
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape[:2] == (96, 96)

    gif = visualize.render_preview(result, result / "preview.gif", visualize.RenderOptions(panel_height=96, stride=2, gif_width=200))
    from PIL import Image

    with Image.open(gif) as im:
        assert im.n_frames == 3 and im.size[0] <= 200


def test_render_preview_without_video_entry(tmp_path):
    result = _synthetic_result(tmp_path)
    meta = json.loads((result / "fusion.json").read_text())
    del meta["video"], meta["fps_source"]
    (result / "fusion.json").write_text(json.dumps(meta))
    assert visualize.render_preview(result, result / "p.mp4").is_file()
