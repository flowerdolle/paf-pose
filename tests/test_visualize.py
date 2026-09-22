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
    fused.save(result, extra_meta={"video": str(tmp_path / "clip.mp4"), "selection": {"body": "a", "hand": "b", "face": "c"}})
    cv2 = pytest.importorskip("cv2")
    writer = cv2.VideoWriter(str(tmp_path / "clip.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for i in range(T):
        writer.write(np.full((48, 64, 3), i * 30, dtype=np.uint8))
    writer.release()
    return result


def test_face_edges_are_valid_indices():
    assert visualize.FACE_EDGES.min() >= 0 and visualize.FACE_EDGES.max() < 68


def test_render_preview_mp4_and_gif(tmp_path):
    result = _synthetic_result(tmp_path)
    cv2 = pytest.importorskip("cv2")
    out = visualize.render_preview(result, None, result / "preview.mp4", visualize.RenderOptions(panel_height=96))
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 6
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape[:2] == (96, 96)  # skeleton only: square panel

    both = visualize.render_preview(result, None, result / "both.mp4", visualize.RenderOptions(panel_height=96, with_video=True))
    cap = cv2.VideoCapture(str(both))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape[0] == 96 and frame.shape[1] > 96  # skeleton + video side by side

    gif = visualize.render_preview(result, None, result / "preview.gif", visualize.RenderOptions(panel_height=96, stride=2, gif_width=200))
    from PIL import Image

    with Image.open(gif) as im:
        assert im.n_frames == 3 and im.size[0] <= 200


def test_render_preview_without_video_entry(tmp_path):
    result = _synthetic_result(tmp_path)
    meta = json.loads((result / "fusion.json").read_text())
    del meta["video"]
    (result / "fusion.json").write_text(json.dumps(meta))
    assert visualize.render_preview(result, None, result / "p.mp4").is_file()  # skeleton-only needs no video
    with pytest.raises(ValueError, match="pass --video"):
        visualize.render_preview(result, None, result / "q.mp4", visualize.RenderOptions(with_video=True))
