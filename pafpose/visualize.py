"""Preview animation of the fused 3D skeleton, optionally next to the source video.

Reads ``fused.npz`` produced by ``pafpose run`` / ``pafpose fuse`` and writes an mp4 or gif.
By default only the 3D skeleton is rendered; ``with_video`` adds the source frame on the right.
Rendering uses matplotlib (Agg) on the host; no container is needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .fusion import BODY8_EDGES, HAND21_EDGES

# Index ranges inside wholebody120 = body8 + left21 + right21 + face70.
BODY = slice(0, 8)
LEFT = slice(8, 29)
RIGHT = slice(29, 50)
FACE = slice(50, 120)

# iBUG-68 connectivity (open chains and closed loops), then the two eye centers are drawn as points.
_FACE_CHAINS = [list(range(0, 17)), list(range(17, 22)), list(range(22, 27)), list(range(27, 31)), list(range(31, 36))]
_FACE_LOOPS = [list(range(36, 42)), list(range(42, 48)), list(range(48, 60)), list(range(60, 68))]
FACE_EDGES = np.asarray(
    [(a, b) for chain in _FACE_CHAINS for a, b in zip(chain[:-1], chain[1:])]
    + [(loop[i], loop[(i + 1) % len(loop)]) for loop in _FACE_LOOPS for i in range(len(loop))],
    dtype=np.int64,
)

COLORS = {"body": "#1f77b4", "left": "#2ca02c", "right": "#d62728", "face": "#ff7f0e"}


@dataclass(frozen=True)
class RenderOptions:
    elev: float = 10.0
    azim: float = -90.0          # camera in front of the person (looking along +y, the depth axis)
    panel_height: int = 480      # pixels; the video is resized to this height
    dpi: int = 100
    stride: int = 1              # render every n-th frame
    max_frames: int | None = None
    fps: float | None = None     # output fps; default = source fps / stride
    gif_width: int = 800         # total width of the gif, keeps gifs small
    title: str = ""
    with_video: bool = False     # also show the source video frame on the right


def load_fused(result_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    result_dir = Path(result_dir)
    with np.load(result_dir / "fused.npz") as data:
        wb = np.asarray(data["wholebody120_xyz"], dtype=np.float32)
        valid = {k: np.asarray(data[k]).astype(bool) for k in ("body_valid", "left_hand_valid", "right_hand_valid", "face_valid")}
    meta_path = result_dir / "fusion.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    part_valid = np.stack([valid["body_valid"], valid["left_hand_valid"], valid["right_hand_valid"], valid["face_valid"]], axis=1)
    return wb, part_valid, meta


def axis_limits(wb: np.ndarray, part_valid: np.ndarray) -> tuple[np.ndarray, float]:
    """Fixed cube around all valid joints so the skeleton does not jump between frames."""
    pts = []
    for part, sl in ((0, BODY), (1, LEFT), (2, RIGHT), (3, FACE)):
        mask = part_valid[:, part]
        if mask.any():
            pts.append(wb[mask][:, sl].reshape(-1, 3))
    if not pts:
        return np.zeros(3, dtype=np.float32), 1.0
    allpts = np.concatenate(pts)
    allpts = allpts[np.isfinite(allpts).all(axis=1)]
    lo, hi = np.percentile(allpts, 1, axis=0), np.percentile(allpts, 99, axis=0)
    center = (lo + hi) / 2
    half = float(np.max(hi - lo)) * 0.5 or 1.0
    return center, half


class SkeletonRenderer:
    """Renders one frame of wholebody120 into an RGB array with matplotlib."""

    def __init__(self, wb: np.ndarray, part_valid: np.ndarray, options: RenderOptions):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.options = options
        self.center, self.half = axis_limits(wb, part_valid)
        size_in = options.panel_height / options.dpi
        self.fig = plt.figure(figsize=(size_in, size_in), dpi=options.dpi)
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=0.9)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.plt = plt

    def close(self) -> None:
        self.plt.close(self.fig)

    def _segments(self, pts: np.ndarray, edges: np.ndarray, color: str, lw: float) -> None:
        for a, b in edges:
            p, q = pts[a], pts[b]
            if np.isfinite(p).all() and np.isfinite(q).all():
                self.ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color=color, linewidth=lw)

    def render(self, joints: np.ndarray, valid: np.ndarray, label: str) -> np.ndarray:
        ax = self.ax
        ax.cla()
        c, h = self.center, self.half
        ax.set_xlim(c[0] - h, c[0] + h)
        ax.set_ylim(c[1] - h, c[1] + h)
        ax.set_zlim(c[2] - h, c[2] + h)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=self.options.elev, azim=self.options.azim)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xlabel("x", labelpad=-10); ax.set_ylabel("y (depth)", labelpad=-10); ax.set_zlabel("z (up)", labelpad=-10)
        ax.grid(False)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_alpha(0.15)
        if valid[0]:
            body = joints[BODY]
            self._segments(body, BODY8_EDGES, COLORS["body"], 2.5)
            ax.scatter(body[:, 0], body[:, 1], body[:, 2], color=COLORS["body"], s=14)
        for part, sl, key in ((1, LEFT, "left"), (2, RIGHT, "right")):
            if valid[part]:
                self._segments(joints[sl], HAND21_EDGES, COLORS[key], 1.2)
        if valid[3]:
            face = joints[FACE]
            self._segments(face[:68], FACE_EDGES, COLORS["face"], 0.8)
            ax.scatter(face[68:70, 0], face[68:70, 1], face[68:70, 2], color=COLORS["face"], s=10)
        ax.set_title(label, fontsize=8, pad=2)
        self.fig.canvas.draw()
        rgb = np.asarray(self.fig.canvas.buffer_rgba())[..., :3]
        return np.ascontiguousarray(rgb)


def iter_video_frames(video: Path) -> Iterator[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(str(video))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame[..., ::-1]  # BGR -> RGB
    finally:
        cap.release()


def video_fps(video: Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return fps or 30.0


def compose(left: np.ndarray, right: np.ndarray, height: int) -> np.ndarray:
    import cv2

    def fit(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        return cv2.resize(img, (max(1, int(round(w * height / h))), height), interpolation=cv2.INTER_AREA)

    return np.concatenate([fit(left), fit(right)], axis=1)


def render_preview(
    result_dir: Path,
    video: Path | None,
    out_path: Path,
    options: RenderOptions | None = None,
) -> Path:
    """Write an mp4 (or gif when out_path ends with .gif) of the 3D skeleton, plus the video when requested."""
    import cv2

    options = options or RenderOptions()
    result_dir = Path(result_dir)
    wb, part_valid, meta = load_fused(result_dir)
    if video is None and "video" in meta:
        video = Path(meta["video"])
    video = Path(video) if video is not None else None
    if options.with_video:
        if video is None:
            raise ValueError("fusion.json has no 'video' entry; pass --video to use --with-video")
        if not video.is_file():
            raise FileNotFoundError(f"video not found: {video}")

    src_fps = video_fps(video) if video is not None and video.is_file() else 30.0
    out_fps = options.fps or (src_fps / options.stride)
    sel = meta.get("selection", {})
    name = video.stem if video is not None else result_dir.name
    title = options.title or "{}\nbody {} | hand {} | face {}".format(
        name[:40], *(sel.get(k, "?") for k in ("body", "hand", "face"))
    )
    renderer = SkeletonRenderer(wb, part_valid, options)
    frames: list[np.ndarray] = []
    writer = None
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    as_gif = out_path.suffix.lower() == ".gif"
    rendered = 0
    if options.with_video:
        source: Iterator[tuple[int, np.ndarray | None]] = enumerate(iter_video_frames(video))  # type: ignore[arg-type]
    else:
        source = ((i, None) for i in range(wb.shape[0]))
    try:
        for index, frame in source:
            if index >= wb.shape[0]:
                break
            if index % options.stride:
                continue
            if options.max_frames is not None and rendered >= options.max_frames:
                break
            status = "".join("BLRF"[i] if part_valid[index, i] else "-" for i in range(4))
            panel = renderer.render(wb[index], part_valid[index], f"{title}\nframe {index}   valid {status}")
            if frame is not None:
                panel = compose(panel, frame, options.panel_height)
            if as_gif:
                if panel.shape[1] > options.gif_width:
                    scale = options.gif_width / panel.shape[1]
                    panel = cv2.resize(panel, (options.gif_width, int(panel.shape[0] * scale)), interpolation=cv2.INTER_AREA)
                frames.append(panel)
            else:
                if writer is None:
                    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (panel.shape[1], panel.shape[0]))
                writer.write(panel[..., ::-1])
            rendered += 1
    finally:
        renderer.close()
        if writer is not None:
            writer.release()
    if rendered == 0:
        raise ValueError("no frames rendered (empty video or fused.npz)")
    if as_gif:
        from PIL import Image

        images = [Image.fromarray(f) for f in frames]
        images[0].save(out_path, save_all=True, append_images=images[1:], duration=int(round(1000 / out_fps)), loop=0, optimize=True)
    return out_path
