"""Recorded (not live) diagnostic visualization of the stereo landmark-BA frontend.

Renders one composite frame per processed image, showing:
  - the current frame's keypoints, colored by whether they ended up used in
    the optimization this round, had valid stereo depth but went unused, or
    never had valid stereo depth at all;
  - for each lookback frame matched against, a side-by-side panel with lines
    connecting matched keypoints, colored by whether that match was kept
    (added/extended a landmark factor) or discarded, with the discard reason
    shown in the frame's console log (see graph_builder.py).

This only covers keypoints/matches that survive DISK's own grid-bucketing
(features.DiskExtractor) -- candidates dropped at that earlier stage never
exist as data we could draw (see docs/pipeline.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

_KEYPOINT_COLORS = {
    "used": (0, 200, 0),  # green (BGR)
    "has_depth_unused": (0, 220, 220),  # yellow
    "no_depth": (110, 110, 110),  # gray
}

_MATCH_COLOR_KEPT = (0, 220, 0)  # green
_MATCH_COLOR_DISCARDED = (0, 0, 220)  # red

_LEGEND_TEXT = (
    "keypoints: green=used  yellow=has-depth,unused  gray=no-depth   |   "
    "matches: green=used  red=discarded"
)


@dataclass
class MatchRecord:
    idx_a: int  # index into the older (lookback) frame's keypoints
    idx_b: int  # index into the current frame's keypoints
    kept: bool
    reason: str


@dataclass
class LookbackPanelData:
    frame_idx: int
    image: np.ndarray
    keypoints: np.ndarray
    matches: list[MatchRecord] = field(default_factory=list)


def _resize_width(image: np.ndarray, width: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = width / w
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (width, new_h)), scale


def _pad_to_height(image: np.ndarray, height: int) -> np.ndarray:
    h = image.shape[0]
    if h >= height:
        return image[:height]
    pad = np.zeros((height - h, image.shape[1], 3), dtype=image.dtype)
    return np.vstack([image, pad])


def _draw_keypoints(image: np.ndarray, keypoints: np.ndarray, status: dict[int, str]) -> np.ndarray:
    out = image.copy()
    for idx in range(len(keypoints)):
        color = _KEYPOINT_COLORS[status.get(idx, "no_depth")]
        x, y = keypoints[idx]
        cv2.circle(out, (int(round(x)), int(round(y))), 3, color, -1, lineType=cv2.LINE_AA)
    return out


def _make_top_panel(image_i: np.ndarray, keypoints_i: np.ndarray, keypoint_status: dict[int, str], width: int) -> np.ndarray:
    annotated = _draw_keypoints(image_i, keypoints_i, keypoint_status)
    resized, _ = _resize_width(annotated, width)
    return resized


def _make_pair_panel(
    image_i: np.ndarray, keypoints_i: np.ndarray, panel: LookbackPanelData | None, width: int
) -> np.ndarray:
    """Renders one [lookback frame | current frame] row, or a fixed-size blank
    placeholder if `panel` is None (e.g. early frames with no lookback frame
    yet, or a lookback slot with zero recorded matches). The video writer
    needs every frame to be the exact same size, so this must never vary the
    output shape based on how much real data is available.
    """
    half_w = width // 2
    resized_i, scale_i = _resize_width(image_i, half_w)
    h = resized_i.shape[0]

    if panel is None:
        blank = np.full((h, width, 3), 40, dtype=np.uint8)
        cv2.putText(blank, "(no lookback frame)", (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
        return blank

    resized_j, scale_j = _resize_width(panel.image, half_w)
    resized_j = _pad_to_height(resized_j, h)
    resized_i = _pad_to_height(resized_i, h)
    row = np.hstack([resized_j, resized_i])

    n_kept = sum(1 for m in panel.matches if m.kept)
    for m in panel.matches:
        xj, yj = panel.keypoints[m.idx_a] * scale_j
        xi, yi = keypoints_i[m.idx_b] * scale_i
        color = _MATCH_COLOR_KEPT if m.kept else _MATCH_COLOR_DISCARDED
        cv2.line(row, (int(xj), int(yj)), (int(xi) + half_w, int(yi)), color, 1, lineType=cv2.LINE_AA)

    label = f"frame {panel.frame_idx}  matches kept={n_kept}/{len(panel.matches)}"
    cv2.putText(row, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return row


def _make_text_strip(text: str, width: int, height: int) -> np.ndarray:
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(strip, text, (8, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return strip


def render_frame(
    image_i: np.ndarray,
    keypoints_i: np.ndarray,
    keypoint_status: dict[int, str],
    lookback_panels: list[LookbackPanelData | None],
    banner_text: str,
    n_panel_rows: int,
    panel_width: int = 960,
) -> np.ndarray:
    """Builds one composite diagnostic frame (see module docstring for legend).

    `lookback_panels` is padded/truncated to exactly `n_panel_rows` entries
    (with None -> blank placeholder row) so every rendered frame is the same
    pixel size, as required by cv2.VideoWriter.
    """
    padded: list[LookbackPanelData | None] = list(lookback_panels[:n_panel_rows])
    padded += [None] * (n_panel_rows - len(padded))

    rows = [
        _make_text_strip(_LEGEND_TEXT, panel_width, 22),
        _make_top_panel(image_i, keypoints_i, keypoint_status, panel_width),
    ]
    for panel in padded:
        rows.append(_make_pair_panel(image_i, keypoints_i, panel, panel_width))
    rows.append(_make_text_strip(banner_text, panel_width, 30))
    return np.vstack(rows)


class LiveViewer:
    """Shows each composite diagnostic frame in a cv2 window as it's built, for
    watching the SLAM frontend run in real time.

    In free-running mode (`step=False`), `cv2.waitKey(1)` both pumps the
    window's event loop and gives the frame a minimum display time; it never
    blocks processing. In step mode (`step=True`), `cv2.waitKey(0)` blocks
    until a key is pressed before returning, so the caller advances to the
    next frame one keypress at a time -- any key advances, 'q'/ESC quits.
    """

    QUIT_KEYS = {ord("q"), 27}  # 27 = ESC

    def __init__(self, window_name: str = "pose_graph_bracketing", initial_height: int = 1000, step: bool = False):
        self.window_name = window_name
        self.initial_height = initial_height  # composite is tall+narrow (stacked panels); this stays readable on-screen
        self.step = step
        self._opened = False

    def show(self, frame_bgr: np.ndarray) -> bool:
        """Displays one frame. Returns True if the user requested to quit
        ('q'/ESC), in which case the caller should stop processing further
        frames."""
        if not self._opened:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            h, w = frame_bgr.shape[:2]
            scale = self.initial_height / h
            cv2.resizeWindow(self.window_name, int(round(w * scale)), self.initial_height)
            self._opened = True
        cv2.imshow(self.window_name, frame_bgr)
        key = cv2.waitKey(0 if self.step else 1) & 0xFF
        return key in self.QUIT_KEYS

    def close(self) -> None:
        if self._opened:
            cv2.destroyWindow(self.window_name)
            self._opened = False


class TrajectoryPlotter:
    """Renders a bird's-eye (top-down) view of the estimated trajectory plus a
    height-over-time strip, redrawn from the full position history each call
    (trajectories here are at most a few thousand poses, so this is cheap).

    Poses live in the rectified-left-camera frame throughout this pipeline
    -- OpenCV/GTSAM's stereo convention: X=right, Y=down, Z=forward. So
    "bird's-eye" is the (X, Z) plane, and "height" is -Y (up is negative Y).
    """

    _PATH_COLOR = (0, 200, 255)  # orange (BGR)
    _START_COLOR = (0, 200, 0)  # green
    _CURRENT_COLOR = (0, 0, 220)  # red
    _BG_COLOR = (25, 25, 25)
    _TEXT_COLOR = (200, 200, 200)

    def __init__(self, width: int = 700, bird_height: int = 700, height_strip_height: int = 200, margin_px: int = 40):
        self.width = width
        self.bird_height = bird_height
        self.height_strip_height = height_strip_height
        self.margin_px = margin_px
        self.positions: list[tuple[float, float, float]] = []  # (x, y, z), camera-frame convention

    def set_positions(self, positions: list[tuple[float, float, float]]) -> None:
        """Replaces the full position history (frame index order). Callers re-read
        every still-active (not yet marginalized) pose from the smoother each
        frame, so already-plotted poses can be corrected as the incremental BA
        refines them, not just frozen at their first estimate."""
        self.positions = positions

    def _render_bird(self) -> np.ndarray:
        w, h = self.width, self.bird_height
        canvas = np.full((h, w, 3), self._BG_COLOR, dtype=np.uint8)
        if self.positions:
            xs = [p[0] for p in self.positions]
            zs = [p[2] for p in self.positions]
            span = max(max(xs) - min(xs), max(zs) - min(zs), 1e-3) * 1.15
            cx, cz = (min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2
            scale = (min(w, h) - 2 * self.margin_px) / span

            def to_px(x: float, z: float) -> tuple[int, int]:
                # Forward (+Z) drawn as "up" on screen, right (+X) drawn as right.
                return int(round(w / 2 + (x - cx) * scale)), int(round(h / 2 - (z - cz) * scale))

            pts = [to_px(x, z) for x, _, z in self.positions]
            for p0, p1 in zip(pts, pts[1:]):
                cv2.line(canvas, p0, p1, self._PATH_COLOR, 2, lineType=cv2.LINE_AA)
            cv2.circle(canvas, pts[0], 5, self._START_COLOR, -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, pts[-1], 6, self._CURRENT_COLOR, -1, lineType=cv2.LINE_AA)
        cv2.putText(canvas, "bird's-eye  (X=right, Z=forward)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._TEXT_COLOR, 1, cv2.LINE_AA)
        return canvas

    def _render_height(self) -> np.ndarray:
        w, h = self.width, self.height_strip_height
        canvas = np.full((h, w, 3), self._BG_COLOR, dtype=np.uint8)
        if self.positions:
            heights = [-p[1] for p in self.positions]  # up = -Y
            h_min, h_max = min(heights), max(heights)
            h_range = max(h_max - h_min, 1e-3)
            n = len(heights)

            def to_px(i: int, height_m: float) -> tuple[int, int]:
                px = int(round(self.margin_px + i / max(n - 1, 1) * (w - 2 * self.margin_px)))
                py = int(round(h - self.margin_px - (height_m - h_min) / h_range * (h - 2 * self.margin_px)))
                return px, py

            pts = [to_px(i, hm) for i, hm in enumerate(heights)]
            for p0, p1 in zip(pts, pts[1:]):
                cv2.line(canvas, p0, p1, self._PATH_COLOR, 2, lineType=cv2.LINE_AA)
            cv2.circle(canvas, pts[-1], 4, self._CURRENT_COLOR, -1, lineType=cv2.LINE_AA)
            cv2.putText(
                canvas, f"height (up=-Y): {heights[-1]:+.2f} m  [{h_min:+.2f}, {h_max:+.2f}]",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._TEXT_COLOR, 1, cv2.LINE_AA,
            )
        else:
            cv2.putText(canvas, "height (up=-Y)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._TEXT_COLOR, 1, cv2.LINE_AA)
        return canvas

    def render(self) -> np.ndarray:
        return np.vstack([self._render_bird(), self._render_height()])


class TrajectoryLiveViewer:
    """Shows a live-updating TrajectoryPlotter render in its own cv2 window,
    separate from the diagnostic LiveViewer window so its own varying content
    doesn't interfere with it. Always non-blocking -- step-mode pausing is
    owned by the diagnostic LiveViewer; this window just redraws whatever
    it's given.
    """

    def __init__(self, window_name: str = "pose_graph_bracketing_trajectory"):
        self.window_name = window_name
        self.plotter = TrajectoryPlotter()
        self._opened = False

    def update(self, positions: list[tuple[float, float, float]]) -> None:
        self.plotter.set_positions(positions)
        canvas = self.plotter.render()
        if not self._opened:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            h, w = canvas.shape[:2]
            cv2.resizeWindow(self.window_name, w, h)
            self._opened = True
        cv2.imshow(self.window_name, canvas)
        cv2.waitKey(1)

    def close(self) -> None:
        if self._opened:
            cv2.destroyWindow(self.window_name)
            self._opened = False
