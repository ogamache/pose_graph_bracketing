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


class VideoRecorder:
    """Thin wrapper around cv2.VideoWriter, lazily sized to the first frame."""

    def __init__(self, output_path: str, fps: int):
        self.output_path = output_path
        self.fps = fps
        self._writer: cv2.VideoWriter | None = None

    def write(self, frame_bgr: np.ndarray) -> None:
        if self._writer is None:
            h, w = frame_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (w, h))
        self._writer.write(frame_bgr)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
