"""DISK feature extraction with grid-based keypoint bucketing."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import kornia.feature as KF
import numpy as np
import torch

from pose_graph_bracketing.config import DiskConfig, TrackingConfig


@dataclass
class FrameFeatures:
    keypoints: np.ndarray  # (N, 2) float32, (x, y) pixel coords
    descriptors: np.ndarray  # (N, D) float32
    scores: np.ndarray  # (N,) float32


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def bucket_keypoints(
    keypoints: np.ndarray,
    scores: np.ndarray,
    image_shape: tuple[int, int],
    max_corners: int,
    grid_rows: int,
    grid_cols: int,
) -> np.ndarray:
    """Return indices of keypoints kept after per-cell score-based top-K selection.

    Distributes the `max_corners` budget evenly across a grid_rows x grid_cols
    grid so features aren't dominated by a single high-texture region.
    """
    h, w = image_shape[:2]
    n_cells = grid_rows * grid_cols
    per_cell_budget = max(1, max_corners // n_cells)

    cell_w = w / grid_cols
    cell_h = h / grid_rows

    col_idx = np.clip((keypoints[:, 0] // cell_w).astype(int), 0, grid_cols - 1)
    row_idx = np.clip((keypoints[:, 1] // cell_h).astype(int), 0, grid_rows - 1)
    cell_id = row_idx * grid_cols + col_idx

    keep: list[int] = []
    for c in range(n_cells):
        idx_in_cell = np.nonzero(cell_id == c)[0]
        if idx_in_cell.size == 0:
            continue
        order = np.argsort(-scores[idx_in_cell])[:per_cell_budget]
        keep.extend(idx_in_cell[order].tolist())

    return np.array(keep, dtype=np.int64)


class DiskExtractor:
    def __init__(self, cfg: DiskConfig, tracking_cfg: TrackingConfig):
        self.cfg = cfg
        self.tracking_cfg = tracking_cfg
        self.device = _resolve_device(cfg.device)
        self.model = KF.DISK.from_pretrained(cfg.checkpoint, device=self.device).eval()

    @torch.inference_mode()
    def extract(self, image: np.ndarray) -> FrameFeatures:
        """`image` is a single-channel grayscale uint8 image (the pipeline is
        grayscale end-to-end -- see imaging.py); replicated to 3 channels
        since DISK expects an RGB-shaped tensor."""
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)

        n_request = self.tracking_cfg.max_corners * self.cfg.kp_oversample_factor
        feats = self.model(
            tensor,
            n=n_request,
            window_size=self.cfg.window_size,
            score_threshold=self.cfg.score_threshold,
            pad_if_not_divisible=True,
        )[0]

        keypoints = feats.keypoints.detach().cpu().numpy().astype(np.float32)
        descriptors = feats.descriptors.detach().cpu().numpy().astype(np.float32)
        scores = feats.detection_scores.detach().cpu().numpy().astype(np.float32)

        if keypoints.shape[0] == 0:
            return FrameFeatures(keypoints, descriptors, scores)

        keep = bucket_keypoints(
            keypoints,
            scores,
            (h, w),
            self.tracking_cfg.max_corners,
            self.tracking_cfg.grid_rows,
            self.tracking_cfg.grid_cols,
        )
        return FrameFeatures(keypoints[keep], descriptors[keep], scores[keep])
