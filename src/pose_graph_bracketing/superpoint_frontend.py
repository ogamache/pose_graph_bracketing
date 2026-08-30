"""SuperPoint feature extraction, as an alternative to DISK (select via
Config.frontend == "superpoint_lightglue"). Reuses the existing kornia-based
`matching.LightGlueMatcher` unchanged -- it already ships a "superpoint"
weight preset (`cfg.lightglue.feature_name = "superpoint"`), so only a
matching extractor was missing. Sourced from the official `lightglue` package
(github.com/cvg/LightGlue, installed as a git dependency -- not on PyPI) for
its `SuperPoint` class, which is the reference implementation the kornia
"superpoint" LightGlue weights were trained against; its own bundled matcher
is not used here.

Recovered from vision-refine-oscillation's history (commit 8293e32~1) --
built and tested there, found worse than DISK in both target regions on the
_10fps dataset pair, removed during cleanup. Re-tested here on the _0fps
pair, which has shown a much clearer/different signal for other ablations
in this scale-bias investigation -- see docs/cycle_bias_findings.md.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from lightglue import SuperPoint

from pose_graph_bracketing.config import SuperPointConfig, TrackingConfig
from pose_graph_bracketing.features import FrameFeatures, _resolve_device, bucket_keypoints


class SuperPointExtractor:
    def __init__(self, cfg: SuperPointConfig, tracking_cfg: TrackingConfig):
        self.cfg = cfg
        self.tracking_cfg = tracking_cfg
        self.device = _resolve_device(cfg.device)
        # max_num_keypoints is oversampled the same way DiskExtractor does,
        # so bucket_keypoints has real candidates to redistribute from.
        self.model = SuperPoint(max_num_keypoints=cfg.max_keypoints * 3).to(self.device).eval()

    @torch.inference_mode()
    def extract(self, image: np.ndarray, max_corners: int | None = None) -> FrameFeatures:
        """Mirrors DiskExtractor.extract's interface -- `image` is a
        single-channel grayscale uint8 image (the pipeline is grayscale
        end-to-end -- see imaging.py), or an already-normalized float [0, 1]
        grayscale image; `max_corners`, if given, overrides
        `cfg.max_keypoints` for this call only (see DiskExtractor.extract)."""
        n_corners = max_corners if max_corners is not None else self.cfg.max_keypoints
        h, w = image.shape[:2]
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0)
        if image.dtype == np.uint8:
            tensor = tensor / 255.0
        tensor = tensor.to(self.device)

        out = self.model({"image": tensor})
        keypoints = out["keypoints"][0].detach().cpu().numpy().astype(np.float32)
        descriptors = out["descriptors"][0].detach().cpu().numpy().astype(np.float32)
        scores = out["keypoint_scores"][0].detach().cpu().numpy().astype(np.float32)

        if keypoints.shape[0] == 0:
            return FrameFeatures(keypoints, descriptors, scores)

        keep = bucket_keypoints(
            keypoints,
            scores,
            (h, w),
            n_corners,
            self.tracking_cfg.grid_rows,
            self.tracking_cfg.grid_cols,
        )
        return FrameFeatures(keypoints[keep], descriptors[keep], scores[keep])
