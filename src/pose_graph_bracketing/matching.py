"""LightGlue matching of DISK features."""

from __future__ import annotations

from dataclasses import dataclass

import kornia.feature as KF
import numpy as np
import torch

from pose_graph_bracketing.config import LightGlueConfig
from pose_graph_bracketing.features import FrameFeatures, _resolve_device


@dataclass
class MatchResult:
    indices_a: np.ndarray  # (M,) indices into features_a
    indices_b: np.ndarray  # (M,) indices into features_b
    confidence: np.ndarray  # (M,) float32, higher = better


class LightGlueMatcher:
    def __init__(self, cfg: LightGlueConfig):
        self.cfg = cfg
        self.device = _resolve_device(cfg.device)
        self.matcher = KF.LightGlueMatcher(cfg.feature_name).to(self.device).eval()

    @torch.inference_mode()
    def match(
        self,
        feats_a: FrameFeatures,
        shape_a: tuple[int, int],
        feats_b: FrameFeatures,
        shape_b: tuple[int, int],
    ) -> MatchResult:
        if feats_a.keypoints.shape[0] == 0 or feats_b.keypoints.shape[0] == 0:
            return MatchResult(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32))

        desc_a = torch.from_numpy(feats_a.descriptors).to(self.device)
        desc_b = torch.from_numpy(feats_b.descriptors).to(self.device)
        kp_a = torch.from_numpy(feats_a.keypoints).to(self.device)
        kp_b = torch.from_numpy(feats_b.keypoints).to(self.device)

        laf_a = KF.laf_from_center_scale_ori(kp_a[None])
        laf_b = KF.laf_from_center_scale_ori(kp_b[None])

        dists, idxs = self.matcher(desc_a, desc_b, laf_a, laf_b, hw1=shape_a, hw2=shape_b)

        if idxs.numel() == 0:
            return MatchResult(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32))

        idxs_np = idxs.detach().cpu().numpy()
        # kornia's LightGlueMatcher returns the raw LightGlue matching score
        # in [0, 1] (higher = better), not a distance.
        confidence = dists.detach().cpu().numpy().reshape(-1)
        keep = confidence >= self.cfg.min_confidence

        return MatchResult(idxs_np[keep, 0], idxs_np[keep, 1], confidence[keep].astype(np.float32))
