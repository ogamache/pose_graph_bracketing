"""Incremental (iSAM2) pose-graph construction over a bracketed-exposure sequence."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import gtsam
import numpy as np

from pose_graph_bracketing.config import Config
from pose_graph_bracketing.dataset import FrameInfo
from pose_graph_bracketing.factors import (
    predict_pose,
    make_motion_prior_factor,
    make_metric_vo_factor,
    motion_prior_noise_model,
)
from pose_graph_bracketing.features import DiskExtractor, FrameFeatures
from pose_graph_bracketing.imaging import load_preprocessed
from pose_graph_bracketing.matching import LightGlueMatcher
from pose_graph_bracketing.odometry import estimate_pose_pnp
from pose_graph_bracketing.stereo import StereoRig, Triangulated, triangulate_stereo

log = logging.getLogger(__name__)


def _pose_key(idx: int) -> int:
    return gtsam.symbol("x", idx)


def _vel_key(idx: int) -> int:
    return gtsam.symbol("v", idx)


@dataclass
class FrameResult:
    frame: FrameInfo
    pose: gtsam.Pose3
    velocity: np.ndarray
    n_vo_factors: int


class PoseGraphBuilder:
    def __init__(self, cfg: Config, rig: StereoRig):
        self.cfg = cfg
        self.rig = rig
        self.extractor = DiskExtractor(cfg.disk, cfg.tracking)
        self.matcher = LightGlueMatcher(cfg.lightglue)

        params = gtsam.ISAM2Params()
        self.isam = gtsam.ISAM2(params)
        self.current_estimate = gtsam.Values()

        self._feature_cache: dict[int, tuple[FrameFeatures, tuple[int, int]]] = {}
        self._right_feature_cache: dict[int, FrameFeatures] = {}
        self._triangulated_cache: dict[int, Triangulated] = {}
        self.results: list[FrameResult] = []

    def _get_left_features(self, idx: int, frame: FrameInfo) -> tuple[FrameFeatures, tuple[int, int]]:
        cached = self._feature_cache.get(idx)
        if cached is not None:
            return cached
        image = load_preprocessed(
            frame.image_path, self.cfg.dataset.bayer_pattern, self.cfg.preprocessing.crop_bottom_px
        )
        feats = self.extractor.extract(image)
        entry = (feats, image.shape[:2])
        self._feature_cache[idx] = entry
        return entry

    def _get_right_features(self, idx: int, frame: FrameInfo) -> FrameFeatures:
        cached = self._right_feature_cache.get(idx)
        if cached is not None:
            return cached
        image = load_preprocessed(
            frame.right_image_path, self.cfg.dataset.bayer_pattern, self.cfg.preprocessing.crop_bottom_px
        )
        feats = self.extractor.extract(image)
        self._right_feature_cache[idx] = feats
        return feats

    def _get_triangulated(self, idx: int, frame: FrameInfo) -> Triangulated:
        cached = self._triangulated_cache.get(idx)
        if cached is not None:
            return cached
        feats_left, shape_left = self._get_left_features(idx, frame)
        feats_right = self._get_right_features(idx, frame)
        match = self.matcher.match(feats_left, shape_left, feats_right, shape_left)
        tri = triangulate_stereo(
            feats_left.keypoints,
            feats_right.keypoints,
            match.indices_a,
            match.indices_b,
            self.rig,
            self.cfg.stereo.min_disparity_px,
            self.cfg.stereo.max_depth_m,
        )
        self._triangulated_cache[idx] = tri
        return tri

    def _evict_old_features(self, current_idx: int) -> None:
        cutoff = current_idx - self.cfg.graph.vo_lookback
        for cache in (self._feature_cache, self._right_feature_cache, self._triangulated_cache):
            for k in [k for k in cache if k < cutoff]:
                del cache[k]

    def _try_vo_factor(self, j: int, i: int, frames: list[FrameInfo]) -> gtsam.BetweenFactorPose3 | None:
        feats_j, shape_j = self._get_left_features(j, frames[j])
        feats_i, shape_i = self._get_left_features(i, frames[i])
        tri_j = self._get_triangulated(j, frames[j])

        if len(tri_j.indices_left) < self.cfg.stereo.min_pnp_points:
            return None

        match = self.matcher.match(feats_j, shape_j, feats_i, shape_i)
        if match.indices_a.shape[0] == 0:
            return None

        # Intersect: left-left matches (j -> i) whose j-side keypoint also has
        # a valid stereo-triangulated 3D point.
        point3d_by_left_idx = dict(zip(tri_j.indices_left.tolist(), tri_j.points3d))
        obj_pts = []
        img_pts = []
        for idx_a, idx_b in zip(match.indices_a, match.indices_b):
            p3d = point3d_by_left_idx.get(int(idx_a))
            if p3d is not None:
                obj_pts.append(p3d)
                img_pts.append(feats_i.keypoints[idx_b])

        if len(obj_pts) < self.cfg.stereo.min_pnp_points:
            return None

        est = estimate_pose_pnp(
            np.asarray(obj_pts), np.asarray(img_pts), self.rig.K_left, self.rig.dist_left, self.cfg.stereo, shape_i
        )
        if est is None or not est.reliable:
            return None

        return make_metric_vo_factor(
            _pose_key(j), _pose_key(i), est.R, est.t, self.cfg.stereo.rotation_sigma, self.cfg.stereo.translation_sigma
        )

    def process_frame(self, idx: int, frames: list[FrameInfo]) -> FrameResult:
        frame = frames[idx]
        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()

        X_i = _pose_key(idx)
        V_i = _vel_key(idx)

        # Ensure features/triangulation for this frame (and its VO-lookback window) are ready.
        self._get_triangulated(idx, frame)

        if idx == 0:
            initial.insert(X_i, gtsam.Pose3())
            initial.insert(V_i, np.zeros(6))
            graph.add(
                gtsam.PriorFactorPose3(
                    X_i,
                    gtsam.Pose3(),
                    gtsam.noiseModel.Isotropic.Sigma(6, self.cfg.motion_prior.initial_pose_prior_sigma),
                )
            )
            graph.add(
                gtsam.PriorFactorVector(
                    V_i,
                    np.zeros(6),
                    gtsam.noiseModel.Isotropic.Sigma(6, self.cfg.motion_prior.initial_velocity_prior_sigma),
                )
            )
        else:
            prev_frame = frames[idx - 1]
            dt = frame.timestamp_s - prev_frame.timestamp_s
            X_prev = _pose_key(idx - 1)
            V_prev = _vel_key(idx - 1)

            prev_pose = self.current_estimate.atPose3(X_prev)
            prev_vel = self.current_estimate.atVector(V_prev)
            predicted_pose = predict_pose(prev_pose, prev_vel, dt)

            initial.insert(X_i, predicted_pose)
            initial.insert(V_i, prev_vel)

            mp_noise = motion_prior_noise_model(
                self.cfg.motion_prior.rotation_sigma,
                self.cfg.motion_prior.translation_sigma,
                self.cfg.motion_prior.angular_velocity_rw_sigma,
                self.cfg.motion_prior.linear_velocity_rw_sigma,
                dt,
            )
            graph.add(make_motion_prior_factor(X_prev, V_prev, X_i, V_i, dt, mp_noise))

        n_vo = 0
        lookback_start = max(0, idx - self.cfg.graph.vo_lookback)
        for j in range(lookback_start, idx):
            factor = self._try_vo_factor(j, idx, frames)
            if factor is not None:
                graph.add(factor)
                n_vo += 1

        self.isam.update(graph, initial)
        self.current_estimate = self.isam.calculateEstimate()
        self._evict_old_features(idx)

        result = FrameResult(
            frame=frame,
            pose=self.current_estimate.atPose3(X_i),
            velocity=self.current_estimate.atVector(V_i),
            n_vo_factors=n_vo,
        )
        self.results.append(result)
        return result

    def run(self, frames: list[FrameInfo]) -> list[FrameResult]:
        for idx in range(len(frames)):
            result = self.process_frame(idx, frames)
            if idx % 50 == 0 or idx == len(frames) - 1:
                log.info(
                    "frame %d/%d ts=%d slot=%s vo_factors=%d",
                    idx,
                    len(frames) - 1,
                    result.frame.timestamp_ns,
                    result.frame.slot_label,
                    result.n_vo_factors,
                )
        return self.results
