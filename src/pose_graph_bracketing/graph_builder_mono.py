"""Incremental monocular bundle-adjustment pose graph over a bracketed-exposure sequence.

Mirrors graph_builder.PoseGraphBuilder's structure (LandmarkTracker, fixed-lag
smoother, diagnostics) but replaces stereo triangulation with two-view mono
triangulation, and GenericStereoFactor3D reprojection factors with
GenericProjectionFactorCal3_S2 ones (see factors.make_mono_observation_factor).

A new landmark's initial 3D value is triangulated from frame j's current pose
estimate and an essential-matrix-estimated relative pose to frame idx (see
_estimate_relative_pose_essential) -- NOT the constant-velocity motion-model
prediction used to initialize X_idx in the graph, since that alone gives zero
baseline at bootstrap (velocity starts at 0), which is degenerate for
triangulation. Monocular SLAM cannot recover metric scale from geometry
alone -- like the (unused) mono VO path, the graph's overall scale is only
weakly resolved via the motion-prior velocity states.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import gtsam
import gtsam_unstable
import numpy as np

from pose_graph_bracketing.calibration import CameraCalibration
from pose_graph_bracketing.config import Config
from pose_graph_bracketing.dataset import FrameInfo
from pose_graph_bracketing.factors import (
    predict_pose,
    make_motion_prior_factor,
    make_mono_observation_factor,
    motion_prior_noise_model,
)
from pose_graph_bracketing.features import DiskExtractor, FrameFeatures
from pose_graph_bracketing.imaging import load_preprocessed
from pose_graph_bracketing.landmarks import LandmarkTracker
from pose_graph_bracketing.matching import LightGlueMatcher
from pose_graph_bracketing.visualization import LookbackPanelData, MatchRecord, VideoRecorder, render_frame

log = logging.getLogger(__name__)


def _pose_key(idx: int) -> int:
    return gtsam.symbol("x", idx)


def _vel_key(idx: int) -> int:
    return gtsam.symbol("v", idx)


def _landmark_key(landmark_id: int) -> int:
    return gtsam.symbol("l", landmark_id)


_ESSENTIAL_MIN_MATCHES = 8
_ESSENTIAL_RANSAC_THRESHOLD_PX = 1.0
_ESSENTIAL_CONFIDENCE = 0.999


def _estimate_relative_pose_essential(
    undist_j: np.ndarray, undist_i: np.ndarray, K: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Essential-matrix 2D-2D relative pose (R, unit-norm t) mapping camera-j points
    into camera-i (p_i ~ R @ p_j + t), or None if there aren't enough correspondences
    or OpenCV fails to produce a usable estimate.

    Used only to seed a new landmark's initial 3D value with a real two-view
    baseline (see MonoPoseGraphBuilder._emit_landmark_observations) -- the
    constant-velocity motion-model prediction alone gives zero baseline at
    bootstrap (velocity starts at 0), which makes two-view triangulation
    degenerate. Translation is scale-free (unit norm); the graph's actual
    scale is resolved (weakly) elsewhere via the motion-prior velocity states.

    Disambiguates the 4 (R, t) hypotheses via a manual cheirality check
    (triangulate + require positive depth in both cameras) rather than
    cv2.recoverPose, which was found unreliable on this data by the (now
    removed) monocular-VO module this pipeline used to have.
    """
    n_matches = len(undist_j)
    if n_matches < _ESSENTIAL_MIN_MATCHES:
        return None

    pts_j = undist_j.reshape(-1, 1, 2).astype(np.float64)
    pts_i = undist_i.reshape(-1, 1, 2).astype(np.float64)

    E, mask = cv2.findEssentialMat(
        pts_j, pts_i, cameraMatrix=K, method=cv2.RANSAC, prob=_ESSENTIAL_CONFIDENCE, threshold=_ESSENTIAL_RANSAC_THRESHOLD_PX
    )
    if E is None or E.shape != (3, 3):
        return None
    inlier_mask = mask.reshape(-1).astype(bool)
    if inlier_mask.sum() < _ESSENTIAL_MIN_MATCHES:
        return None

    R1, R2, t = cv2.decomposeEssentialMat(E)
    t = t.reshape(3, 1)

    pts_j_in = undist_j[inlier_mask]
    pts_i_in = undist_i[inlier_mask]
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])

    best = None
    for R_cand in (R1, R2):
        for sign in (1.0, -1.0):
            t_cand = t * sign
            P2 = K @ np.hstack([R_cand, t_cand])
            pts4d = cv2.triangulatePoints(P1, P2, pts_j_in.T, pts_i_in.T)
            pts3d = (pts4d[:3] / pts4d[3]).T
            depth_j = pts3d[:, 2]
            depth_i = (R_cand @ pts3d.T + t_cand).T[:, 2]
            n_good = int(((depth_j > 0) & (depth_i > 0)).sum())
            if best is None or n_good > best[0]:
                best = (n_good, R_cand, t_cand.reshape(-1))

    n_good, R_best, t_best = best
    if n_good == 0:
        return None
    return R_best, t_best / (np.linalg.norm(t_best) + 1e-12)


_MIN_PARALLAX_DEG = 2.0
_MIN_PARALLAX_COS = np.cos(np.radians(_MIN_PARALLAX_DEG))


def _triangulate_landmark(
    pose_j: gtsam.Pose3,
    pose_i: gtsam.Pose3,
    K_mono: gtsam.Cal3_S2,
    point_j: np.ndarray,
    point_i: np.ndarray,
    max_depth_m: float,
) -> np.ndarray | None:
    """Two-view triangulation of a new landmark from its two (undistorted) sightings.

    Uses the current pose estimate for frame j and an essential-matrix-derived
    pose for the in-progress frame i as the two camera poses -- there is no
    stereo baseline to triangulate against, so the landmark's initial value
    comes entirely from the estimated trajectory so far (weak scale, same as
    the rest of the mono pipeline). Returns None if the point ends up behind
    either camera or beyond max_depth_m (gtsam.TriangulationResult.valid(),
    also checking reprojection error via dynamicOutlierRejectionThreshold), or
    if the two rays have too little parallax (explicit angle check below --
    triangulateSafe's own rank-tolerance gate was found to still occasionally
    let near-degenerate configurations through, which later produced an
    indeterminate linear system in the smoother).
    """
    cameras = gtsam.CameraSetCal3_S2()
    cameras.append(gtsam.PinholeCameraCal3_S2(pose_j, K_mono))
    cameras.append(gtsam.PinholeCameraCal3_S2(pose_i, K_mono))
    measurements = gtsam.Point2Vector([gtsam.Point2(*point_j), gtsam.Point2(*point_i)])

    params = gtsam.TriangulationParameters(
        rankTolerance=1.0,
        enableEPI=False,
        landmarkDistanceThreshold=max_depth_m,
        dynamicOutlierRejectionThreshold=3.0,
    )
    result = gtsam.triangulateSafe(cameras, measurements, params)
    if not result.valid():
        return None
    point = result.get()
    if not np.all(np.isfinite(point)):
        return None

    ray_j = point - pose_j.translation()
    ray_i = point - pose_i.translation()
    norm_j, norm_i = np.linalg.norm(ray_j), np.linalg.norm(ray_i)
    if norm_j < 1e-9 or norm_i < 1e-9:
        return None
    cos_parallax = float(np.dot(ray_j, ray_i) / (norm_j * norm_i))
    if cos_parallax > _MIN_PARALLAX_COS:
        return None

    return point


@dataclass
class FrameResult:
    frame: FrameInfo
    pose: gtsam.Pose3
    velocity: np.ndarray
    n_landmark_observations: int


class MonoPoseGraphBuilder:
    def __init__(self, cfg: Config, calib: CameraCalibration):
        self.cfg = cfg
        self.calib = calib
        fx, fy = calib.K[0, 0], calib.K[1, 1]
        u0, v0 = calib.K[0, 2], calib.K[1, 2]
        self.K_mono = gtsam.Cal3_S2(fx, fy, 0.0, u0, v0)
        self.extractor = DiskExtractor(cfg.disk, cfg.tracking)
        self.matcher = LightGlueMatcher(cfg.lightglue)
        self.landmark_tracker = LandmarkTracker()

        self.smoother = gtsam_unstable.IncrementalFixedLagSmoother(cfg.graph.smoother_lag_s, gtsam.ISAM2Params())
        self.current_estimate = gtsam.Values()

        self._feature_cache: dict[int, tuple[FrameFeatures, tuple[int, int]]] = {}
        self._undistorted_cache: dict[int, np.ndarray] = {}
        self.results: list[FrameResult] = []
        self.zero_obs_frames: list[int] = []  # frame indices where n_landmark_observations==0 this round
        self.n_backend_resets = 0  # count of smoother resets due to indeterminate linear systems (see process_frame)
        self._earliest_valid_pose_idx = 0  # bumped on a backend reset; older poses no longer exist in the smoother

        self._video: VideoRecorder | None = None
        if cfg.visualization.enabled and cfg.visualization.output_path:
            self._video = VideoRecorder(cfg.visualization.output_path, cfg.visualization.fps)

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

    def _get_undistorted(self, idx: int, frame: FrameInfo) -> np.ndarray:
        """Keypoints undistorted to the K_mono pixel frame (matches GenericProjectionFactorCal3_S2's
        assumption of an undistorted Cal3_S2 measurement)."""
        cached = self._undistorted_cache.get(idx)
        if cached is not None:
            return cached
        feats, _ = self._get_left_features(idx, frame)
        if feats.keypoints.shape[0] == 0:
            undist = np.empty((0, 2), dtype=np.float64)
        else:
            pts = feats.keypoints.reshape(-1, 1, 2).astype(np.float64)
            undist = cv2.undistortPoints(pts, self.calib.K, self.calib.dist, P=self.calib.K).reshape(-1, 2)
        self._undistorted_cache[idx] = undist
        return undist

    def _evict_old_features(self, current_idx: int) -> None:
        cutoff = current_idx - self.cfg.graph.vo_lookback
        for cache in (self._feature_cache, self._undistorted_cache):
            for k in [k for k in cache if k < cutoff]:
                del cache[k]
        self.landmark_tracker.evict_before(cutoff)

    def _emit_landmark_observations(
        self,
        idx: int,
        frames: list[FrameInfo],
        graph: gtsam.NonlinearFactorGraph,
        initial: gtsam.Values,
        landmark_timestamps: dict[int, float],
        keypoint_status: dict[int, str] | None = None,
        visual_matches: dict[int, list[MatchRecord]] | None = None,
    ) -> int:
        """Adds mono landmark factors/values for frame `idx` against its lookback
        window. Mirrors graph_builder.PoseGraphBuilder._emit_landmark_observations,
        but seeds a new landmark via two-view mono triangulation from the current
        pose estimates instead of gating on stereo depth availability.
        """
        frame = frames[idx]
        feats_i, shape_i = self._get_left_features(idx, frame)
        undist_i = self._get_undistorted(idx, frame)

        if keypoint_status is not None:
            for kp_idx in range(len(feats_i.keypoints)):
                keypoint_status[kp_idx] = "no_depth"

        n_obs = 0
        lookback_start = max(0, idx - self.cfg.graph.vo_lookback, self._earliest_valid_pose_idx)
        for j in range(lookback_start, idx):
            feats_j, shape_j = self._get_left_features(j, frames[j])
            undist_j = self._get_undistorted(j, frames[j])

            match = self.matcher.match(feats_j, shape_j, feats_i, shape_i)
            if match.indices_a.shape[0] == 0:
                continue

            # Real two-view baseline for triangulating any brand-new landmark
            # in this j->idx pair: the constant-velocity-predicted pose alone
            # gives zero baseline at bootstrap (velocity starts at 0), which
            # makes triangulation degenerate. Computed once per j (not per
            # correspondence) from the full match set between the two frames.
            pose_j_estimate = self.current_estimate.atPose3(_pose_key(j))
            essential_est = _estimate_relative_pose_essential(
                undist_j[match.indices_a], undist_i[match.indices_b], self.calib.K
            )
            pose_i_for_triangulation = (
                pose_j_estimate.compose(gtsam.Pose3(gtsam.Rot3(essential_est[0]), essential_est[1]))
                if essential_est is not None
                else None
            )

            for idx_a_raw, idx_b_raw in zip(match.indices_a, match.indices_b):
                idx_a, idx_b = int(idx_a_raw), int(idx_b_raw)

                def discard(reason: str) -> None:
                    if visual_matches is not None:
                        visual_matches.setdefault(j, []).append(MatchRecord(idx_a, idx_b, kept=False, reason=reason))

                def keep(reason: str) -> None:
                    if visual_matches is not None:
                        visual_matches.setdefault(j, []).append(MatchRecord(idx_a, idx_b, kept=True, reason=reason))
                    if keypoint_status is not None:
                        keypoint_status[idx_b] = "used"

                if self.landmark_tracker.landmark_id_at(idx, idx_b) is not None:
                    discard("already_linked")
                    continue  # already tied to a landmark via an earlier j this frame

                landmark_id = self.landmark_tracker.landmark_id_at(j, idx_a)
                is_new_landmark = landmark_id is None

                if landmark_id is None:
                    if pose_i_for_triangulation is None:
                        discard("essential_matrix_failed")
                        continue
                    point_init = _triangulate_landmark(
                        pose_j_estimate,
                        pose_i_for_triangulation,
                        self.K_mono,
                        undist_j[idx_a],
                        undist_i[idx_b],
                        self.cfg.stereo.max_depth_m,
                    )
                    if point_init is None:
                        discard("triangulation_failed")
                        continue
                    landmark_id, _ = self.landmark_tracker.get_or_create(j, idx_a)
                    landmark_timestamps[landmark_id] = frames[j].timestamp_s
                    initial.insert(_landmark_key(landmark_id), point_init)
                    graph.add(
                        gtsam.PriorFactorPoint3(
                            _landmark_key(landmark_id),
                            point_init,
                            gtsam.noiseModel.Isotropic.Sigma(3, self.cfg.stereo.landmark_prior_sigma),
                        )
                    )
                    graph.add(
                        make_mono_observation_factor(
                            _pose_key(j),
                            _landmark_key(landmark_id),
                            undist_j[idx_a],
                            self.K_mono,
                            self.cfg.stereo.pixel_sigma,
                            self.cfg.stereo.huber_k,
                        )
                    )
                    n_obs += 1

                self.landmark_tracker.get_or_create(idx, idx_b, existing_landmark_id=landmark_id)
                landmark_timestamps[landmark_id] = frame.timestamp_s
                graph.add(
                    make_mono_observation_factor(
                        _pose_key(idx),
                        _landmark_key(landmark_id),
                        undist_i[idx_b],
                        self.K_mono,
                        self.cfg.stereo.pixel_sigma,
                        self.cfg.stereo.huber_k,
                    )
                )
                n_obs += 1
                keep("new_landmark" if is_new_landmark else "extended_landmark")

        return n_obs

    def process_frame(self, idx: int, frames: list[FrameInfo]) -> FrameResult:
        frame = frames[idx]
        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()

        X_i = _pose_key(idx)
        V_i = _vel_key(idx)

        # Ensure features for this frame (and its lookback window) are ready.
        self._get_left_features(idx, frame)

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

        landmark_timestamps: dict[int, float] = {}
        keypoint_status: dict[int, str] | None = {} if self._video is not None else None
        visual_matches: dict[int, list[MatchRecord]] | None = {} if self._video is not None else None
        n_obs = self._emit_landmark_observations(
            idx, frames, graph, initial, landmark_timestamps, keypoint_status, visual_matches
        )
        if idx > 0 and n_obs == 0:
            self.zero_obs_frames.append(idx)

        timestamps = gtsam_unstable.FixedLagSmootherKeyTimestampMap()
        timestamps.insert((X_i, frame.timestamp_s))
        timestamps.insert((V_i, frame.timestamp_s))
        for landmark_id, ts in landmark_timestamps.items():
            timestamps.insert((_landmark_key(landmark_id), ts))

        try:
            self.smoother.update(graph, initial, timestamps)
            self.current_estimate = self.smoother.calculateEstimate()
        except RuntimeError as e:
            # Mono bundle adjustment occasionally seeds/marginalizes a landmark
            # that's numerically underdetermined despite passing the
            # triangulation-validity + parallax gates above (e.g. a Huber-
            # downweighted inlier whose effective information collapses right
            # as it ages out of the fixed-lag window) -- "Indeterminant linear
            # system" from GTSAM's elimination. Rather than crash the whole
            # run, reset the backend: re-anchor a fresh smoother at the last
            # known-good pose/velocity estimate (tightly, since we trust it)
            # and re-run just this frame's motion-prior/pose update with no
            # landmark factors, so the trajectory keeps going. Landmarks lost
            # this way simply get re-seeded on a later lookback pair.
            log.warning("Smoother update failed at frame %d (%s) -- resetting backend and retrying without landmarks", idx, e)
            self.n_backend_resets += 1
            self._earliest_valid_pose_idx = idx  # only X_i/V_i survive the reset below
            self.landmark_tracker = LandmarkTracker()
            self.smoother = gtsam_unstable.IncrementalFixedLagSmoother(self.cfg.graph.smoother_lag_s, gtsam.ISAM2Params())

            reset_graph = gtsam.NonlinearFactorGraph()
            reset_initial = gtsam.Values()
            reset_initial.insert(X_i, initial.atPose3(X_i))
            reset_initial.insert(V_i, initial.atVector(V_i))
            reset_graph.add(
                gtsam.PriorFactorPose3(
                    X_i, initial.atPose3(X_i), gtsam.noiseModel.Isotropic.Sigma(6, self.cfg.motion_prior.initial_pose_prior_sigma)
                )
            )
            reset_graph.add(
                gtsam.PriorFactorVector(
                    V_i, initial.atVector(V_i), gtsam.noiseModel.Isotropic.Sigma(6, self.cfg.motion_prior.initial_velocity_prior_sigma)
                )
            )
            reset_timestamps = gtsam_unstable.FixedLagSmootherKeyTimestampMap()
            reset_timestamps.insert((X_i, frame.timestamp_s))
            reset_timestamps.insert((V_i, frame.timestamp_s))
            self.smoother.update(reset_graph, reset_initial, reset_timestamps)
            self.current_estimate = self.smoother.calculateEstimate()
            n_obs = 0
            if idx not in self.zero_obs_frames:
                self.zero_obs_frames.append(idx)

        if self._video is not None:
            self._render_and_write(idx, frames, keypoint_status, visual_matches, n_obs)

        self._evict_old_features(idx)

        result = FrameResult(
            frame=frame,
            pose=self.current_estimate.atPose3(X_i),
            velocity=self.current_estimate.atVector(V_i),
            n_landmark_observations=n_obs,
        )
        self.results.append(result)
        return result

    def _render_and_write(
        self,
        idx: int,
        frames: list[FrameInfo],
        keypoint_status: dict[int, str],
        visual_matches: dict[int, list[MatchRecord]],
        n_obs: int,
    ) -> None:
        frame = frames[idx]
        image_i = load_preprocessed(frame.image_path, self.cfg.dataset.bayer_pattern, self.cfg.preprocessing.crop_bottom_px)
        feats_i, _ = self._get_left_features(idx, frame)

        lookback_start = max(0, idx - self.cfg.graph.vo_lookback)
        panels = []
        for j in range(lookback_start, idx):
            matches = visual_matches.get(j, [])
            if not matches:
                continue
            frame_j = frames[j]
            image_j = load_preprocessed(
                frame_j.image_path, self.cfg.dataset.bayer_pattern, self.cfg.preprocessing.crop_bottom_px
            )
            feats_j, _ = self._get_left_features(j, frame_j)
            panels.append(LookbackPanelData(frame_idx=j, image=image_j, keypoints=feats_j.keypoints, matches=matches))

        status = "LOW-INFO" if n_obs < self.cfg.visualization.low_info_threshold else "OK"
        banner = (
            f"frame {idx} ts={frame.timestamp_ns} slot={frame.slot_label} "
            f"landmark_obs_added={n_obs} status={status}"
        )
        composite = render_frame(
            image_i, feats_i.keypoints, keypoint_status, panels, banner, n_panel_rows=self.cfg.graph.vo_lookback
        )
        self._video.write(composite)

    def close(self) -> None:
        """Release the diagnostic video file, if one is open. Safe to call multiple times."""
        if self._video is not None:
            self._video.close()

    def run(self, frames: list[FrameInfo]) -> list[FrameResult]:
        for idx in range(len(frames)):
            result = self.process_frame(idx, frames)
            if idx % 50 == 0 or idx == len(frames) - 1:
                log.info(
                    "frame %d/%d ts=%d slot=%s landmark_obs=%d",
                    idx,
                    len(frames) - 1,
                    result.frame.timestamp_ns,
                    result.frame.slot_label,
                    result.n_landmark_observations,
                )
        self.close()
        return self.results
