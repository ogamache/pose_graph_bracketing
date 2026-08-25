"""Incremental stereo bundle-adjustment pose graph over a bracketed-exposure sequence.

Landmarks (persistent 3D points) are tracked across frames via `LandmarkTracker`
and tied to each observing pose with a `GenericStereoFactor3D` reprojection
factor (see factors.make_stereo_observation_factor), giving the optimizer real
multi-view geometric redundancy per landmark instead of the one-shot pairwise
relative-pose estimate an ordinary VO frontend would produce.

The backend is `gtsam_unstable.IncrementalFixedLagSmoother` (real marginalization,
not the earlier windowed-ISAM2-reset workaround this repo used before a custom
GTSAM build -- with `GTSAM_BUILD_UNSTABLE=ON` -- made this available in
Python). Every pose/velocity/landmark variable is tagged with the frame
timestamp it was last touched at; the smoother automatically marginalizes out
anything older than `smoother_lag_s` behind the newest timestamp, properly
propagating the discarded variables' information into the remaining graph
instead of approximating it with a fresh re-anchoring prior.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import gtsam
import gtsam_unstable
import numpy as np

from pose_graph_bracketing.config import Config
from pose_graph_bracketing.dataset import FrameInfo
from pose_graph_bracketing.factors import (
    predict_pose,
    make_motion_prior_factor,
    make_stereo_observation_factor,
    motion_prior_noise_model,
)
from pose_graph_bracketing.features import DiskExtractor, FrameFeatures
from pose_graph_bracketing.imaging import load_preprocessed
from pose_graph_bracketing.landmarks import LandmarkTracker
from pose_graph_bracketing.matching import LightGlueMatcher
from pose_graph_bracketing.stereo import StereoObservations, StereoRig, compute_stereo_observations, stereo_calibration
from pose_graph_bracketing.visualization import (
    LiveViewer,
    LookbackPanelData,
    MatchRecord,
    TrajectoryLiveViewer,
    render_frame,
)

log = logging.getLogger(__name__)


def _pose_key(idx: int) -> int:
    return gtsam.symbol("x", idx)


def _vel_key(idx: int) -> int:
    return gtsam.symbol("v", idx)


def _landmark_key(landmark_id: int) -> int:
    return gtsam.symbol("l", landmark_id)


def _triangulate_landmark(
    current_estimate: gtsam.Values, K_stereo: gtsam.Cal3_S2Stereo, pose_frame_idx: int, stereo_point: np.ndarray
) -> np.ndarray | None:
    """Backproject a stereo observation into a landmark initial value, or None if degenerate."""
    pose_est = current_estimate.atPose3(_pose_key(pose_frame_idx))
    camera = gtsam.StereoCamera(pose_est, K_stereo)
    point = camera.backproject(gtsam.StereoPoint2(float(stereo_point[0]), float(stereo_point[1]), float(stereo_point[2])))
    if not np.all(np.isfinite(point)):
        return None
    return point


@dataclass
class FrameResult:
    frame: FrameInfo
    pose: gtsam.Pose3
    velocity: np.ndarray
    n_landmark_observations: int


class PoseGraphBuilder:
    def __init__(self, cfg: Config, rig: StereoRig):
        self.cfg = cfg
        self.rig = rig
        self.K_stereo = stereo_calibration(rig)
        self.extractor = DiskExtractor(cfg.disk, cfg.tracking)
        self.matcher = LightGlueMatcher(cfg.lightglue)
        self.landmark_tracker = LandmarkTracker()

        self.smoother = gtsam_unstable.IncrementalFixedLagSmoother(cfg.graph.smoother_lag_s, gtsam.ISAM2Params())
        self.current_estimate = gtsam.Values()

        # Frame idx+1's DISK/LightGlue stereo-observation extraction (GPU-bound,
        # ~40% of per-frame time -- see profiling) is largely independent of
        # frame idx's GTSAM smoother update (CPU-bound, ~37%), so we prefetch it
        # on a background thread while the main thread finishes idx. Both
        # PyTorch's CUDA calls and GTSAM's C++ calls release the GIL, so this
        # gets real wall-clock overlap despite Python's GIL.
        self._prefetch_pool = ThreadPoolExecutor(max_workers=1)

        self._feature_cache: dict[int, tuple[FrameFeatures, tuple[int, int]]] = {}
        self._right_feature_cache: dict[int, FrameFeatures] = {}
        self._stereo_obs_cache: dict[int, StereoObservations] = {}
        self._prefetch_futures: dict[int, "Future[StereoObservations]"] = {}
        self.results: list[FrameResult] = []
        self.zero_obs_frames: list[int] = []  # frame indices where n_landmark_observations==0 this round
        self.n_backend_resets = 0  # count of smoother resets due to a broken linear system (see process_frame)
        self._earliest_valid_pose_idx = 0  # bumped on a backend reset; older poses no longer exist in the smoother

        self._live: LiveViewer | None = None
        self._traj_live: TrajectoryLiveViewer | None = None
        if cfg.visualization.enabled:
            self._live = LiveViewer(step=cfg.visualization.step)
            self._traj_live = TrajectoryLiveViewer()
        self._traj_positions: dict[int, tuple[float, float, float]] = {}
        self._processed_indices: list[int] = []  # every frame idx processed so far, in order
        self._quit_requested = False

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

    def _get_stereo_observations(self, idx: int, frame: FrameInfo) -> StereoObservations:
        cached = self._stereo_obs_cache.get(idx)
        if cached is not None:
            return cached
        future = self._prefetch_futures.pop(idx, None)
        if future is not None:
            # A background prefetch for this frame is in flight (or done) --
            # wait on it instead of racing to compute the same thing twice.
            return future.result()
        return self._compute_stereo_observations(idx, frame)

    def _compute_stereo_observations(self, idx: int, frame: FrameInfo) -> StereoObservations:
        feats_left, shape_left = self._get_left_features(idx, frame)
        feats_right = self._get_right_features(idx, frame)
        match = self.matcher.match(feats_left, shape_left, feats_right, shape_left)
        obs = compute_stereo_observations(
            feats_left.keypoints,
            feats_right.keypoints,
            match.indices_a,
            match.indices_b,
            self.rig,
            self.cfg.stereo.min_disparity_px,
            self.cfg.stereo.max_depth_m,
        )
        self._stereo_obs_cache[idx] = obs
        return obs

    def _evict_old_features(self, current_idx: int) -> None:
        cutoff = current_idx - self.cfg.graph.vo_lookback
        for cache in (self._feature_cache, self._right_feature_cache, self._stereo_obs_cache):
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
        """Adds stereo landmark factors/values for frame `idx` against its lookback
        window, and records each touched landmark key's timestamp (this frame's,
        or the older frame `j`'s for a landmark's seeding observation) into
        `landmark_timestamps` so the caller can tell the fixed-lag smoother to
        keep it alive. Returns the number of observation factors added.

        `keypoint_status`/`visual_matches`, when given (only when the diagnostic
        video is enabled -- see visualization.py), get filled in with per-keypoint
        and per-match keep/discard bookkeeping for rendering. Left None on the
        normal/benchmark path to avoid the extra bookkeeping overhead.
        """
        frame = frames[idx]
        feats_i, shape_i = self._get_left_features(idx, frame)
        obs_i = self._get_stereo_observations(idx, frame)
        stereo_by_idx_i = {int(k): sp for k, sp in zip(obs_i.indices_left, obs_i.stereo_points)}

        if keypoint_status is not None:
            for kp_idx in range(len(feats_i.keypoints)):
                keypoint_status[kp_idx] = "has_depth_unused" if kp_idx in stereo_by_idx_i else "no_depth"

        n_obs = 0
        lookback_start = max(0, idx - self.cfg.graph.vo_lookback, self._earliest_valid_pose_idx)
        for j in range(lookback_start, idx):
            feats_j, shape_j = self._get_left_features(j, frames[j])
            obs_j = self._get_stereo_observations(j, frames[j])
            stereo_by_idx_j = {int(k): sp for k, sp in zip(obs_j.indices_left, obs_j.stereo_points)}

            match = self.matcher.match(feats_j, shape_j, feats_i, shape_i)
            if match.indices_a.shape[0] == 0:
                continue

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
                stereo_point_i = stereo_by_idx_i.get(idx_b)

                if landmark_id is None:
                    # Only seed a brand-new landmark if BOTH endpoints have a valid
                    # triangulated depth, so every new landmark starts with two
                    # observations (well-conditioned) rather than a single one that
                    # might never get a second constraint (a degenerate landmark
                    # with too little support can make the linear system indeterminate).
                    stereo_point_j = stereo_by_idx_j.get(idx_a)
                    if stereo_point_j is None or stereo_point_i is None:
                        discard("no_depth_j" if stereo_point_j is None else "no_depth_i")
                        continue
                    point_init = _triangulate_landmark(self.current_estimate, self.K_stereo, j, stereo_point_j)
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
                        make_stereo_observation_factor(
                            _pose_key(j),
                            _landmark_key(landmark_id),
                            stereo_point_j,
                            self.K_stereo,
                            self.cfg.stereo.pixel_sigma,
                            self.cfg.stereo.huber_k,
                        )
                    )
                    n_obs += 1

                if stereo_point_i is None:
                    discard("no_depth_i")
                    continue
                self.landmark_tracker.get_or_create(idx, idx_b, existing_landmark_id=landmark_id)
                landmark_timestamps[landmark_id] = frame.timestamp_s
                graph.add(
                    make_stereo_observation_factor(
                        _pose_key(idx),
                        _landmark_key(landmark_id),
                        stereo_point_i,
                        self.K_stereo,
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

        # Ensure features/stereo observations for this frame (and its lookback window) are ready.
        self._get_stereo_observations(idx, frame)

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
            zero_motion = self.cfg.motion_prior.zero_motion
            predicted_pose = prev_pose if zero_motion else predict_pose(prev_pose, prev_vel, dt)

            initial.insert(X_i, predicted_pose)
            initial.insert(V_i, prev_vel)

            mp_noise = motion_prior_noise_model(
                self.cfg.motion_prior.rotation_sigma,
                self.cfg.motion_prior.translation_sigma,
                self.cfg.motion_prior.angular_velocity_rw_sigma,
                self.cfg.motion_prior.linear_velocity_rw_sigma,
                dt,
                zero_motion=zero_motion,
                zero_motion_rotation_sigma=self.cfg.motion_prior.zero_motion_rotation_sigma,
                zero_motion_translation_sigma=self.cfg.motion_prior.zero_motion_translation_sigma,
            )
            graph.add(make_motion_prior_factor(X_prev, V_prev, X_i, V_i, dt, mp_noise, zero_motion=zero_motion))

        landmark_timestamps: dict[int, float] = {}
        keypoint_status: dict[int, str] | None = {} if self._live is not None else None
        visual_matches: dict[int, list[MatchRecord]] | None = {} if self._live is not None else None
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
        except (RuntimeError, IndexError) as e:
            # The incremental smoother can occasionally end up with a
            # numerically broken linear system (e.g. a landmark whose
            # effective information collapsed right as it aged out of the
            # fixed-lag window, or a pose left almost entirely unconstrained
            # by vision on a frame with very little real support). This
            # doesn't always surface as GTSAM's own IndeterminantLinearSystemException
            # (a RuntimeError) at update() -- sometimes update() "succeeds"
            # but leaves the Bayes tree incomplete, and calculateEstimate()
            # then raises a plain IndexError instead. Rather than crash the
            # whole run, reset the backend: build a COMPLETELY FRESH smoother
            # (not a retry against the same one -- update() is not atomic, a
            # failed call can leave it corrupted, so retrying in place just
            # fails differently) re-anchored at this frame's own
            # (already-computed) initial pose/velocity guess, and retry with
            # no landmark factors, so the trajectory keeps going. Landmarks
            # lost this way simply get re-seeded on a later lookback pair.
            log.warning("Smoother update failed at frame %d (%s: %s) -- resetting backend and retrying without landmarks", idx, type(e).__name__, e)
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

        self._processed_indices.append(idx)
        if self._traj_live is not None:
            # Walk backward through processed frames while the smoother still
            # holds that pose key (i.e. it hasn't been marginalized out of the
            # fixed-lag window yet), overwriting each with its latest
            # re-optimized value -- not just appending the newest pose -- so
            # poses inside the lag window get corrected on screen as later
            # observations refine them.
            for k in reversed(self._processed_indices):
                if not self.current_estimate.exists(_pose_key(k)):
                    break
                t = self.current_estimate.atPose3(_pose_key(k)).translation()
                self._traj_positions[k] = (float(t[0]), float(t[1]), float(t[2]))
            self._traj_live.update([self._traj_positions[k] for k in sorted(self._traj_positions)])

        if self._live is not None:
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
        self._quit_requested = self._live.show(composite)

    def close(self) -> None:
        """Release the live windows, if open. Safe to call multiple times."""
        if self._live is not None:
            self._live.close()
        if self._traj_live is not None:
            self._traj_live.close()

    def run(self, frames: list[FrameInfo]) -> list[FrameResult]:
        # Prefetch depth 2: profiling showed a single frame's DISK/LightGlue
        # extraction (~300ms) takes about as long as one loop iteration's
        # main-thread work, so a depth-1 prefetch never finishes in time and
        # the main thread ends up redundantly recomputing it anyway. Depth 2
        # gives the background thread roughly two iterations' head start.
        prefetch_depth = 2
        if frames:
            self._get_stereo_observations(0, frames[0])
        for k in range(1, min(prefetch_depth, len(frames))):
            self._prefetch_futures[k] = self._prefetch_pool.submit(self._compute_stereo_observations, k, frames[k])

        for idx in range(len(frames)):
            next_idx = idx + prefetch_depth
            if next_idx < len(frames):
                self._prefetch_futures[next_idx] = self._prefetch_pool.submit(
                    self._compute_stereo_observations, next_idx, frames[next_idx]
                )
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
            if self._quit_requested:
                log.info("Live view quit requested at frame %d/%d -- stopping early", idx, len(frames) - 1)
                break
        self.close()
        if self.n_backend_resets:
            log.info("Backend resets: %d (see warnings above for the triggering frames)", self.n_backend_resets)
        return self.results
