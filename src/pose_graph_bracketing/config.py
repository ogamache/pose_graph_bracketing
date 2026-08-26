"""YAML configuration loading into typed dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class DatasetConfig:
    bayer_pattern: str = "RGGB"
    side: str = "left"


@dataclass
class PreprocessingConfig:
    crop_bottom_px: int = 175
    # true (default): local-contrast normalization, applied per-frame
    # before feature extraction -- confirmed fix for bracketed-exposure
    # oscillation (frame-to-frame instability from exposure non-uniformity
    # between SAE/MAE/LAE), see docs/cycle_bias_findings.md. A real
    # gain/exposure-invariant radiance-domain alternative (radiance_enabled
    # below) was tried as a more principled replacement and found clearly
    # worse (both crf and linear modes) -- CLAHE preserves local spatial
    # contrast structure that DISK's learned features rely on, while the
    # radiance domain's percentile-normalized log compression throws much
    # of that away. Kept as the shipped default; radiance_enabled kept
    # available, off, for further tuning if revisited.
    clahe_enabled: bool = True
    clahe_clip_limit: float = 10.0
    clahe_tile_grid_size: int = 8
    gaussian_blur_enabled: bool = False
    gaussian_blur_ksize: int = 5
    gaussian_blur_sigma: float = 0.0
    radiance_enabled: bool = False
    radiance_mode: str = "crf"  # "crf" | "linear" -- see RadianceConfig
    radiance_crf_path: str = "/home/alien/Documents/research/third_article/radiance_bracketing_tracking/crf_output/crf.npz"
    # true (default): loads each frame's image up front, computes its mean
    # pixel brightness, and drops the frame entirely if it's below
    # drop_low_info_min_brightness or above drop_low_info_max_brightness --
    # near-featureless (crushed/saturated), see
    # dataset.drop_low_information_frames. false: every frame is processed.
    drop_low_info_frames: bool = True
    drop_low_info_min_brightness: float = 10.0
    drop_low_info_max_brightness: float = 245.0
    # false (default): tested on b_0fps region3 (raw DISK+LightGlue match
    # count against the graph.vo_lookback nearest original-sequence frames,
    # drop if below drop_low_match_min_matches) and found a net regression
    # (ATE 0.28m->0.47m, scale 1.303->1.535) -- raw 2D keypoint match count
    # doesn't reflect whether a frame is a useful depth-validated-stereo
    # graph anchor, so this drops frames that still mattered. true: runs
    # the pass anyway -- see dataset.drop_low_match_frames.
    drop_low_match_frames: bool = False
    drop_low_match_min_matches: int = 20


@dataclass
class TrackingConfig:
    max_corners: int = 1000
    grid_rows: int = 2
    grid_cols: int = 2
    hysteresis_margin: float = 0.15
    hysteresis_radius: float = 5.0
    # true (default): skip matching entirely between a LAE frame and a SAE
    # frame (the two most exposure-dissimilar slots) -- they're never each
    # other's only path to a shared landmark since a MAE frame always sits
    # between them in the bracket cycle, so blocking the direct match
    # doesn't lose connectivity, only the least reliable correspondences.
    # false: every frame pair within vo_lookback is matched regardless of
    # exposure slot. See docs/cycle_bias_findings.md.
    block_lae_sae_matches: bool = True


@dataclass
class DiskConfig:
    device: str = "cuda"
    checkpoint: str = "depth"
    window_size: int = 5
    score_threshold: float = 0.0
    match_max_distance: float = 0.5
    kp_oversample_factor: int = 3


@dataclass
class LightGlueConfig:
    device: str = "cuda"
    feature_name: str = "disk"
    min_confidence: float = 0.9


@dataclass
class StereoConfig:
    min_disparity_px: float = 1.0
    max_depth_m: float = 60.0
    pixel_sigma: float = 1.0    # rectified-pixel reprojection noise for GenericStereoFactor3D
    huber_k: float = 1.345      # standard Huber constant (~95% efficiency under Gaussian noise)
    landmark_prior_sigma: float = 3.0  # m, weak prior anchoring each new landmark near its initial triangulation
    # Mono only: a new landmark's seeding baseline (essential-matrix relative
    # pose between the two frames that first triangulate it) is scale-free by
    # construction -- assumed as assumed_speed_mps * elapsed_time instead of
    # left at an arbitrary unit norm, so every landmark shares a consistent
    # scale reference rather than each being anchored (via landmark_prior_sigma
    # above) at whatever scale its own seeding pair happened to imply.
    assumed_speed_mps: float = 2.5
    # false (default): every stereo observation factor uses the flat
    # pixel_sigma above regardless of the landmark's depth. true: scales
    # pixel_sigma up linearly with the observation's own disparity-implied
    # depth past depth_scaled_noise_reference_m -- an empirical
    # down-weighting of far/less-reliable observations in the bundle
    # adjustment -- confirmed fix for a scale bias found in bracketed
    # sequences, see docs/cycle_bias_findings.md and
    # factors.make_stereo_observation_factor.
    depth_scaled_noise: bool = False
    depth_scaled_noise_reference_m: float = 5.0  # m, below this depth pixel_sigma is unscaled
    depth_scaled_noise_power: float = 1.0  # 1.0=linear, 2.0=matches metric-uncertainty growth rate (depth^2/(fx*baseline))


@dataclass
class MotionPriorConfig:
    # Loosened from 0.05 -- swept 0.5/1.0/2.0 on b_0fps region3 (current
    # defaults otherwise), 1.0 was best (ATE 0.260m/scale 1.268 vs. tight
    # 0.05's 0.311m/1.293 and zero_motion's 0.280m/1.303). See
    # docs/cycle_bias_findings.md.
    rotation_sigma: float = 1.0
    translation_sigma: float = 1.0
    angular_velocity_rw_sigma: float = 2.0
    linear_velocity_rw_sigma: float = 5.0
    initial_velocity_prior_sigma: float = 1.0
    initial_pose_prior_sigma: float = 1.0e-3
    # true (default): every frame's prediction is Identity (assume no
    # motion happened), uniformly, using the loose flat zero_motion_*
    # sigmas below. Constant-velocity (rotation_sigma/translation_sigma
    # above) beat identity on b_0fps region3 in isolation, but identity is
    # the one that doesn't catastrophically diverge on datasets with real
    # extended blind streaks. See factors.make_motion_prior_factor's
    # zero_motion docstring and docs/cycle_bias_findings.md. false:
    # constant-velocity extrapolation instead.
    zero_motion: bool = True
    # Only used when zero_motion is true. Deliberately loose, flat
    # (not dt-scaled) sigmas -- rotation_sigma/translation_sigma above are
    # calibrated for deviation from a good constant-velocity prediction,
    # not for a zero-motion prediction that's systematically wrong by
    # roughly the real per-step displacement (a very confident, wrong
    # prior fighting real motion if reused). These should barely constrain
    # anything -- just enough to keep the pose full-rank -- see
    # factors.motion_prior_noise_model's docstring.
    zero_motion_rotation_sigma: float = 3.14159  # rad, flat (not scaled by sqrt(dt))
    zero_motion_translation_sigma: float = 10.0  # m, flat (not scaled by sqrt(dt))


@dataclass
class GraphConfig:
    vo_lookback: int = 4
    # gtsam_unstable.IncrementalFixedLagSmoother's window, in seconds: any
    # pose/velocity/landmark variable not re-touched within this many seconds
    # of the newest timestamp gets properly marginalized (not just dropped).
    # Must comfortably exceed vo_lookback frames' worth of real elapsed time
    # for the slowest-fps dataset in use, or active variables could be
    # marginalized out from under a still-in-window frame.
    #
    # (An earlier experiment tried decoupling mono's landmark-matching window
    # from vo_lookback and tripling it, on the theory that mono's stricter
    # seeding gate was starving landmarks of multi-view redundancy. It didn't
    # help -- ATE got slightly worse despite ~2x more observations/frame, and
    # path length stayed wildly inflated even after Sim(3) alignment -- so
    # mono's real limiting factor isn't observation count; reverted.)
    smoother_lag_s: float = 1.0
    # true (default): after the incremental run, also run a full batch
    # (non-fixed-lag) bundle adjustment over every factor added
    # (PoseGraphBuilder.global_bundle_adjust) and write it to
    # <out>_global_ba.tum -- meaningfully improves the typical-frame
    # accuracy (median ATE, scale bias) over the incremental result, at
    # the cost of a known artifact on a weakly-observed early frame (see
    # docs/cycle_bias_findings.md). Stereo mode only. false: skip it.
    global_bundle_adjust: bool = True


@dataclass
class VisualizationConfig:
    enabled: bool = False  # set by run_trajectory.py --visualize; shows a live matches window + a live trajectory window
    step: bool = False  # set by run_trajectory.py --step; matches window waits for a keypress before each next frame
    low_info_threshold: int = 20  # n_landmark_observations below this -> frame flagged LOW-INFO


_VALID_MODES = {"stereo", "mono"}


@dataclass
class Config:
    mode: str
    dataset: DatasetConfig
    preprocessing: PreprocessingConfig
    tracking: TrackingConfig
    disk: DiskConfig
    lightglue: LightGlueConfig
    stereo: StereoConfig
    motion_prior: MotionPriorConfig
    graph: GraphConfig
    visualization: VisualizationConfig

    @staticmethod
    def load(path: str | Path) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        mode = raw.get("mode", "stereo")
        if mode not in _VALID_MODES:
            raise ValueError(f"config `mode` must be one of {_VALID_MODES}, got {mode!r}")
        return Config(
            mode=mode,
            dataset=DatasetConfig(**raw.get("dataset", {})),
            preprocessing=PreprocessingConfig(**raw.get("preprocessing", {})),
            tracking=TrackingConfig(**raw.get("tracking", {})),
            disk=DiskConfig(**raw.get("disk", {})),
            lightglue=LightGlueConfig(**raw.get("lightglue", {})),
            stereo=StereoConfig(**raw.get("stereo", {})),
            motion_prior=MotionPriorConfig(**raw.get("motion_prior", {})),
            graph=GraphConfig(**raw.get("graph", {})),
            visualization=VisualizationConfig(**raw.get("visualization", {})),
        )
