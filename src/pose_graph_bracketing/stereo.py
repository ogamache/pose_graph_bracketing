"""Stereo calibration loading and triangulation of metric 3D landmark observations.

The stereo pair here is already rectified in calibration (rectification_matrix
+ projection_matrix in both stereo_calibration_{left,right}.yaml share a common
focal length), so rectification only needs undistort+rectify of keypoints
(cv2.undistortPoints with R=rectification_matrix, P=projection_matrix) rather
than warping whole images. The graph's pose convention is this rectified
left-camera frame (see graph_builder.py), so all outputs here stay in that
frame -- no rotation back to the original distorted-image frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import gtsam
import numpy as np

from pose_graph_bracketing.calibration import CameraCalibration, load_stereo_calibration


@dataclass
class StereoRig:
    K_left: np.ndarray
    dist_left: np.ndarray
    K_right: np.ndarray
    dist_right: np.ndarray
    R1: np.ndarray  # rectification rotation, original left frame -> rectified left frame
    R2: np.ndarray  # rectification rotation, original right frame -> rectified right frame
    P1: np.ndarray  # rectified left projection matrix (3x4)
    P2: np.ndarray  # rectified right projection matrix (3x4)
    baseline_m: float


def load_stereo_rig(calib_dir: str | Path) -> StereoRig:
    calib_dir = Path(calib_dir)

    def load_raw(side: str):
        import yaml

        with open(calib_dir / f"stereo_calibration_{side}.yaml", "r") as f:
            return yaml.safe_load(f)

    raw_l = load_raw("left")
    raw_r = load_raw("right")

    calib_l = load_stereo_calibration(calib_dir / "stereo_calibration_left.yaml")
    calib_r = load_stereo_calibration(calib_dir / "stereo_calibration_right.yaml")

    R1 = np.array(raw_l["rectification_matrix"]["data"], dtype=np.float64).reshape(3, 3)
    R2 = np.array(raw_r["rectification_matrix"]["data"], dtype=np.float64).reshape(3, 3)
    P1 = np.array(raw_l["projection_matrix"]["data"], dtype=np.float64).reshape(3, 4)
    P2 = np.array(raw_r["projection_matrix"]["data"], dtype=np.float64).reshape(3, 4)

    fx = P1[0, 0]
    baseline_m = -P2[0, 3] / fx

    return StereoRig(
        K_left=calib_l.K,
        dist_left=calib_l.dist,
        K_right=calib_r.K,
        dist_right=calib_r.dist,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        baseline_m=baseline_m,
    )


def crop_bottom_rig(rig: StereoRig, crop_bottom_px: int) -> StereoRig:
    """Cropping rows off the bottom doesn't change K, R, P (principal point untouched)."""
    return rig


def rectify_points(pts: np.ndarray, K: np.ndarray, dist: np.ndarray, R: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Undistort + rectify a set of 2D pixel points into the rectified pixel frame defined by (R, P)."""
    if len(pts) == 0:
        return np.empty((0, 2))
    pts = pts.reshape(-1, 1, 2).astype(np.float64)
    return cv2.undistortPoints(pts, K, dist, R=R, P=P).reshape(-1, 2)


def stereo_calibration(rig: StereoRig) -> gtsam.Cal3_S2Stereo:
    """GTSAM rectified-stereo calibration built from the rig's shared P1 intrinsics + baseline."""
    fx, fy = rig.P1[0, 0], rig.P1[1, 1]
    u0, v0 = rig.P1[0, 2], rig.P1[1, 2]
    return gtsam.Cal3_S2Stereo(fx, fy, 0.0, u0, v0, rig.baseline_m)


@dataclass
class StereoObservations:
    """Per-frame stereo observations, all in the RECTIFIED left-camera pixel/3D frame.

    `points3d` is only used to seed a new landmark's initial value at first
    sighting (via triangulation); the actual optimization is driven by the
    `GenericStereoFactor3D` reprojection factors built from `stereo_points`.
    """

    indices_left: np.ndarray  # (M,) indices into the left frame's (original, unrectified) keypoint array
    stereo_points: np.ndarray  # (M, 3): [uL, uR, v], rectified pixel coordinates
    points3d: np.ndarray  # (M, 3), in the RECTIFIED left-camera frame


def compute_stereo_observations(
    kp_left: np.ndarray,
    kp_right: np.ndarray,
    match_indices_left: np.ndarray,
    match_indices_right: np.ndarray,
    rig: StereoRig,
    min_disparity_px: float = 1.0,
    max_depth_m: float = 60.0,
) -> StereoObservations:
    """Rectify + triangulate left/right stereo matches.

    Since the graph's pose convention is the rectified left-camera frame (see
    graph_builder.py), everything here stays in rectified pixel/3D space --
    no rotation back to the original distorted frame (unlike the earlier
    pairwise-PnP-VO design, which needed the original frame for K_left/dist_left).
    """
    if len(match_indices_left) == 0:
        return StereoObservations(np.empty((0,), dtype=np.int64), np.empty((0, 3)), np.empty((0, 3)))

    rect_l = rectify_points(kp_left[match_indices_left], rig.K_left, rig.dist_left, rig.R1, rig.P1)
    rect_r = rectify_points(kp_right[match_indices_right], rig.K_right, rig.dist_right, rig.R2, rig.P2)

    disparity = rect_l[:, 0] - rect_r[:, 0]
    valid = disparity > min_disparity_px

    if not np.any(valid):
        # cv2.triangulatePoints errors ("Input parameters must be matrices")
        # on an empty (2, 0) input rather than just returning an empty
        # result -- every matched pair failed the disparity gate this frame
        # (e.g. a near-featureless frame that still produced a few raw
        # LightGlue matches). Real crash observed on a full-trajectory run;
        # no valid stereo observations either way, so just return empty.
        return StereoObservations(np.empty((0,), dtype=np.int64), np.empty((0, 3)), np.empty((0, 3)))

    pts4d = cv2.triangulatePoints(rig.P1, rig.P2, rect_l[valid].T, rect_r[valid].T)
    pts3d_rect = (pts4d[:3] / pts4d[3]).T

    depth_ok = (pts3d_rect[:, 2] > 0) & (pts3d_rect[:, 2] < max_depth_m)

    kept = np.nonzero(valid)[0][depth_ok]
    stereo_points = np.stack([rect_l[kept, 0], rect_r[kept, 0], rect_l[kept, 1]], axis=1)
    return StereoObservations(match_indices_left[kept], stereo_points, pts3d_rect[depth_ok])
