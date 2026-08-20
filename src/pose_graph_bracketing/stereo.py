"""Stereo calibration loading and triangulation of metric 3D points.

The stereo pair here is already rectified in calibration (rectification_matrix
+ projection_matrix in both stereo_calibration_{left,right}.yaml share a common
focal length), so triangulation only needs undistort+rectify of keypoints
(cv2.undistortPoints with R=rectification_matrix, P=projection_matrix) rather
than warping whole images.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
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


@dataclass
class Triangulated:
    points3d: np.ndarray  # (M, 3), in the ORIGINAL (unrectified) left-camera frame
    indices_left: np.ndarray  # (M,) indices into the left frame's keypoint array


def triangulate_stereo(
    kp_left: np.ndarray,
    kp_right: np.ndarray,
    match_indices_left: np.ndarray,
    match_indices_right: np.ndarray,
    rig: StereoRig,
    min_disparity_px: float = 1.0,
    max_depth_m: float = 60.0,
) -> Triangulated:
    """Triangulate left/right stereo matches into metric 3D points.

    Points are returned in the ORIGINAL (unrectified) left-camera frame, so
    they're directly usable with the standard K_left/dist_left used elsewhere
    in the pipeline (PnP, etc.) rather than the rectified frame P1 implies.
    """
    if len(match_indices_left) == 0:
        return Triangulated(np.empty((0, 3)), np.empty((0,), dtype=np.int64))

    pts_l = kp_left[match_indices_left].reshape(-1, 1, 2).astype(np.float64)
    pts_r = kp_right[match_indices_right].reshape(-1, 1, 2).astype(np.float64)

    rect_l = cv2.undistortPoints(pts_l, rig.K_left, rig.dist_left, R=rig.R1, P=rig.P1).reshape(-1, 2)
    rect_r = cv2.undistortPoints(pts_r, rig.K_right, rig.dist_right, R=rig.R2, P=rig.P2).reshape(-1, 2)

    disparity = rect_l[:, 0] - rect_r[:, 0]
    valid = disparity > min_disparity_px

    pts4d = cv2.triangulatePoints(rig.P1, rig.P2, rect_l[valid].T, rect_r[valid].T)
    pts3d_rect = (pts4d[:3] / pts4d[3]).T  # in the RECTIFIED left-camera frame

    depth_ok = (pts3d_rect[:, 2] > 0) & (pts3d_rect[:, 2] < max_depth_m)

    pts3d_orig = (rig.R1.T @ pts3d_rect.T).T  # rectified -> original left-camera frame (pure rotation)

    kept = np.nonzero(valid)[0][depth_ok]
    return Triangulated(pts3d_orig[depth_ok], match_indices_left[kept])
