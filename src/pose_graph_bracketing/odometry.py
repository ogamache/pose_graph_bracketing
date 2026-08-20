"""Essential-matrix visual odometry with a reliability gate.

Estimates the relative pose (R, t) that takes points expressed in camera i
into camera j (i.e. p_j ~ R @ p_i + t, t up to unknown scale), from a set of
2D-2D correspondences, and flags the estimate unreliable when there is too
little geometric evidence to trust it (too few inliers, or inliers clustered
in a small part of the frame).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from pose_graph_bracketing.config import OdometryConfig, StereoConfig


@dataclass
class PoseEstimate:
    R: np.ndarray  # (3, 3) rotation, camera_i -> camera_j
    t: np.ndarray  # (3,) unit-norm translation direction, camera_i -> camera_j
    n_inliers: int
    n_matches: int
    inlier_mask: np.ndarray  # (n_matches,) bool
    bbox_coverage: float
    reliable: bool
    warnings: list[str] = field(default_factory=list)


def _recover_pose_from_essential(
    E: np.ndarray,
    undist_i: np.ndarray,
    undist_j: np.ndarray,
    K: np.ndarray,
    inlier_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Disambiguate the 4 (R, t) hypotheses from an essential matrix via cheirality.

    Equivalent in purpose to cv2.recoverPose(E, points1, points2, cameraMatrix,
    mask=...), reimplemented manually: this cv2 build's recoverPose(E, ...)
    overload was found to always return 0 inliers on real data despite valid
    hypotheses existing (verified against a hand-rolled cheirality check), so
    we do the essential-matrix decomposition + triangulation cheirality test
    ourselves instead of trusting that overload.

    Returns (R, t, per_point_pass_mask) for the best hypothesis, where R, t
    map points from camera i into camera j (p_j ~ R @ p_i + t).
    """
    R1, R2, t = cv2.decomposeEssentialMat(E)
    t = t.reshape(3, 1)

    pts_i = undist_i[inlier_mask]
    pts_j = undist_j[inlier_mask]
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])

    best = None
    for R_cand in (R1, R2):
        for sign in (1.0, -1.0):
            t_cand = t * sign
            P2 = K @ np.hstack([R_cand, t_cand])
            pts4d = cv2.triangulatePoints(P1, P2, pts_i.T, pts_j.T)
            pts3d = (pts4d[:3] / pts4d[3]).T
            depth_i = pts3d[:, 2]
            depth_j = (R_cand @ pts3d.T + t_cand).T[:, 2]
            pass_mask = (depth_i > 0) & (depth_j > 0)
            n_good = int(pass_mask.sum())
            if best is None or n_good > best[0]:
                best = (n_good, R_cand, t_cand.reshape(-1), pass_mask)

    _, R_best, t_best, pass_mask = best
    full_pass = np.zeros(len(undist_i), dtype=bool)
    full_pass[np.nonzero(inlier_mask)[0]] = pass_mask
    return R_best, t_best, full_pass


def estimate_pose(
    pts_i: np.ndarray,
    pts_j: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray | None,
    cfg: OdometryConfig,
    img_shape: tuple[int, int] | None = None,
) -> PoseEstimate | None:
    """Estimate relative pose from camera i to camera j given 2D correspondences.

    Returns None if there aren't even enough correspondences to attempt an
    estimate (`< cfg.min_matches`), or if OpenCV fails to produce an essential
    matrix / recover a pose. Otherwise returns a PoseEstimate whose `reliable`
    flag reports whether the frontend trusts this estimate enough to be used
    as an odometry factor.
    """
    n_matches = len(pts_i)
    if n_matches < cfg.min_matches:
        return None

    pts_i = pts_i.reshape(-1, 1, 2).astype(np.float64)
    pts_j = pts_j.reshape(-1, 1, 2).astype(np.float64)

    undist_i = cv2.undistortPoints(pts_i, K, dist, P=K).reshape(-1, 2)
    undist_j = cv2.undistortPoints(pts_j, K, dist, P=K).reshape(-1, 2)

    E, mask = cv2.findEssentialMat(
        undist_i,
        undist_j,
        cameraMatrix=K,
        method=cv2.RANSAC,
        prob=cfg.confidence,
        threshold=cfg.ransac_threshold_px,
    )
    if E is None or E.shape != (3, 3):
        return None

    essential_inlier_mask = mask.reshape(-1).astype(bool)
    R, t, inlier_mask = _recover_pose_from_essential(E, undist_i, undist_j, K, essential_inlier_mask)
    n_inliers = int(inlier_mask.sum())
    if n_inliers == 0:
        return None

    warnings: list[str] = []
    if n_inliers < cfg.min_reliable_inliers:
        warnings.append(f"low inlier count ({n_inliers})")

    bbox_coverage = 1.0
    if img_shape is not None:
        h, w = img_shape[:2]
        inlier_pts = undist_j[inlier_mask]
        if inlier_pts.shape[0] > 0:
            x_range = inlier_pts[:, 0].max() - inlier_pts[:, 0].min()
            y_range = inlier_pts[:, 1].max() - inlier_pts[:, 1].min()
            bbox_coverage = (x_range * y_range) / (w * h)
        else:
            bbox_coverage = 0.0
        if bbox_coverage < cfg.min_reliable_bbox_coverage:
            warnings.append(f"points clustered ({bbox_coverage:.0%} of frame)")

    return PoseEstimate(
        R=R,
        t=t.reshape(-1) / (np.linalg.norm(t) + 1e-12),
        n_inliers=int(n_inliers),
        n_matches=n_matches,
        inlier_mask=inlier_mask,
        bbox_coverage=float(bbox_coverage),
        reliable=len(warnings) == 0,
        warnings=warnings,
    )


def estimate_pose_pnp(
    pts3d_j: np.ndarray,
    pts2d_i: np.ndarray,
    K_i: np.ndarray,
    dist_i: np.ndarray | None,
    cfg: StereoConfig,
    img_shape: tuple[int, int] | None = None,
) -> PoseEstimate | None:
    """Metric relative pose from camera j to camera i via stereo-triangulated 3D-2D PnP.

    `pts3d_j` are 3D points triangulated in frame j's own (left) camera frame
    (see stereo.triangulate_stereo); `pts2d_i` are their matched 2D pixel
    observations in frame i's left image. Unlike `estimate_pose` (essential
    matrix), this recovers a genuinely metric translation directly -- no scale
    ambiguity, since the 3D points already have real-world scale from the
    known stereo baseline.

    Returns None if there aren't enough correspondences, or PnP fails.
    Otherwise returns a PoseEstimate with `R`, `t` mapping frame j's points
    into frame i's camera coordinates (p_i ~ R @ p_j + t, `t` in meters), plus
    the same inlier-count/bbox-coverage reliability gate as `estimate_pose`.
    """
    n_matches = len(pts3d_j)
    if n_matches < cfg.min_pnp_points:
        return None

    obj_pts = pts3d_j.reshape(-1, 1, 3).astype(np.float64)
    img_pts = pts2d_i.reshape(-1, 1, 2).astype(np.float64)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_pts,
        img_pts,
        K_i,
        dist_i,
        reprojectionError=cfg.ransac_reprojection_error_px,
        confidence=cfg.confidence,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) == 0:
        return None

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(-1)

    inlier_idx = inliers.reshape(-1)
    inlier_mask = np.zeros(n_matches, dtype=bool)
    inlier_mask[inlier_idx] = True
    n_inliers = int(inlier_mask.sum())

    warnings: list[str] = []
    if n_inliers < cfg.min_reliable_inliers:
        warnings.append(f"low inlier count ({n_inliers})")

    bbox_coverage = 1.0
    if img_shape is not None:
        h, w = img_shape[:2]
        inlier_pts = img_pts.reshape(-1, 2)[inlier_mask]
        if inlier_pts.shape[0] > 0:
            x_range = inlier_pts[:, 0].max() - inlier_pts[:, 0].min()
            y_range = inlier_pts[:, 1].max() - inlier_pts[:, 1].min()
            bbox_coverage = (x_range * y_range) / (w * h)
        else:
            bbox_coverage = 0.0
        if bbox_coverage < cfg.min_reliable_bbox_coverage:
            warnings.append(f"points clustered ({bbox_coverage:.0%} of frame)")

    return PoseEstimate(
        R=R,
        t=t,
        n_inliers=n_inliers,
        n_matches=n_matches,
        inlier_mask=inlier_mask,
        bbox_coverage=float(bbox_coverage),
        reliable=len(warnings) == 0,
        warnings=warnings,
    )
