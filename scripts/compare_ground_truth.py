#!/usr/bin/env python3
"""Compare an estimated TUM trajectory against the lidar ground truth.

Ground truth: use `<trajectory_dir>/offline/<run_timestamp>/traj.tum` — an
offline ICP lidar-mapping run (topic `/mapping/icp_odom_offline`) that covers
the *full* recorded sequence with smooth, physically-plausible motion.

Do NOT use `<trajectory_dir>/f_*.txt` or `kf_*.txt` — despite looking like the
obvious ground-truth candidates, both were verified to be an incomplete/glitchy
export (covering as little as 15% of a sequence, with stretches of a frozen
repeated pose from lidar tracking dropouts) from a different, less reliable
pipeline. Also do NOT use `<trajectory_dir>/.cuvslam_edex/trajectory_tum.txt`
(NVIDIA cuVSLAM stereo-VO output) — it covers the full sequence but diverges
catastrophically (400+ m single-frame jumps) partway through on this
bracketed-exposure data.

All of the above are TUM format with the timestamp expressed in raw
nanoseconds (not seconds) as the first column's literal value.

Since our estimate is monocular (no metric scale) and the lidar and camera have
an unknown-but-fixed rigid extrinsic offset, we align estimate -> ground-truth
with a similarity transform (rotation + scale + translation, Umeyama's method)
before computing errors. This validates trajectory *shape*, not absolute scale.
Note: Umeyama's rotation fit is poorly constrained for near-collinear (e.g.
out-and-back corridor) trajectories -- verified this isn't hiding a coordinate-
convention bug by (a) testing the stated lidar/camera axis convention as a
fixed rotation, (b) brute-forcing all 24 valid axis-permutation rotations, and
(c) deriving a fixed extrinsic from orientation data instead of position data
-- none beat the free Umeyama fit, so it's used as-is.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


def load_tum(path: str | Path, ns_timestamps: bool = False) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    ts = data[:, 0]
    if ns_timestamps:
        ts = ts * 1e-9
    xyz = data[:, 1:4]
    return ts, xyz


def load_tum_full(path: str | Path, ns_timestamps: bool = False) -> tuple[np.ndarray, np.ndarray, Rotation]:
    """Like load_tum but also returns orientation (quaternion columns qx qy qz qw)."""
    data = np.loadtxt(path)
    ts = data[:, 0]
    if ns_timestamps:
        ts = ts * 1e-9
    xyz = data[:, 1:4]
    rot = Rotation.from_quat(data[:, 4:8])  # scipy expects [x, y, z, w], matches TUM column order
    return ts, xyz, rot


def associate(ts_a: np.ndarray, ts_b: np.ndarray, max_dt: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbor association of ts_a onto ts_b. Returns index arrays."""
    idx_b = np.searchsorted(ts_b, ts_a)
    idx_b = np.clip(idx_b, 1, len(ts_b) - 1)
    left = idx_b - 1
    right = idx_b
    use_left = np.abs(ts_a - ts_b[left]) <= np.abs(ts_a - ts_b[right])
    best_b = np.where(use_left, left, right)
    dt = np.abs(ts_a - ts_b[best_b])
    keep = dt <= max_dt
    idx_a = np.nonzero(keep)[0]
    return idx_a, best_b[keep]


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares similarity transform mapping src -> dst: dst ~ s * R @ src + t."""
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt

    var_src = (src_c**2).sum() / len(src)
    scale = (D * np.diag(S)).sum() / var_src

    t = mu_dst - scale * R @ mu_src
    return R, t, scale


def compute_rpe(
    idx_est: np.ndarray,
    idx_gt: np.ndarray,
    xyz_est: np.ndarray,
    rot_est: Rotation,
    xyz_gt: np.ndarray,
    rot_gt: Rotation,
    scale: float,
    delta_s: float,
    ts_est: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Standard (TUM-style) Relative Pose Error at a fixed time delta.

    For each synced pair (idx_est[k], idx_gt[k]) and its partner ~delta_s later
    in the synced sequence, compares the relative motion (translation +
    rotation) each trajectory made over that interval. Unlike ATE, RPE is a
    *local drift* metric: rotation error is invariant to any fixed camera/lidar
    extrinsic rotation offset (it only changes the relative-rotation axis, not
    its angle), and translation error only needs the estimate's translations
    scaled by the Sim(3) alignment `scale` (a fixed global rotation/translation
    offset cancels out of a relative-motion computation).

    Returns (translational_errors [m], rotational_errors [deg]), one entry per
    valid pair.
    """
    dt_est = np.median(np.diff(ts_est[idx_est])) if len(idx_est) > 1 else 0.03
    step = max(1, round(delta_s / max(dt_est, 1e-6)))

    trans_errors = []
    rot_errors = []
    for k in range(len(idx_est) - step):
        i_est, j_est = idx_est[k], idx_est[k + step]
        i_gt, j_gt = idx_gt[k], idx_gt[k + step]

        R_est_i, R_est_j = rot_est[i_est], rot_est[j_est]
        t_rel_est = R_est_i.inv().apply(xyz_est[j_est] - xyz_est[i_est]) * scale
        R_rel_est = R_est_i.inv() * R_est_j

        R_gt_i, R_gt_j = rot_gt[i_gt], rot_gt[j_gt]
        t_rel_gt = R_gt_i.inv().apply(xyz_gt[j_gt] - xyz_gt[i_gt])
        R_rel_gt = R_gt_i.inv() * R_gt_j

        trans_err = np.linalg.norm(t_rel_est - t_rel_gt)
        rot_err_deg = (R_rel_gt.inv() * R_rel_est).magnitude() * 180.0 / np.pi

        trans_errors.append(trans_err)
        rot_errors.append(rot_err_deg)

    return np.array(trans_errors), np.array(rot_errors)


def compute_rpe_distance(
    idx_est: np.ndarray,
    idx_gt: np.ndarray,
    xyz_est: np.ndarray,
    rot_est: Rotation,
    xyz_gt: np.ndarray,
    rot_gt: Rotation,
    scale: float,
    window_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """RPE over a fixed *distance* window (KITTI-style), instead of a fixed time delta.

    For every synced start index k, finds the synced end index whose ground-truth
    path length since k is closest to `window_m`, then computes the same
    translation/rotation relative-motion error as `compute_rpe`. This is more
    informative than a single time-based delta because drift usually scales
    with distance traveled, not elapsed time -- useful to see e.g. "error at
    1m of travel" vs "error at 20m of travel" independent of how fast the
    trajectory happened to move.

    Returns (translational_errors [m], rotational_errors [deg]).
    """
    dst = xyz_gt[idx_gt]
    seg_len = np.linalg.norm(np.diff(dst, axis=0), axis=1)
    cumdist = np.concatenate([[0.0], np.cumsum(seg_len)])
    n = len(cumdist)

    trans_errors = []
    rot_errors = []
    for k in range(n):
        target = cumdist[k] + window_m
        if target > cumdist[-1]:
            break
        j = int(np.searchsorted(cumdist, target, side="left"))
        if j <= k or j >= n:
            continue

        i_est, j_est = idx_est[k], idx_est[j]
        i_gt, j_gt = idx_gt[k], idx_gt[j]

        R_est_i, R_est_j = rot_est[i_est], rot_est[j_est]
        t_rel_est = R_est_i.inv().apply(xyz_est[j_est] - xyz_est[i_est]) * scale
        R_rel_est = R_est_i.inv() * R_est_j

        R_gt_i, R_gt_j = rot_gt[i_gt], rot_gt[j_gt]
        t_rel_gt = R_gt_i.inv().apply(xyz_gt[j_gt] - xyz_gt[i_gt])
        R_rel_gt = R_gt_i.inv() * R_gt_j

        trans_errors.append(np.linalg.norm(t_rel_est - t_rel_gt))
        rot_errors.append((R_rel_gt.inv() * R_rel_est).magnitude() * 180.0 / np.pi)

    return np.array(trans_errors), np.array(rot_errors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate", required=True, help="Our output TUM trajectory (seconds timestamps)")
    parser.add_argument("--ground-truth", required=True, help="Lidar f_/kf_ TUM file (raw-ns timestamps)")
    parser.add_argument("--out-plot", default=None, help="Optional path to save an XY/XZ comparison plot")
    parser.add_argument("--max-dt", type=float, default=0.05, help="Max association time gap in seconds")
    parser.add_argument("--rpe-delta", type=float, default=1.0, help="RPE time interval in seconds")
    parser.add_argument(
        "--rpe-windows",
        default="1,5,10,20,40",
        help="Comma-separated RPE distance windows in meters (KITTI-style), empty string to skip",
    )
    args = parser.parse_args()

    ts_est, xyz_est, rot_est = load_tum_full(args.estimate, ns_timestamps=False)
    ts_gt, xyz_gt, rot_gt = load_tum_full(args.ground_truth, ns_timestamps=True)

    idx_est, idx_gt = associate(ts_est, ts_gt, max_dt=args.max_dt)
    print(f"Associated {len(idx_est)}/{len(ts_est)} estimate poses to ground truth (max_dt={args.max_dt}s)")
    if len(idx_est) < 10:
        raise SystemExit("Too few associations to align/evaluate.")

    src = xyz_est[idx_est]
    dst = xyz_gt[idx_gt]
    R, t, scale = umeyama(src, dst)
    print(f"Similarity alignment: scale={scale:.4f}")

    aligned_est_full = (scale * (R @ xyz_est.T).T) + t
    aligned_src = (scale * (R @ src.T).T) + t

    err = np.linalg.norm(aligned_src - dst, axis=1)
    print(f"ATE after Sim(3) alignment: RMSE={np.sqrt((err**2).mean()):.4f} m, "
          f"mean={err.mean():.4f} m, median={np.median(err):.4f} m, max={err.max():.4f} m")

    gt_path_len = np.sum(np.linalg.norm(np.diff(dst, axis=0), axis=1))
    est_path_len = np.sum(np.linalg.norm(np.diff(aligned_src, axis=0), axis=1))
    print(f"Ground-truth path length (associated span): {gt_path_len:.2f} m")
    print(f"Estimate path length (aligned, associated span): {est_path_len:.2f} m")

    trans_errs, rot_errs = compute_rpe(
        idx_est, idx_gt, xyz_est, rot_est, xyz_gt, rot_gt, scale, args.rpe_delta, ts_est
    )
    if len(trans_errs) == 0:
        print(f"RPE: not enough synced samples for a {args.rpe_delta}s interval.")
    else:
        trans_rmse = np.sqrt((trans_errs**2).mean())
        rot_rmse = np.sqrt((rot_errs**2).mean())
        print(
            f"RPE (delta={args.rpe_delta}s, n={len(trans_errs)}): "
            f"trans RMSE={trans_rmse:.4f} m (mean={trans_errs.mean():.4f}, median={np.median(trans_errs):.4f}, "
            f"max={trans_errs.max():.4f}), "
            f"rot RMSE={rot_rmse:.3f} deg (mean={rot_errs.mean():.3f}, median={np.median(rot_errs):.3f}, "
            f"max={rot_errs.max():.3f})"
        )

    if args.rpe_windows.strip():
        windows = [float(w) for w in args.rpe_windows.split(",")]
        print("RPE by distance window (KITTI-style):")
        for w in windows:
            trans_errs_w, rot_errs_w = compute_rpe_distance(
                idx_est, idx_gt, xyz_est, rot_est, xyz_gt, rot_gt, scale, w
            )
            if len(trans_errs_w) == 0:
                print(f"  {w:>6.1f} m: not enough associated span for this window.")
                continue
            trans_rmse_w = np.sqrt((trans_errs_w**2).mean())
            rot_rmse_w = np.sqrt((rot_errs_w**2).mean())
            pct = 100.0 * trans_rmse_w / w
            print(
                f"  {w:>6.1f} m (n={len(trans_errs_w):4d}): "
                f"trans RMSE={trans_rmse_w:.4f} m ({pct:.2f}%), "
                f"rot RMSE={rot_rmse_w:.3f} deg ({rot_rmse_w / w:.3f} deg/m)"
            )

    if args.out_plot:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].plot(xyz_gt[:, 0], xyz_gt[:, 1], label="ground truth (lidar)", color="tab:blue")
        axs[0].plot(aligned_est_full[:, 0], aligned_est_full[:, 1], label="estimate (aligned)", color="tab:orange")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")
        axs[0].set_title("XY")
        axs[0].legend()
        axs[0].axis("equal")

        axs[1].plot(xyz_gt[:, 0], xyz_gt[:, 2], label="ground truth (lidar)", color="tab:blue")
        axs[1].plot(aligned_est_full[:, 0], aligned_est_full[:, 2], label="estimate (aligned)", color="tab:orange")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("z")
        axs[1].set_title("XZ")
        axs[1].legend()
        axs[1].axis("equal")

        plt.tight_layout()
        plt.savefig(args.out_plot, dpi=120)
        print(f"Saved comparison plot to {args.out_plot}")


if __name__ == "__main__":
    main()
