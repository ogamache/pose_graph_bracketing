#!/usr/bin/env python3
"""For a single aug_31 trajectory, compare every available method's TUM
trajectory against lidar ground truth: ATE and RPE (meters, multiple distance
windows) via `evo`, plus a combined XY overlay plot.

Ground truth: `<data-root>/<yoda-seq>/<trajectory-name>/offline/<run-timestamp>/traj.tum`
(raw nanosecond timestamps, lidar/body axis convention: X forward, Y left,
Z up).

Camera-based estimates (this repo's pipeline, AirSLAM, cuVSLAM, ORB_SLAM3) use
the camera optical convention (X right, Y down, Z forward); ground truth uses
body convention (X forward, Y left, Z up). A fixed rotation mapping camera axes
onto body axes was tried and found to be mathematically inert here: evo's
align(correct_scale=False) always solves for the *globally optimal* rotation
via Kabsch/Umeyama SVD over the given correspondences, so pre-rotating the
input by any fixed rotation first provably doesn't change the final aligned
result (verified empirically: identical output with/without it). What
actually matters is that ground truth's own map frame carries an arbitrary,
per-capture heading offset at t=0 (its first pose is nowhere near identity),
on top of the fixed camera/body axis convention -- only a free-rotation fit
can absorb both, which is what align() already does. This matches
compare_ground_truth.py's own documented finding for its whole-trajectory fit.

Only the first --n-align poses are used to fit each alignment (evo's
PosePath3D.align(n=...)), rather than compare_ground_truth.py's whole-
trajectory fit. Too few poses (e.g. 10) make the fit poorly conditioned when
the initial segment is close to straight-line motion (verified: n=10 gave
~20x worse ATE than a whole-trajectory fit on a test trajectory here); n=30
(default) was chosen empirically as a reasonable floor that stays "local"
while giving the rotation fit enough directional variation to be reliable.

`evo` is used only for TUM loading, timestamp association, and the n-pose
alignment fit. ATE/RPE are computed with compare_ground_truth.py's own
functions, not evo's built-in RPE metric: evo's default RPE
(PoseRelation.translation_part) decomposes relative motion into each
trajectory's own LOCAL/body pose frame (Q_i^-1 @ Q_j vs P_i^-1 @ P_j) before
comparing -- this is exactly the axis-convention bug compare_ground_truth.py
already found and fixed (camera-forward=local-Z vs lidar-forward=local-X is a
fixed right-multiplied offset that does NOT cancel under that decomposition).
Verified empirically here: evo's RPE gave ~130-170% relative error at every
window regardless of alignment quality, while a GT-vs-GT self-check with the
same evo RPE call correctly gave ~0 -- confirming the metric itself isn't
broken, just unusable across the axis-convention gap. compare_ground_truth.py's
compute_rpe_distance diffs world-frame displacement vectors directly (alignment-
invariant, no local-frame decomposition) and gives physically sane numbers
(e.g. ~8cm at a 1m window vs. the same evo call's ~1.7m) on the same aligned
data.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from evo.core import sync
from evo.tools import file_interface

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_ground_truth as cgt  # noqa: E402

SOTA_METHOD_FILES = {
    "AirSLAM": "sota/AirSLAM/{name}.txt",
    "cuVSLAM": "sota/cuVSLAM/{name}.txt",
    "ORBSLAM3": "sota/ORBSLAM3/{name}.txt",
}

PIPELINE_METHOD_FILES = {
    "global_ba": {
        "default": "pipeline_default/{name}/traj_global_ba.tum",
        "clahe": "pipeline_clahe/{name}/traj_global_ba.tum",
    },
    "incremental": {
        "default": "pipeline_default/{name}/traj.tum",
        "clahe": "pipeline_clahe/{name}/traj.tum",
    },
}

COLORS = {
    "default": "tab:orange",
    "clahe": "tab:green",
    "AirSLAM": "tab:red",
    "cuVSLAM": "tab:purple",
    "ORBSLAM3": "tab:brown",
}


def stage_camera_tum(src: Path, dst: Path) -> None:
    """Copy a seconds-timestamp TUM file to dst unchanged (evo's free-rotation
    alignment absorbs the camera/body axis convention difference -- see
    module docstring)."""
    data = np.loadtxt(src)
    if data.ndim == 1:
        data = data[None, :]
    np.savetxt(dst, data, fmt="%.9f")


def stage_ground_truth_tum(src: Path, dst: Path) -> None:
    """Copy a raw-nanosecond-timestamp TUM file to dst with seconds timestamps."""
    data = np.loadtxt(src)
    if data.ndim == 1:
        data = data[None, :]
    data[:, 0] = data[:, 0] * 1e-9
    np.savetxt(dst, data, fmt="%.9f")


def find_ground_truth(data_root: Path, name: str) -> Path | None:
    # name is e.g. "<yoda_seq>_region0_occ0"; GT lives one level deeper than
    # the pipeline/sota results, nested under the parent yoda_seq directory:
    # <data_root>/<yoda_seq>/<name>/offline/<run-timestamp>/traj.tum
    yoda_seq = name.rsplit("_region0_", 1)[0]
    candidates = sorted((data_root / yoda_seq / name / "offline").glob("*/traj.tum"))
    if not candidates:
        return None
    if len(candidates) > 1:
        print(f"WARNING: multiple ground-truth files found for {name}, using {candidates[0]}")
    return candidates[0]


def align_and_evaluate(traj_ref, traj_est, n_align: int, max_dt: float):
    """Associate + align (evo), return world-frame xyz/rotations for both,
    ready for compare_ground_truth.py's ATE/RPE math."""
    traj_ref_sync, traj_est_sync = sync.associate_trajectories(traj_ref, traj_est, max_diff=max_dt)
    traj_est_sync.align(traj_ref_sync, correct_scale=False, n=n_align)

    def to_xyz_rot(traj):
        xyz = traj.positions_xyz
        quat_wxyz = traj.orientations_quat_wxyz
        quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
        return xyz, Rotation.from_quat(quat_xyzw)

    xyz_gt, rot_gt = to_xyz_rot(traj_ref_sync)
    xyz_est, rot_est = to_xyz_rot(traj_est_sync)
    return traj_ref_sync.num_poses, xyz_gt, rot_gt, xyz_est, rot_est


def compute_metrics(xyz_gt, rot_gt, xyz_est, rot_est, windows_m: list[float]) -> dict:
    idx = np.arange(len(xyz_gt))

    err = np.linalg.norm(xyz_est - xyz_gt, axis=1)
    ate = {
        "rmse": float(np.sqrt((err**2).mean())),
        "mean": float(err.mean()),
        "median": float(np.median(err)),
        "max": float(err.max()),
    }

    rpe_by_window = {}
    for w in windows_m:
        trans_errs, rot_errs = cgt.compute_rpe_distance(idx, idx, xyz_est, rot_est, xyz_gt, rot_gt, w)
        if len(trans_errs) == 0:
            rpe_by_window[w] = {"trans_m": None, "rot_deg": None}
            continue
        rpe_by_window[w] = {
            "trans_m": {
                "rmse": float(np.sqrt((trans_errs**2).mean())),
                "mean": float(trans_errs.mean()),
                "median": float(np.median(trans_errs)),
                "max": float(trans_errs.max()),
                "n": int(len(trans_errs)),
            },
            "rot_deg": {
                "rmse": float(np.sqrt((rot_errs**2).mean())),
                "mean": float(rot_errs.mean()),
                "median": float(np.median(rot_errs)),
                "max": float(rot_errs.max()),
                "n": int(len(rot_errs)),
            },
        }

    return {"ate": ate, "rpe": rpe_by_window}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trajectory-name", required=True, help="e.g. yoda_bridge_0fps_ae_1_1969_12_31-19_04_09_region0_occ0")
    parser.add_argument("--result-name", default=None,
                         help="Override the name used to locate/name result files and out-dir (pipeline_default/"
                         "{result-name}/..., sota/<Method>/{result-name}.txt, multi_method/{result-name}/), while "
                         "--trajectory-name alone still drives ground-truth lookup. Default: same as --trajectory-name. "
                         "Use e.g. '<name>_hdrflow' to file an alternate-image-set run's results separately without "
                         "duplicating that trajectory's ground truth.")
    parser.add_argument("--results-root", default="docs/results/aug_31_all", help="Root containing pipeline_default/pipeline_clahe/sota")
    parser.add_argument("--data-root", default="/home/alien/data/yoda/aug_31",
                         help="Root containing <yoda_seq>/<trajectory-name>/offline/*/traj.tum")
    parser.add_argument("--variant", choices=["global_ba", "incremental"], default="global_ba",
                         help="Which pipeline_default/pipeline_clahe TUM file to use "
                              "(traj_global_ba.tum vs traj.tum); SOTA methods are unaffected")
    parser.add_argument("--rpe-windows", default="1,5,10,20,40", help="Comma-separated RPE distance windows in meters")
    parser.add_argument("--n-align", type=int, default=30,
                         help="Number of leading associated poses used to fit alignment "
                              "(empirically chosen floor for a well-conditioned rotation fit -- see module docstring)")
    parser.add_argument("--max-dt", type=float, default=0.05, help="Max association time gap in seconds")
    parser.add_argument("--out-dir", default=None, help="Default: <results-root>/multi_method/<trajectory-name>")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    data_root = Path(args.data_root)
    name = args.trajectory_name
    result_name = args.result_name or name
    windows = [float(w) for w in args.rpe_windows.split(",") if w.strip()]
    out_dir = Path(args.out_dir) if args.out_dir else results_root / "multi_method" / result_name
    out_dir.mkdir(parents=True, exist_ok=True)
    method_files = {**PIPELINE_METHOD_FILES[args.variant], **SOTA_METHOD_FILES}

    gt_src = find_ground_truth(data_root, name)
    if gt_src is None:
        yoda_seq = name.rsplit("_region0_", 1)[0]
        raise SystemExit(f"No ground truth found for {name} under {data_root}/{yoda_seq}/{name}/offline/*/traj.tum")

    all_metrics = {}
    plot_data = {}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        gt_staged = tmp / "gt.tum"
        stage_ground_truth_tum(gt_src, gt_staged)
        traj_ref = file_interface.read_tum_trajectory_file(gt_staged)
        plot_data["ground truth (lidar)"] = ("tab:blue", traj_ref.positions_xyz)

        for method, rel_path in method_files.items():
            src = results_root / rel_path.format(name=result_name)
            if not src.exists():
                print(f"[{method}] SKIPPED (missing: {src})")
                continue

            staged = tmp / f"{method}.tum"
            stage_camera_tum(src, staged)
            traj_est = file_interface.read_tum_trajectory_file(staged)

            try:
                n_poses, xyz_gt, rot_gt, xyz_est, rot_est = align_and_evaluate(
                    traj_ref, traj_est, args.n_align, args.max_dt
                )
            except Exception as e:
                print(f"[{method}] FAILED to align/associate: {e}")
                continue

            if n_poses < args.n_align:
                print(f"[{method}] WARNING: only {n_poses} associated poses, "
                      f"fewer than --n-align={args.n_align}")

            result = compute_metrics(xyz_gt, rot_gt, xyz_est, rot_est, windows)
            all_metrics[method] = result
            plot_data[method] = (COLORS.get(method, "tab:gray"), xyz_est)
            print(f"[{method}] ATE RMSE={result['ate']['rmse']:.4f} m "
                  f"(mean={result['ate']['mean']:.4f}, median={result['ate']['median']:.4f}, "
                  f"max={result['ate']['max']:.4f}), n_poses={n_poses}")
            for w in windows:
                entry = result["rpe"][w]
                trans = entry.get("trans_m")
                rot = entry.get("rot_deg")
                if trans is None:
                    print(f"    {w:>6.1f} m: not enough associated span for this window.")
                    continue
                print(f"    {w:>6.1f} m (n={trans['n']:4d}): trans RMSE={trans['rmse']:.4f} m "
                      f"({100.0 * trans['rmse'] / w:.2f}%), rot RMSE={rot['rmse']:.3f} deg")

    metrics_txt = out_dir / f"metrics_{args.variant}.txt"
    with open(metrics_txt, "w") as f:
        f.write(f"Trajectory: {name}\n")
        f.write(f"Ground truth: {gt_src}\n")
        f.write(f"Alignment: first {args.n_align} associated poses, rigid (no scale, evo free-rotation fit)\n\n")
        for method, result in all_metrics.items():
            f.write(f"=== {method} ===\n")
            ate = result["ate"]
            f.write(f"ATE: RMSE={ate['rmse']:.4f} m, mean={ate['mean']:.4f} m, "
                    f"median={ate['median']:.4f} m, max={ate['max']:.4f} m\n")
            f.write("RPE by distance window:\n")
            for w in windows:
                entry = result["rpe"][w]
                trans = entry.get("trans_m")
                rot = entry.get("rot_deg")
                if trans is None:
                    f.write(f"  {w:>6.1f} m: not enough associated span for this window.\n")
                    continue
                f.write(f"  {w:>6.1f} m (n={trans['n']:4d}): trans RMSE={trans['rmse']:.4f} m "
                        f"({100.0 * trans['rmse'] / w:.2f}%), median={trans['median']:.4f} m, "
                        f"rot RMSE={rot['rmse']:.3f} deg ({rot['rmse'] / w:.3f} deg/m)\n")
            f.write("\n")
    print(f"Wrote {metrics_txt}")

    metrics_json = out_dir / f"metrics_{args.variant}.json"
    with open(metrics_json, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Wrote {metrics_json}")

    fig, ax = plt.subplots(figsize=(8, 8))
    for label, (color, xyz) in plot_data.items():
        style = "--" if label.startswith("ground truth") else "-"
        ax.plot(xyz[:, 0], xyz[:, 1], style, label=label, color=color)
    ax.set_xlabel("x (forward) [m]")
    ax.set_ylabel("y (left) [m]")
    ax.set_title(f"{name} ({args.variant})\n(aligned on first {args.n_align} poses, lidar/body frame)")
    ax.legend()
    ax.axis("equal")
    plt.tight_layout()
    plot_path = out_dir / f"trajectories_overlay_{args.variant}.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
