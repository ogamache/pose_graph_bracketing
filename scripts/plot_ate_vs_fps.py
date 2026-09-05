#!/usr/bin/env python3
"""Plot ATE RMSE (from compare_multi_method_ground_truth.py's metrics_<variant>.json)
vs. capture FPS, for every _occ0 trajectory, using only this repo's own
pipeline (--method, default "default" = global-BA custom SLAM). Two series:
"bracketing" (trajectory name has no "_ae_") and "ae" (auto-exposure,
"_ae_" in the name).

Always produces two plots per run: one using the global-BA pipeline estimates
(metrics_global_ba.json) and one using the incremental estimates
(metrics_incremental.json).

Each (category, fps) combination now has multiple repeat trajectories (e.g.
"yoda_bridge_0fps_ae_1_...", "..._ae_2_...", up to 5 repeats). These are
aggregated per (category, fps) two ways, each saved as its own plot: mean
with a mean +/- 1 std shaded band, and median with a Q1-Q3 (25th-75th
percentile) shaded band. Combined with the two pipeline variants
(global_ba, incremental), this produces four plots per run.

FPS is parsed from the trajectory folder name (pattern "<N>fps"); trajectories
with no fps in their name (e.g. "yoda_bridge_2_...", "yoda_fixedev_1_...") are
assumed to be captured at the sensor's native 32 FPS. "0fps" in a name is
likewise treated as equivalent to no fps written at all (32 FPS), not literal
0 -- it means no fps limit was applied.

The x-axis is not FPS itself but an "apparent speed for a 30 FPS camera":
each trajectory's actual vehicle speed is estimated from its ground-truth
(lidar) trajectory as the median instantaneous speed (|delta position| /
delta t between consecutive GT poses). At `fps` frames/sec the vehicle moves
actual_speed/fps meters between frames; a 30 FPS camera would need to travel
at actual_speed * (30/fps) to see that same per-frame displacement, so that
is the value plotted on the x-axis -- it folds the fps-reduction effect into
an equivalent apparent-speed effect at a fixed 30 FPS.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_multi_method_ground_truth as cmg  # noqa: E402

FPS_RE = re.compile(r"(\d+)fps")
REFERENCE_FPS = 30.0

COLORS = {
    "bracketing": "tab:orange",
    "ae": "tab:blue",
}


def parse_fps(name: str) -> int:
    m = FPS_RE.search(name)
    fps = int(m.group(1)) if m else 32
    return 32 if fps == 0 else fps


def category(name: str) -> str:
    return "ae" if "_ae_" in name else "bracketing"


def median_gt_speed(data_root: Path, name: str) -> float | None:
    """Median instantaneous speed (m/s) over the ground-truth trajectory."""
    gt_src = cmg.find_ground_truth(data_root, name)
    if gt_src is None:
        return None
    data = np.loadtxt(gt_src)
    if data.ndim == 1 or data.shape[0] < 2:
        return None
    t = data[:, 0] * 1e-9
    xyz = data[:, 1:4]
    dt = np.diff(t)
    dist = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    valid = dt > 0
    if not np.any(valid):
        return None
    return float(np.median(dist[valid] / dt[valid]))


AGG_LABELS = {
    "mean_std": "line=mean, shaded=mean±1 std",
    "median_iqr": "line=median, shaded=Q1-Q3",
}


def center_and_band(values: list[float], agg: str) -> tuple[float, float, float]:
    """Return (center, lower, upper) for a list of repeat values."""
    if agg == "mean_std":
        center = float(np.mean(values))
        std = float(np.std(values))
        return center, max(center - std, 0.0), center + std
    center = float(np.median(values))
    lower = float(np.percentile(values, 25))
    upper = float(np.percentile(values, 75))
    return center, lower, upper


def run(args, variant: str, agg: str) -> None:
    rpe_windows = [float(w) for w in args.rpe_windows.split(",") if w.strip()]

    results_root = Path(args.results_root)
    data_root = Path(args.data_root)
    multi_method_dir = results_root / "multi_method"
    if args.out:
        out_base = Path(args.out)
        out_path = out_base.with_name(f"{out_base.stem}_{variant}_{agg}{out_base.suffix}")
    else:
        out_path = results_root / f"ate_vs_speed_{args.occ}_{variant}_{agg}.png"

    traj_dirs = sorted(
        d for d in multi_method_dir.iterdir()
        if d.is_dir() and d.name.endswith(f"_{args.occ}") and "fixedev" not in d.name
    )
    if not traj_dirs:
        raise SystemExit(f"No *_{args.occ} trajectories found under {multi_method_dir}")

    # category ("bracketing"/"ae") -> list of (fps, ate_rmse, rpe_median_rmse, traj_name)
    series: dict[str, list[tuple[int, float, float, str]]] = {}
    for traj_dir in traj_dirs:
        metrics_path = traj_dir / f"metrics_{variant}.json"
        if not metrics_path.exists():
            print(f"SKIPPED (no metrics_{variant}.json): {traj_dir.name}")
            continue
        with open(metrics_path) as f:
            data = json.load(f)
        if args.method not in data:
            print(f"SKIPPED (no '{args.method}' result): {traj_dir.name}")
            continue
        result = data[args.method]

        rpe_rmses = []
        for w in rpe_windows:
            entry = result["rpe"].get(str(w))
            if entry and entry.get("trans_m"):
                rpe_rmses.append(entry["trans_m"]["rmse"])
        if not rpe_rmses:
            print(f"SKIPPED (no RPE windows {rpe_windows} available): {traj_dir.name}")
            continue
        rpe_median = float(np.median(rpe_rmses))

        fps = parse_fps(traj_dir.name)
        speed = median_gt_speed(data_root, traj_dir.name)
        if speed is None:
            print(f"SKIPPED (no ground truth for speed estimate): {traj_dir.name}")
            continue
        apparent_speed = speed * (REFERENCE_FPS / fps)

        cat = category(traj_dir.name)
        series.setdefault(cat, []).append((fps, apparent_speed, result["ate"]["rmse"], rpe_median, traj_dir.name))

    # category -> fps -> list of (apparent_speed, ate_rmse, rpe_median_rmse) across repeat trajectories
    grouped: dict[str, dict[int, list[tuple[float, float, float]]]] = {}
    for cat, points in series.items():
        by_fps: dict[int, list[tuple[float, float, float]]] = {}
        for fps, apparent_speed, ate, rpe_med, _name in points:
            by_fps.setdefault(fps, []).append((apparent_speed, ate, rpe_med))
        grouped[cat] = by_fps

    fig, (ax_ate, ax_rpe) = plt.subplots(2, 1, figsize=(9, 10), sharex=True)
    for cat in ("bracketing", "ae"):
        by_fps = grouped.get(cat, {})
        if not by_fps:
            continue
        fps_vals = sorted(by_fps)
        speed_x = np.array([center_and_band([v[0] for v in by_fps[f]], agg)[0] for f in fps_vals])
        order = np.argsort(speed_x)
        speed_x = speed_x[order]
        fps_sorted = [fps_vals[i] for i in order]

        ate_bands = [center_and_band([v[1] for v in by_fps[f]], agg) for f in fps_sorted]
        ate_center = np.array([b[0] for b in ate_bands])
        ate_lower = np.array([b[1] for b in ate_bands])
        ate_upper = np.array([b[2] for b in ate_bands])

        rpe_bands = [center_and_band([v[2] for v in by_fps[f]], agg) for f in fps_sorted]
        rpe_center = np.array([b[0] for b in rpe_bands])
        rpe_lower = np.array([b[1] for b in rpe_bands])
        rpe_upper = np.array([b[2] for b in rpe_bands])

        ax_ate.plot(speed_x, ate_center, "o-", label=cat, color=COLORS[cat])
        ax_ate.fill_between(speed_x, ate_lower, ate_upper, color=COLORS[cat], alpha=0.2, linewidth=0)
        ax_rpe.plot(speed_x, rpe_center, "o-", label=cat, color=COLORS[cat])
        ax_rpe.fill_between(speed_x, rpe_lower, rpe_upper, color=COLORS[cat], alpha=0.2, linewidth=0)

    agg_label = AGG_LABELS[agg]
    ax_ate.set_ylabel("ATE RMSE [m]")
    ax_ate.set_title(f"ATE RMSE vs. apparent speed for a {REFERENCE_FPS:g} FPS camera\n"
                      f"(_{args.occ} trajectories, method={args.method}, variant={variant}, {agg_label})",
                      fontsize=11)
    ax_ate.legend()
    ax_ate.grid(True, alpha=0.3)

    windows_str = ",".join(f"{w:g}" for w in rpe_windows)
    ax_rpe.set_xlabel(f"apparent speed for a {REFERENCE_FPS:g} FPS camera [m/s] "
                       f"(= GT median speed × {REFERENCE_FPS:g}/fps)")
    ax_rpe.set_ylabel("median RPE trans RMSE [m] (over trajectory's own windows)")
    ax_rpe.set_title(f"RPE RMSE over {windows_str} m windows vs. apparent speed ({agg_label})")
    ax_rpe.legend()
    ax_rpe.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")

    print("\nData points used:")
    for cat, by_fps in grouped.items():
        for fps in sorted(by_fps):
            speeds = [v[0] for v in by_fps[fps]]
            ates = [v[1] for v in by_fps[fps]]
            rpes = [v[2] for v in by_fps[fps]]
            speed_c, _, _ = center_and_band(speeds, agg)
            ate_c, ate_lo, ate_hi = center_and_band(ates, agg)
            rpe_c, rpe_lo, rpe_hi = center_and_band(rpes, agg)
            print(f"  [{cat:>10s}] fps={fps:>3d}  n={len(ates)}  "
                  f"apparent_speed={speed_c:.4f} m/s  "
                  f"ATE={ate_c:.4f} [{ate_lo:.4f}, {ate_hi:.4f}] m  "
                  f"RPE={rpe_c:.4f} [{rpe_lo:.4f}, {rpe_hi:.4f}] m")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="docs/results/aug_31_all")
    parser.add_argument("--method", default="default",
                         help="Which method's ATE to plot from metrics_<variant>.json (default: 'default' = this repo's pipeline)")
    parser.add_argument("--rpe-windows", default="1,5,10",
                         help="Comma-separated RPE distance windows (m) to median over for the bottom subplot "
                              "(must match windows present in metrics_<variant>.json)")
    parser.add_argument("--occ", default="occ0", choices=["occ0", "occ1"], help="Which occlusion variant to plot")
    parser.add_argument("--data-root", default="/home/alien/data/yoda/aug_31",
                         help="Root containing <yoda_seq>/<trajectory-name>/offline/*/traj.tum (for speed estimation)")
    parser.add_argument("--out", default=None,
                         help="Output path override. If given, all four combinations are written next to it "
                              "with a _<variant>_<agg> suffix inserted before the extension; default is "
                              "<results-root>/ate_vs_speed_<occ>_<variant>_<agg>.png for each")
    args = parser.parse_args()

    for variant in ("global_ba", "incremental"):
        for agg in ("mean_std", "median_iqr"):
            print(f"=== variant: {variant}, agg: {agg} ===")
            run(args, variant, agg)
            print()


if __name__ == "__main__":
    main()
