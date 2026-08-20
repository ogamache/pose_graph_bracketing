"""TUM-format trajectory output."""

from __future__ import annotations

from pathlib import Path

import gtsam


def write_tum(path: str | Path, timestamps_s: list[float], poses: list[gtsam.Pose3]) -> None:
    with open(path, "w") as f:
        for ts, pose in zip(timestamps_s, poses):
            t = pose.translation()
            q = pose.rotation().toQuaternion()  # w, x, y, z
            f.write(
                f"{ts:.9f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                f"{q.x():.9f} {q.y():.9f} {q.z():.9f} {q.w():.9f}\n"
            )
