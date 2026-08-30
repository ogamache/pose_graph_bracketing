#!/usr/bin/env python3
"""Compute per-frame saturation stats (pixels <5 and >250) for images_left of a trajectory."""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True, help="Trajectory data dir (contains images_left/)")
    p.add_argument("--out", required=True, help="Output CSV path")
    p.add_argument("--low-thresh", type=int, default=5)
    p.add_argument("--high-thresh", type=int, default=250)
    args = p.parse_args()

    images_dir = Path(args.data_dir) / "images_left"
    paths = sorted(images_dir.glob("*.png"))
    if not paths:
        raise SystemExit(f"No images found in {images_dir}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp_ns", "filename", "n_pixels",
            "n_low", "n_high", "frac_low", "frac_high", "frac_saturated",
        ])
        for path in paths:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: failed to read {path}, skipping")
                continue
            n_pixels = img.size
            n_low = int(np.count_nonzero(img < args.low_thresh))
            n_high = int(np.count_nonzero(img > args.high_thresh))
            writer.writerow([
                path.stem, path.name, n_pixels,
                n_low, n_high,
                n_low / n_pixels, n_high / n_pixels,
                (n_low + n_high) / n_pixels,
            ])

    print(f"Wrote saturation stats for {len(paths)} frames to {args.out}")


if __name__ == "__main__":
    main()
