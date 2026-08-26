"""Loading of the Yoda-rig image sequence + per-frame exposure metadata."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

# Physical 4-slot bracket-cycle order (sequence_index -> nominal exposure slot).
_SEQUENCE_SLOT_LABELS = {0: "MAE", 1: "LAE", 2: "MAE", 3: "SAE"}


def exposure_label(exposure_factor: float) -> str:
    """Label derived from the *actual* commanded exposure factor for this frame."""
    if exposure_factor < 0.5:
        return "SAE"
    if exposure_factor < 5.0:
        return "MAE"
    return "LAE"


def slot_label(sequence_index: int) -> str:
    """Label derived from the fixed cyclic bracket-slot position."""
    return _SEQUENCE_SLOT_LABELS.get(sequence_index % 4, "UNKNOWN")


@dataclass
class FrameInfo:
    timestamp_ns: int
    image_path: Path
    exposure_us: float
    gain_db: float
    exposure_factor: float
    sequence_index: int
    converged: bool
    exposure_label: str
    slot_label: str
    right_image_path: Path | None = None

    @property
    def timestamp_s(self) -> float:
        return self.timestamp_ns * 1e-9


def load_stereo_sequence(data_dir: str | Path) -> list[FrameInfo]:
    """Like load_sequence(side="left"), but only keeps frames that also have a
    right-image counterpart (left/right frame counts can differ slightly),
    and populates `right_image_path`.
    """
    data_dir = Path(data_dir)
    left_frames = load_sequence(data_dir, side="left")
    right_dir = data_dir / "images_right"
    right_by_ts = {int(p.stem): p for p in right_dir.glob("*.png")}

    frames = []
    for fr in left_frames:
        right_path = right_by_ts.get(fr.timestamp_ns)
        if right_path is None:
            continue
        fr.right_image_path = right_path
        frames.append(fr)
    return frames


def drop_low_information_frames(
    frames: list[FrameInfo],
    bayer_pattern: str = "RGGB",
    crop_bottom_px: int = 0,
    min_brightness: float = 10.0,
    max_brightness: float = 245.0,
) -> list[FrameInfo]:
    """Loads each frame's left image, computes its mean pixel brightness
    (0-255, post demosaic/crop), and drops the frame entirely if it's below
    `min_brightness` (near-black/crushed) or above `max_brightness`
    (near-white/saturated) -- almost certainly near-featureless, so not
    worth the frame's own matching cost or its noisy contribution to
    downstream landmarks. Unlike a metadata-only prediction (exposure
    factor / commanded brightness target), this measures the actual
    rendered image, at the cost of loading every frame once up front.
    """
    from pose_graph_bracketing.imaging import load_preprocessed

    kept = []
    for fr in frames:
        image = load_preprocessed(fr.image_path, bayer_pattern, crop_bottom_px)
        mean_brightness = float(image.mean())
        if min_brightness <= mean_brightness <= max_brightness:
            kept.append(fr)
    return kept


def load_sequence(data_dir: str | Path, side: str = "left") -> list[FrameInfo]:
    """Join images_{side}/<timestamp>.png with images_meta_{side}/images_meta_{side}.csv.

    Returns frames sorted by timestamp.
    """
    data_dir = Path(data_dir)
    images_dir = data_dir / f"images_{side}"
    meta_csv = data_dir / f"images_meta_{side}" / f"images_meta_{side}.csv"

    meta_by_ts: dict[int, dict] = {}
    with open(meta_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = int(row["timestamp"])
            meta_by_ts[ts] = row

    frames: list[FrameInfo] = []
    for image_path in images_dir.glob("*.png"):
        ts = int(image_path.stem)
        row = meta_by_ts.get(ts)
        if row is None:
            continue
        exposure_factor = float(row["exposure_factor"])
        sequence_index = int(row["sequence_index"])
        frames.append(
            FrameInfo(
                timestamp_ns=ts,
                image_path=image_path,
                exposure_us=float(row["exposure_us"]),
                gain_db=float(row["gain_db"]),
                exposure_factor=exposure_factor,
                sequence_index=sequence_index,
                converged=row["converged"].strip().lower() == "true",
                exposure_label=exposure_label(exposure_factor),
                slot_label=slot_label(sequence_index),
            )
        )

    frames.sort(key=lambda fr: fr.timestamp_ns)
    return frames
