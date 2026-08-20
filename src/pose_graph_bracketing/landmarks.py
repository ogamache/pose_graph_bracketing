"""Windowed landmark-track bookkeeping for the stereo bundle-adjustment backend.

Maps a (frame_idx, left_keypoint_idx) sighting to a persistent landmark id,
allocating a new id the first time a keypoint is seen. There is no
re-identification across a gap (no loop closure/place recognition, by
design/scope) -- if a landmark's track falls out of the lookback window and
the same physical point is seen again later, it simply becomes a new landmark.
"""

from __future__ import annotations


class LandmarkTracker:
    def __init__(self) -> None:
        self._next_id: int = 0
        # frame_idx -> {left_keypoint_idx -> landmark_id}
        self._tracks: dict[int, dict[int, int]] = {}

    def get_or_create(self, frame_idx: int, left_kp_idx: int, existing_landmark_id: int | None = None) -> tuple[int, bool]:
        """Look up (or allocate) the landmark id for a keypoint sighting.

        `existing_landmark_id`, if given, is the id already assigned to this
        same physical point in another frame's track (e.g. matched from frame
        j); it gets propagated so both frames' sightings share one landmark.

        Returns (landmark_id, is_new_landmark).
        """
        frame_tracks = self._tracks.setdefault(frame_idx, {})
        if left_kp_idx in frame_tracks:
            return frame_tracks[left_kp_idx], False

        if existing_landmark_id is not None:
            frame_tracks[left_kp_idx] = existing_landmark_id
            return existing_landmark_id, False

        landmark_id = self._next_id
        self._next_id += 1
        frame_tracks[left_kp_idx] = landmark_id
        return landmark_id, True

    def landmark_id_at(self, frame_idx: int, left_kp_idx: int) -> int | None:
        return self._tracks.get(frame_idx, {}).get(left_kp_idx)

    def forget(self, frame_idx: int, left_kp_idx: int) -> None:
        """Undo a registration (used to roll back a batch the caller couldn't commit)."""
        frame_tracks = self._tracks.get(frame_idx)
        if frame_tracks is not None:
            frame_tracks.pop(left_kp_idx, None)

    def evict_before(self, cutoff_frame_idx: int) -> None:
        for idx in [idx for idx in self._tracks if idx < cutoff_frame_idx]:
            del self._tracks[idx]
