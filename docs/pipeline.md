# Pipeline walkthrough

This document traces exactly what happens to one image as it moves through the
`pose_graph_bracketing` stereo landmark-BA pipeline (branch `slam-landmark-ba`),
and lists every point where a keypoint, a match, or a whole frame can be
dropped from -- or down-weighted in -- the optimization, with the exact config
field controlling it. It reflects the code as it exists in `src/pose_graph_bracketing/`
and `configs/default.yaml` at the time of writing.

Config values quoted below are `configs/default.yaml`'s checked-in defaults
(`tracking.max_corners: 1000`) unless noted. The overnight benchmark run
(2026-08-19) used `--max-corners 500` as a CLI override -- see the results
summary for those numbers; the doc's examples still reference the repo
default of 1000 for clarity.

## 1. Overview

Per stereo frame `i`, the pipeline:

1. Loads + demosaics + crops the left and right raw images.
2. Extracts DISK keypoints/descriptors independently on each image, bucketed
   by a spatial grid to spread them across the frame.
3. Matches left-vs-right keypoints with LightGlue and triangulates the
   matched pairs into metric 3D stereo observations (rectified pixel
   coordinates + a 3D point), using the calibrated stereo rig.
4. Matches frame `i`'s left keypoints against each of the last
   `graph.vo_lookback` frames' left keypoints (also via LightGlue), and for
   each match either extends an existing persistent landmark's track or
   seeds a brand new one.
5. Adds one `GenericStereoFactor3D` reprojection factor per (frame, landmark)
   sighting, plus a constant-body-velocity motion-prior factor between
   consecutive frames, to a `gtsam_unstable.IncrementalFixedLagSmoother`.
6. The smoother updates its Bayes tree and marginalizes out any
   pose/velocity/landmark variable older than `graph.smoother_lag_s` behind
   the newest observed timestamp, and returns the current pose estimate for
   frame `i`.

There is no place recognition / loop closure: a landmark that falls out of
the lookback window is never re-identified later, even if the camera revisits
the same physical location.

## 2. Image loading & preprocessing (`imaging.py`)

- `load_raw`: reads the raw single-channel Bayer image via `cv2.imread(..., IMREAD_UNCHANGED)`.
- `demosaic`: converts to BGR using the pattern in `dataset.bayer_pattern`
  (`RGGB`/`BGGR`/`GRBG`/`GBRG`, or `none` for already-demosaiced input).
- `crop_bottom`: drops the bottom `preprocessing.crop_bottom_px` rows (default
  175px -- removes a vendor overlay/status bar baked into the raw frame).
- Optional (disabled by default): `apply_clahe` (`preprocessing.clahe_enabled`)
  and `apply_gaussian_blur` (`preprocessing.gaussian_blur_enabled`), neither
  used in the current benchmark configuration.

No keypoint/frame discarding happens at this stage -- every loaded frame is
processed.

## 3. Stereo calibration & rectification (`stereo.py`, `calibration.py`)

`load_stereo_rig` reads `calibration/stereo_calibration_{left,right}.yaml`:
intrinsics (`K`, `dist`) plus each side's precomputed rectification rotation
(`R1`/`R2`) and projection matrix (`P1`/`P2`). The vendor calibration is
already stereo-rectified, so per-point rectification is just
`cv2.undistortPoints(..., R=R1_or_R2, P=P1_or_P2)` (`rectify_points`) --
no image warping. `stereo_calibration` builds a `gtsam.Cal3_S2Stereo` from
`P1`'s shared focal length/principal point and the baseline derived from
`P2`'s translation term (`baseline_m = -P2[0,3] / fx`).

All graph poses live in this rectified-left-camera frame; nothing is rotated
back to the original distorted-image frame anywhere downstream.

## 4. Feature extraction (`features.py`, `DiskExtractor`) -- discard point (a)

DISK (via `kornia.feature.DISK.from_pretrained`) runs independently on the
left and right image each frame:

- Requests `tracking.max_corners * disk.kp_oversample_factor` candidate
  keypoints (default `1000 * 3 = 3000`) with `disk.window_size` (5) and
  `disk.score_threshold` (0.0).
- **Grid bucketing** (`bucket_keypoints`): splits the image into a
  `tracking.grid_rows x tracking.grid_cols` grid (default 2x2 = 4 cells),
  and within each cell keeps only the top `max_corners // n_cells` keypoints
  by DISK's own detection score (default `1000 // 4 = 250` per cell, 1000
  total). Everything else DISK proposed is dropped here and is never
  represented as data anywhere downstream -- the diagnostic video (Section
  11) cannot show these rejects, since `FrameFeatures` never contains them.

This is a **hard, silent discard** driven purely by spatial coverage +
detector confidence, independent of anything about matching or 3D geometry.

## 5. Feature matching (`matching.py`, `LightGlueMatcher`)

LightGlue (`kornia.feature.LightGlueMatcher("disk")`) matches two
`FrameFeatures` sets (used both for left/right stereo matching in Section 6,
and for left/left temporal matching in Section 7). It returns, per candidate
match, a raw score `dists` in `[0, 1]` (higher = better -- note this is the
kornia *score*, not a distance, despite the variable name; discovered/fixed
earlier in this project's development).

- **Discard**: any candidate pair with `confidence < lightglue.min_confidence`
  (default 0.9) is dropped before `MatchResult` is even returned. This is the
  only matching-stage filter; there is no ratio test or cross-check beyond
  what LightGlue's own attention-based matching already encodes.

## 6. Stereo triangulation (`stereo.py::compute_stereo_observations`) -- defines "has valid stereo depth"

For each surviving left/right match: rectify both points, compute
`disparity = uL - uR`, and:

- **Discard** (`min_disparity_px`, default 1.0px): points with
  `disparity <= min_disparity_px` are dropped before triangulation -- avoids
  the near-degenerate/near-infinite depth that a tiny disparity implies.
- Triangulate the surviving pairs via `cv2.triangulatePoints(P1, P2, ...)`.
- **Discard** (`max_depth_m`, default 60.0m): triangulated points with
  `z <= 0` (behind the camera) or `z >= max_depth_m` are dropped.

What's left is exactly the set of left keypoints with "valid stereo depth"
referenced throughout the rest of this doc and in the visualization's
keypoint coloring (Section 11) -- a `{left_keypoint_idx -> (uL, uR, v, xyz)}`
map, cached per frame in `PoseGraphBuilder._stereo_obs_cache`.

## 7. Landmark tracking & creation (`landmarks.py`, `graph_builder._emit_landmark_observations`) -- discard point (b)

For frame `i`, temporal matching runs against every frame `j` in
`[max(0, i - graph.vo_lookback), i)` (default lookback = 4 frames). For each
LightGlue match `(idx_a in frame j, idx_b in frame i)`, in order:

1. **`already_linked`** -- if frame `i`'s keypoint `idx_b` is already tied to
   a landmark from an earlier `j` this same round, the later match is
   dropped (each keypoint sighting maps to at most one landmark per frame).
2. Look up whether frame `j`'s keypoint `idx_a` already belongs to a
   landmark (i.e. `j` itself matched some earlier frame and this point is
   already tracked).
   - **If not** (this would be a *brand-new* landmark): require **both**
     endpoints to have valid stereo depth (Section 6) --
     `no_depth_j` / `no_depth_i` discard otherwise. This is deliberate: a
     landmark seeded from only one stereo-depth observation could later
     receive no second constraint and destabilize the optimizer (an
     `IndeterminantLinearSystemException` was observed in earlier
     development before this rule was added). If both have depth,
     triangulate an initial 3D value from frame `j`'s stereo observation
     (`_triangulate_landmark`); **`triangulation_failed`** discard if the
     backprojected point is non-finite. Otherwise: allocate a new landmark
     id, insert its initial value + a weak `PriorFactorPoint3`
     (`stereo.landmark_prior_sigma`, default 3.0m -- see Section 8), and add
     frame `j`'s `GenericStereoFactor3D` observation.
   - **If it does** (extending an existing landmark): no depth requirement
     on frame `j` (it was already resolved when the landmark was created or
     last extended).
3. Whichever branch above, frame `i`'s side still needs its own valid stereo
   depth: **`no_depth_i`** discard if `idx_b` has none.
4. Otherwise: register `idx_b` against the landmark, add frame `i`'s
   `GenericStereoFactor3D` observation, and mark the match **kept**
   (`new_landmark` or `extended_landmark`).

Every kept observation records its landmark's timestamp for the fixed-lag
smoother (Section 9). A per-frame `n_landmark_observations` count (`n_obs`)
is the sum of factors added this round.

## 8. Factor graph construction (`factors.py`)

Two factor types are added per frame:

- **Motion prior** (`make_motion_prior_factor`, `gtsam.CustomFactor` with
  numerical Jacobians): a 12-dim residual between consecutive frames
  `[Logmap(predicted(X_i, V_i, dt).between(X_j)); V_j - V_i]`, `predicted`
  = constant-body-velocity extrapolation of `X_i` by default. Noise scales
  with `sqrt(dt)` (`motion_prior.rotation_sigma`, `translation_sigma`,
  `angular_velocity_rw_sigma`, `linear_velocity_rw_sigma`). This is not a
  discard mechanism -- it always contributes, softly regularizing the pose
  estimate between consecutive frames.

  **`motion_prior.zero_motion` toggle** (`configs/zero_motion_prior.yaml`):
  when true, `predicted = X_i` directly (assume no motion happened, applied
  uniformly to every frame) instead of the constant-velocity extrapolation,
  with its own deliberately loose, flat (not `sqrt(dt)`-scaled) sigmas
  (`zero_motion_rotation_sigma: 3.14159`, `zero_motion_translation_sigma:
  10.0`) rather than reusing the constant-velocity sigmas above, which are
  calibrated for deviation from a *good* prediction and would be an
  extremely confident, wrong prior if reused here. Ported (with the same
  ablation already validated) from `pose_graph_bracketing`'s
  `vision-refine-oscillation` branch: on a bracketed-exposure sequence, ATE
  barely changes with `zero_motion` (0.78m->0.83m, 1.39m->1.46m regionally
  -- there's usually enough vision to not need the kinematic model); on a
  single-exposure baseline, regional ATE collapses 5-8x specifically where
  it goes fully blind for multiple consecutive frames (0.54m->4.16m,
  0.89m->4.80m) -- confirming the default constant-velocity model was
  masking a real, load-bearing accuracy gap, not a difference any
  reasonable motion model would paper over regardless of exposure
  strategy. Reproduces the `vision-refine-oscillation` numbers to within a
  few percent on this branch's plain DISK+LightGlue pipeline (no
  `refine.py` sub-pixel correction here) -- not an artifact of that later
  work.
- **Stereo observation factor** (`make_stereo_observation_factor`): wraps
  GTSAM's built-in `GenericStereoFactor3D` in a `noiseModel.Robust` +
  `mEstimator.Huber` kernel (`stereo.pixel_sigma` = 1.0px base noise,
  `stereo.huber_k` = 1.345, the standard ~95%-efficiency constant). This is a
  **soft discard**: an outlier observation (e.g. a mismatch during a fast
  rotation) is down-weighted in the optimizer's cost function rather than
  hard-dropped -- unlike everything in Sections 4-7, which are hard
  keep/drop decisions made *before* the factor graph is touched.
- **Landmark prior** (`gtsam.PriorFactorPoint3`, `stereo.landmark_prior_sigma`
  = 3.0m): added once, at landmark creation, anchoring it near its initial
  triangulated position. Prevents rare numerical divergence under
  Huber-downweighted support (see Section 7, item 2).

## 9. Incremental optimization & marginalization (`graph_builder.py`)

`gtsam_unstable.IncrementalFixedLagSmoother(graph.smoother_lag_s, ISAM2Params())`
(default `smoother_lag_s` = 1.0s) receives, every frame, the new factors +
initial values + a `FixedLagSmootherKeyTimestampMap` tagging every
pose/velocity/landmark key touched this round with its owning frame's
timestamp (a landmark newly created from frame `j`'s observation is tagged
with `j`'s timestamp, not `i`'s, so it doesn't outlive its actual last
sighting just because it was *referenced* again later).

**Frame-level discard is implicit, not explicit**: there is no rule that
drops a whole frame from the trajectory. Every frame gets a pose estimate
regardless of how few landmark observations it produced (`n_obs` can be very
low -- see `visualization.low_info_threshold`, Section 11 -- without the
frame being skipped). A frame with too little geometric support is instead
*implicitly* weakly-constrained: its motion-prior factor is still the only
thing anchoring it, so the video's `LOW-INFO` banner is a diagnostic flag,
not an active filter.

`smoother.update(graph, initial, timestamps)` triggers ISAM2's incremental
Bayes-tree update; any variable whose most recent timestamp is now more than
`smoother_lag_s` behind the newest inserted timestamp gets properly
marginalized (its information folded into the remaining graph via the
Bayes tree, not simply dropped or replaced with a crude re-anchoring prior,
which is what this repo did before building GTSAM from source with
`GTSAM_BUILD_UNSTABLE=ON`). `smoother.calculateEstimate()` returns the
updated `current_estimate` used for the next frame's motion prediction.

`smoother_lag_s` must comfortably exceed `vo_lookback` frames' worth of real
elapsed time for the slowest-fps dataset in use (e.g. a 10fps dataset needs
`vo_lookback * 0.1s = 0.4s` of lookback headroom; the default 1.0s gives
margin), or a still-in-lookback-window frame's variables could be
marginalized out from under it.

## 10. Summary table: every discard / down-weight criterion

| # | Stage | Criterion | Config field | Default | Effect |
|---|-------|-----------|---------------|---------|--------|
| a | DISK extraction (`features.py`) | per-grid-cell top-K by detection score | `tracking.max_corners`, `tracking.grid_rows`, `tracking.grid_cols` | 1000, 2, 2 | hard drop, keypoint never exists downstream |
| b1 | LightGlue matching (`matching.py`) | match confidence threshold | `lightglue.min_confidence` | 0.9 | hard drop, candidate match never returned |
| b2 | Stereo triangulation (`stereo.py`) | minimum disparity | `stereo.min_disparity_px` | 1.0 px | hard drop, no stereo depth for that keypoint |
| b3 | Stereo triangulation (`stereo.py`) | max triangulated depth / positive depth | `stereo.max_depth_m` | 60.0 m | hard drop, no stereo depth for that keypoint |
| c1 | Landmark creation (`graph_builder.py`) | already linked to a landmark this frame | -- | -- | hard drop (`already_linked`) |
| c2 | Landmark creation (`graph_builder.py`) | new landmark requires depth at **both** endpoints | (depends on b2/b3 above) | -- | hard drop (`no_depth_i`/`no_depth_j`) |
| c3 | Landmark creation (`graph_builder.py`) | triangulated init value must be finite | -- | -- | hard drop (`triangulation_failed`) |
| c4 | Landmark extension (`graph_builder.py`) | frame `i`'s own depth still required | (depends on b2/b3) | -- | hard drop (`no_depth_i`) |
| d | Stereo observation factor (`factors.py`) | robust down-weighting of reprojection residual | `stereo.pixel_sigma`, `stereo.huber_k` | 1.0 px, 1.345 | soft (down-weight, not drop) |
| e | Landmark prior (`factors.py`) | anchoring around initial triangulation | `stereo.landmark_prior_sigma` | 3.0 m | soft (regularization, prevents divergence) |
| f | Fixed-lag smoother (`graph_builder.py`) | variable age since last touch | `graph.smoother_lag_s` | 1.0 s | marginalization (proper, not a drop) |
| g | (diagnostic only, no effect on optimization) | frame's `n_landmark_observations` | `visualization.low_info_threshold` | 20 | flags `LOW-INFO` in the video banner; frame is still fully processed and included in the trajectory |

Whole *frames* are never dropped from the output trajectory -- every input
frame gets a pose (row g is diagnostic-only).

## 11. Reading the diagnostic visualization video (`visualization.py`, `--visualize`)

Enable with `scripts/run_trajectory.py --visualize [--visualize-out PATH]`
(`visualization.enabled` / `output_path` in config). Disabled by default and
adds essentially zero overhead when off (`keypoint_status`/`visual_matches`
stay `None` and the extra bookkeeping in `_emit_landmark_observations` is
skipped entirely). One composite frame is written per processed frame, top
to bottom:

1. **Legend strip.**
2. **Current frame panel**: frame `i`'s left image with every surviving
   (post-bucketing) keypoint drawn as a dot, colored:
   - green = **used** this round (tied to a kept landmark observation)
   - yellow = has valid stereo depth (Section 6) but wasn't matched/used
     this round
   - gray = no valid stereo depth at all this round
3. **Up to `graph.vo_lookback` stacked row-panels**, one per lookback frame
   `j` that had at least one recorded match this round (padded with a plain
   gray "(no lookback frame)" placeholder otherwise, so every video frame is
   pixel-identical in size -- required by `cv2.VideoWriter`): `[frame j |
   frame i]` side by side, with a line per LightGlue match candidate
   (**note**: only candidates that passed `lightglue.min_confidence`, i.e.
   row b1 in the table -- earlier rejects are invisible here too, same as
   DISK bucketing) -- green = kept (added/extended a landmark factor), red =
   discarded (see the reason codes in Section 7 / the table's row c1-c4),
   plus a `frame {j} matches kept={n}/{total}` label.
4. **Bottom banner**: `frame {idx} ts={ts} slot={slot} landmark_obs_added={n}
   status={OK|LOW-INFO}`.

Caveat inherited from Sections 4 and 5: this video can only show data that
survived DISK's grid-bucketing (a) and LightGlue's confidence threshold
(b1) -- earlier candidates are never materialized as drawable data.
