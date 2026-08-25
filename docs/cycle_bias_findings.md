# Bracketed vs. single-exposure: ablation findings (`slam-landmark-ba`)

Research log for accuracy/robustness ablations run directly on this
branch's own pipeline (plain DISK+LightGlue stereo landmark-BA, no
radiance normalization or sub-pixel refine -- see `docs/pipeline.md` and
`docs/branch_comparison.md` for what that means relative to
`vision-refine-oscillation`, where most of this project's other ablations
were originally run and documented in more depth).

## `b_0fps` vs. `ae_0fps`: bracketing's advantage doesn't always show up (2026-08-25)

Ran the full evaluation methodology (ATE, RPE at multiple windows,
trajectory-continuity + landmark-provenance robustness metrics) on a
second dataset pair, distinct from the `..._10fps_...` pair used
throughout the rest of this project's ablations:
`yoda_aug_9_b_0fps_02ema_1969_12_31-19_13_05` (bracketed) and
`yoda_aug_9_ae_0fps_02ema_1969_12_31-19_28_58` (single-exposure AE-only).
Despite the "0fps" name, these run at native/unthrottled capture rate, not
literally zero -- ~1120-1140 frames span only ~38s here vs. the `_10fps`
pair's fixed 10fps spacing over a similar frame count.

`configs/default.yaml` (constant-velocity motion prior), both datasets,
region3/region4 (from each dataset's own `segment.txt`) + full sequence,
against each dataset's own offline lidar-mapping ground truth.

### ATE (m)

| scope | bracketed (b_0fps) | ae (ae_0fps) |
|---|---|---|
| region3 | 1.627 | **0.386** |
| region4 | 2.262 | **0.423** |
| full | 2.426 | **0.520** |

### RPE, translation RMSE (m) @ 1m window

| scope | bracketed | ae |
|---|---|---|
| region3 | 1.417 | 1.194 |
| region4 | 1.387 | 1.377 |
| full | 1.275 | 1.325 |

### Trajectory continuity (section 5a)

| scope | metric | bracketed | ae |
|---|---|---|---|
| region3 | zero-obs frames | 5/147 (3.4%) | **0/120 (0%)** |
| region4 | zero-obs frames | 5/347 (1.4%) | **0/337 (0%)** |
| full | zero-obs frames | 11/1118 (1.0%) | **0/1133 (0%)** |
| all scopes | max consecutive zero-obs | 1 | 0 |

Backend resets: 0 in both runs.

### Landmark provenance (section 5b)

Bracketed: 30-35% of landmarks in every scope only exist because SAE or
LAE contributed to them (MAE alone never would have) -- consistent with
every other bracketed sequence tested in this project, including the
`_10fps` pair. `ae_0fps` is single-exposure, so this is trivially 100%
pure-MAE / 0% mixed / 0% SAE-LAE-exclusive in every scope, same as every
other AE-only control case.

### Reading: this reverses the pattern found on the `_10fps` pair

Elsewhere in this project (the `_10fps` dataset pair, see
`vision-refine-oscillation`'s `docs/cycle_bias_findings.md`), the
single-exposure baseline went blind for up to 12 consecutive frames
during saturation transitions, and bracketing's advantage in ATE showed
up specifically once that failure mode was accounted for (either via the
full-trajectory metric, or via the zero-motion-prior ablation that
stripped out the motion model's masking effect).

**On this dataset pair, `ae_0fps` never goes blind at all** -- zero
zero-observation frames in any scope, vs. bracketed's 1.0-3.4%. Whatever
lighting/saturation event drove the `_10fps` pair's ae failure mode
either doesn't occur on this route, or this AE controller handles it
without the multi-frame blackout the other sequence showed. With no
blind streak for bracketing to protect against, bracketing's ATE is
consistently *worse* here (2.4m vs. 0.5m full-trajectory) -- more
matching/triangulation opportunities across three brackets is not an
automatic accuracy win; it's a real cost (more surface area for a
cross-exposure detection-bias or mismatch to enter) when there's no
corresponding blind-streak benefit to offset it against.

**This is useful negative evidence, not a contradiction.** It sharpens
the actual claim: bracketing's accuracy benefit is conditional on the
failure mode it's designed for (saturation-driven multi-frame blind
streaks) actually occurring in the sequence. The information-capture
claim (section 5b, section 5c from `vision-refine-oscillation`) is
unconditional -- bracketing recovers real scene information and produces
real landmarks a single exposure couldn't -- but whether that translates
into a *trajectory accuracy* win depends on whether the single-exposure
baseline actually hits its failure mode on the route in question. Not
every sequence will show it, and this dataset pair is a clean
demonstration of that: same rig, same general route family, opposite
accuracy result.
