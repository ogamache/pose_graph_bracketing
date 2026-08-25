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

## Chasing bracketed's scale bias: what actually fixes it (2026-08-25)

Follow-up to the section above: bracketed's worse-than-ae ATE on the
`_0fps` pair turned out to be dominated by a **scale bias**, not a shape/
direction error -- confirmed with a scale-inclusive (Sim3) Umeyama
alignment, computed once per scope as
`scale = trace(diag(S)@D) / var_src` (Umeyama's own formula, both terms
consistently normalized -- an inconsistent normalization bug the first
time round produced nonsense 60-400x "scale" values before being caught
and fixed).

**Confirmed on two independent dataset pairs, two branches:**

| | bracketed | ae |
|---|---|---|
| `_10fps`, `slam-landmark-ba` | region3 +13.1%, region4 +9.3%, full +9.9% | region3 -1.6%, region4 +0.1% |
| `_0fps`, `slam-landmark-ba` | region3 +31.6%, region4 +15.5%, full +12.1% | region3 +6.2%, region4 +1.7%, full +1.0% |
| `_0fps`, `vision-refine-oscillation` (refine+radiance) | region3 +31.3%, region4 +15.4%, full +12.1% | region3 +6.3%, region4 +1.7%, full +1.0% |

Bracketed consistently runs "long" relative to GT by ~9-32%; single-
exposure doesn't (aside from its own separate, already-documented
blind-streak compounding-drift effect on `_10fps`'s full scope). This is
real and reproducible, not dataset noise.

### Two hypotheses tested and ruled out

1. **Cross-exposure temporal keypoint-detection bias** (the mechanism
   `refine.py` fixes): tested by running the identical scale-bias
   measurement on `vision-refine-oscillation` (refine+radiance enabled)
   vs. `slam-landmark-ba` (neither) -- see table above, numbers match to
   within 0.3 percentage points. Refine.py makes no measurable difference.
2. **SAE/LAE's own stereo disparity being systematically biased vs. MAE**
   (not a temporal effect, a same-frame stereo-matching-quality effect):
   tested directly by backprojecting every observation of every
   MAE+SAE/LAE "mixed" landmark (2826 in region3) via
   `results[frame_idx].pose` (available for every frame regardless of
   smoother marginalization) and comparing the SAE/LAE-implied 3D position
   against the MAE-implied one along the camera-to-landmark ray. Mean
   signed difference +0.033m, median +0.0005m, 50.4% land farther / 49.6%
   closer -- a coin flip. No directional bias found.

### A theoretical deduction, before more empirical testing

Since excluding/down-weighting far landmarks turned out to fix the bias
(see below), it's worth noting explicitly: **a uniform calibration error**
(wrong baseline, residual rectification error) would bias every depth by
the same factor regardless of range -- near and far points equally. Since
the fix is specifically far-point-selective, a flat calibration
miscalibration is ruled out analytically, without needing a dedicated
test for it.

### Three mechanisms tested empirically, region3 (49 GT-associated points) and region4 (114 points), `b_0fps`

| mechanism | region3 ATE / scale | region4 ATE / scale |
|---|---|---|
| baseline (flat prior, flat noise, `max_depth_m: 60`) | 1.56m / +30.0% | 2.23m / +15.3% |
| `depth_scaled_prior` alone (any floor 0.5-3.0m tried) | 1.37-1.47m / +25-28% | 2.05m / +14.0% |
| `depth_scaled_noise` alone (`ref=2.0m, power=3.0`, tuned) | **0.12m / +1.4%** | **0.42m / +2.4%** |
| `max_depth_m=5.0` alone (existing param, no new code) | **0.12m / +1.1%** | **0.41m / +2.4%** |
| `depth_scaled_prior` + `depth_scaled_noise` combined | 0.13m / +1.4% | 0.51m / +3.0% |

**`depth_scaled_prior` barely moves anything, on either region, at any
floor tested** -- consistent with its earlier rejection on the `_10fps`
pair (`vision-refine-oscillation`, different conditions), now confirmed
a third time under conditions specifically chosen to give it the best
possible chance (a much larger, clearer scale-bias signal than either
prior test had). The mechanism (a landmark's real triangulation
uncertainty growing as `depth^2/(fx*baseline)`, far outstripping the flat
3.0m prior at long range -- confirmed directly from this rig's own
calibration: theoretical sigma is 1.4x the flat prior at 20m, 2x at 24m
(region3's own landmark p90 depth), 6.3x at 43m) is real and correctly
reasoned, but empirically doesn't matter much: a landmark accumulates
many reprojection-factor observations over its lifetime and only one
prior factor at creation, so the repeated per-frame pressure dominates
regardless of how tight or loose that single one-time anchor is.

**`depth_scaled_noise` (the *ongoing*, per-observation reprojection
weighting) is the mechanism that actually matters** -- and a simple hard
`max_depth_m` cutoff achieves essentially the identical result with zero
new code, an existing parameter. Combining prior+noise doesn't improve on
noise alone (region4 slightly worse, region3 a wash) -- further
confirming the prior isn't contributing.

### What this means for the original question ("why does bracketing show a scale bias ae doesn't?")

The mechanism is **not exposure-slot-specific** -- it isn't about SAE/LAE
being lower-quality than MAE (ruled out directly above). It's simpler and
less specific to bracketing than that: **far landmarks are unreliable in
this pipeline's stereo triangulation regardless of which exposure
produced them**, and bracketing's much larger total landmark count (2-3x
a single-exposure run at comparable per-frame rates, throughout this
project) means more far, noisy landmarks enter the graph in absolute
terms, at a fixed flat `pixel_sigma`/`landmark_prior_sigma` that doesn't
account for them. ae_0fps isn't immune because its landmarks are somehow
better -- it's just producing fewer of them overall, including fewer far
ones, so this failure mode has less surface area to bite.

### Recommendation

Given `max_depth_m=5.0` (already-existing parameter) reproduces
`depth_scaled_noise`'s result almost exactly with zero new code, the
parsimonious choice is to tune `max_depth_m` down rather than keep the
newly-built `depth_scaled_noise`/`depth_scaled_prior` toggles as the
shipped fix. That said, there's a real, untested-here tradeoff: a hard
`max_depth_m=5.0` cutoff *discards* far landmarks entirely (at this
region's landmark depth distribution, median ~14.8m -- over half the
population), while `depth_scaled_noise` *keeps* them, just weighted down,
preserving some fallback value in scenes where near-field features are
scarce (e.g. wide-open spaces). Neither `_0fps` region tested here
distinguishes between these two failure modes, since both regions have
plenty of close structure either way. Not resolved here -- worth keeping
in mind before picking one as the shipped default. `depth_scaled_prior`
should stay off; three independent tests (two dataset pairs, one on each
branch) now agree it isn't doing meaningful work.

All three toggles (`depth_scaled_prior`, `depth_scaled_noise`,
`max_depth_m`) are kept as-is in config, default-off/default-60m --
no default changed as a result of this investigation, pending a decision
on which fix (if any) to actually ship.
