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

## Chasing the *why*: ruling out mixing, brightness, fps, and detector choice (2026-08-25)

Follow-up to the section above: the fix (far-landmark down-weighting) was
confirmed, but *why* `b_0fps` has more far, unreliable landmarks than
`ae_0fps` in the first place was still open. `b_0fps` and `ae_0fps` are
two separate recordings (different dates/times) of the **same physical
route**, not a controlled simultaneous capture -- confirmed by the user,
which rules out "different route geometry" as an explanation and narrows
the field to what's actually different between the two recording
sessions.

### Raw keypoint depth, whole trajectory, windowed by bracket-cycle length

`scripts/analyze_keypoint_depth_windows.py` (new): pure per-frame
DISK+LightGlue stereo triangulation, no temporal tracking or BA, windowed
by 4 consecutive frames (one bracket cycle) over the *entire* route (not
just region3/4):

| | frames | keypoints/frame | per-window median depth |
|---|---|---|---|
| `b_0fps` | 1119 | 336.6 | **8.60m** |
| `ae_0fps` | 1134 | 299.4 | **4.49m** |

Bracketed's raw keypoint population sits at roughly *double* the median
depth of ae's, across the whole route -- not just in region3/4. This
matters because it means whatever's different between the two recordings
isn't confined to the saturation-heavy transition zones; it's present
throughout.

### MAE alone has the same bias as full bracketing -- rules out "mixing exposures"

Original hypothesis under test: does bracketing's own mechanism (using
3 exposures, more total landmarks, more far ones from SAE/LAE) cause the
bias? Tested directly by filtering `b_0fps`'s frame stream to MAE-slot
frames only (same recording, same route, ~half the frames since MAE
occurs at 2 of 4 cycle positions) and running the pipeline unchanged:

| | region3 ATE / scale | region4 ATE / scale |
|---|---|---|
| bracketed, full (MAE+SAE+LAE) | 1.56m / +30.0% | 2.23m / +15.3% |
| **MAE-only, no brightening** | **1.55m / +29.5%** (same) | **2.23m / +15.2%** (same) |

Essentially identical. **Bracketing itself is not the cause** -- MAE
alone, on this route, already carries the full bias. Whatever's
different about this recording's keypoint population, it isn't about
combining three exposures.

### MAE brightness vs. ae_0fps's own exposure target

Measured directly (properly region-sampled, not just the first N
sequential frames -- an initial region3-only check using a biased
sequential sample was corrected here):

| | MAE only (mean brightness) | ae_0fps (mean brightness) | ratio |
|---|---|---|---|
| region3 | 125.6/255 (49.3%) | 167.6/255 (65.7%) | 1.33x |
| region4 | 68.9/255 (**27.0%**) | 131.5/255 (51.6%) | **1.91x** |

Region4's ratio matches the "MAE targets ~25%" framing almost exactly.
Tested MAE x2 brightening (`pixel * 2`, clipped at 255) as a synthetic
re-exposure, region3 and region4, then compared directly against the
*real* `ae_0fps` recording -- first at native fps (confounded by fps,
see next section), then fps-matched:

| | region3 ATE / scale | region4 ATE / scale |
|---|---|---|
| MAE x2, half-rate (inherent to MAE-only filtering) | 1.77m / +35.3% | 2.16m / +14.1% |
| ae_0fps, half-rate (matched) | **0.76m / +13.5%** | **0.88m / +4.1%** |

Even brightness-corrected *and* fps-matched, MAE-derived frames are still
2.3-3.4x worse than ae_0fps's own frames, in both regions. Brightness
compensation is real (region4's ratio matches the physical target gap
almost exactly) but only closes a fraction of the total gap -- it is not
the primary driver.

### FPS: a real, partial, and asymmetric effect

Tested directly on `ae_0fps` alone (same recording, no cross-recording
confound): halving its own frame rate roughly **doubles** its error --
region3 ATE 0.39m->0.76m, scale +6.2%->+13.5%; region4 ATE 0.42m->0.88m,
scale +1.7%->+4.1%. FPS is a real, independent contributing factor.

But it's asymmetric: MAE-only's *inherent* half-rate (a side effect of
filtering to one slot) did **not** add meaningfully on top of bracketing's
already-elevated baseline (29.5% vs. 30.0%, no real difference) -- see
above. Read together: bracketing's bias is large enough on its own that
an additional fps cut barely moves it (a ceiling effect), while ae's own,
much lower baseline has real headroom for fps to matter. FPS is real but
doesn't explain the *difference* between the two recordings on its own --
it compounds with something else rather than being the root cause.

### Keypoint detector choice (SuperPoint vs. DISK)

Recovered SuperPoint frontend from `vision-refine-oscillation`'s history
(built + tested there on the `_10fps` pair, rejected as "clearly worse in
both regions," removed during cleanup) and re-ported here
(`config.frontend`, `configs/superpoint.yaml`) for a fresh test, flat
noise/prior (no `depth_scaled_noise`) in both cases:

| | DISK ATE / scale | SuperPoint ATE / scale |
|---|---|---|
| `b_0fps` region3 | 1.56m / +30.0% | 1.59m / +29.8% (same) |
| `b_0fps` region4 | 2.23m / +15.3% | 1.93m / +12.4% (slightly better) |
| `ae_0fps` region3 | 0.39m / +6.2% | 0.49m / +8.4% (slightly worse) |
| `ae_0fps` region4 | 0.42m / +1.7% | 0.53m / +2.8% (slightly worse) |

Unlike the earlier `_10fps` rejection, SuperPoint is roughly a wash here
-- not "clearly worse." Same pattern seen repeatedly this investigation:
an earlier verdict on one dataset pair doesn't automatically generalize.
The scale bias itself persists almost unchanged with SuperPoint too,
confirming (again) it isn't a DISK-specific artifact.

### Zero-motion prior noise: tightness only matters if the dataset actually goes blind

Tangential to the scale-bias chase, but tested in the same session:
`motion_prior.zero_motion`'s flat noise sigmas
(`zero_motion_rotation_sigma`/`zero_motion_translation_sigma`, originally
tuned loose: pi rad / 10m) were re-tested at a tighter setting (90 deg /
2m) on both dataset pairs.

On `ae_10fps` (long blind streaks, up to 12 consecutive frames): tightening
made things dramatically worse -- scale bias blew up to +150.6% in
region3 (from a much smaller value at the loose setting). On `ae_0fps`
(confirmed zero blind streaks anywhere): tightening made **no measurable
difference at all** (region3: 0.707m/+12.8% at both settings; region4:
0.700m->0.694m, +3.7%->+3.6%, noise-level only).

Confirms the mechanism directly: the loose noise setting isn't an
arbitrary choice, it's specifically load-bearing for datasets with real
blind streaks to survive. For a dataset that never goes blind, the
identity prior's tightness is nearly irrelevant, since real observations
dominate every frame regardless.

### Is it *only* a scale problem? Decomposing rigid vs. scale-corrected ATE

Computed both the rigid (SE3, no-scale) ATE and the Sim3 (scale-corrected)
ATE directly, rather than inferring from the scale factor alone:

| | rigid ATE | Sim3 (scale-corrected) ATE | scale bias | % of error explained by scale |
|---|---|---|---|---|
| `b_0fps` region3 | 1.63m | 0.29m | +31.6% | **82%** |
| `b_0fps` region4 | 2.26m | 0.74m | +15.5% | **68%** |
| `ae_0fps` region3 | 0.39m | 0.16m | +6.2% | 59% |
| `ae_0fps` region4 | 0.42m | 0.34m | +1.7% | 19% |

Mostly scale, but not *only* scale: region3's residual (0.29m) is small,
but region4 leaves a 0.74m residual after scale correction -- itself
larger than ae's entire rigid-ATE error in that region (0.42m). Scale is
the dominant, largest single component of bracketed's excess error, but a
real, non-trivial shape/rotation component remains too, especially in
region4 -- fixing scale alone would not be sufficient to bring bracketed
fully to ae's level there.

### RPE never sees any of this, at any scope

Multi-window RPE, `b_0fps` vs. `ae_0fps`, both baseline configs, computed
at region3, region4, *and* full-trajectory scope:

| scope | window | b_0fps | ae_0fps |
|---|---|---|---|
| region3 | 1m | 1.417 | 1.194 |
| region3 | 5m | 6.060 | 6.599 |
| region4 | 1m | 1.387 | 1.377 |
| region4 | 20m | 24.768 | 26.029 |
| full | 1m | 1.275 | 1.325 |
| full | 30m | 34.811 | 36.203 |

At every window, every scope: the two are essentially comparable, and
`b_0fps` is if anything slightly *better* than `ae_0fps` more often than
not. RPE never reveals any part of the large rigid-ATE/scale-bias gap
documented above, consistently across all three scopes -- the same
RPE-dilutes-a-systematic-drift pattern this project has now confirmed
repeatedly (first with `ae_10fps`'s blind streaks, now with `b_0fps`'s
scale bias): RPE's short, re-anchored-per-pair windows just don't
accumulate a global scale/alignment error the way a single rigid
alignment over a whole region does.

### Synthesis: full list of tested candidate explanations

**Ruled out** (tested directly, no measurable effect on the scale bias):
- Cross-exposure temporal keypoint-detection bias (`refine.py`'s
  mechanism) -- identical bias with/without it (`vision-refine-oscillation`
  vs. `slam-landmark-ba` comparison).
- SAE/LAE's own stereo-matching quality vs. MAE -- direct same-landmark
  cross-slot position comparison found no directional bias.
- Uniform stereo calibration/baseline error -- ruled out analytically
  (depth-selective fix wouldn't work on a flat miscalibration).
- Bracketing "mixing" three exposures / more total landmarks -- MAE alone
  shows the same bias as full bracketing.
- Keypoint detector choice (DISK vs. SuperPoint) -- bias persists either
  way.

**Real, confirmed, but partial** (contribute something, don't close the
gap alone):
- MAE's exposure target being darker than ae_0fps's own AE target --
  real, measured, but brightness+fps-corrected MAE is still 2.3-3.4x
  worse than ae_0fps.
- Frame-rate/sampling density -- real (ae's own error roughly doubles
  when halved), but doesn't explain the gap alone (see the ceiling-effect
  asymmetry above).

**Confirmed as the effective fix** (addresses the symptom regardless of
root cause): far landmarks getting flat, depth-independent
measurement/prior weighting -- `depth_scaled_noise` or `max_depth_m`
both nearly eliminate the bias; `depth_scaled_prior` doesn't help.

**Still genuinely open**: why `b_0fps`'s raw keypoint population is
farther-skewed than `ae_0fps`'s in the first place, given the same
physical route and ruling out exposure-mixing, brightness, fps, and
detector choice as sufficient explanations on their own. Untested
candidates: different driving-day lighting/weather conditions, different
vehicle speed or lane position affecting which points get matched at
what range, or a subtler exposure-specific effect on local texture
visibility that mean brightness alone doesn't capture. This investigation
identified a robust, effective *fix* and ruled out most of the obvious
*causes* -- the full causal chain from "why this route, this recording"
to "farther keypoints" remains open.
