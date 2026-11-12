# Surface consistency: measured findings and revised theory

8 November 2026. This study implements the retained Surface-PMC proposal as a
one-step candidate-selection experiment. It does not establish an improved full
progressive benchmark method. The primary full-surface result is inconclusive.

Read the [illustrated report](../output/surface-consistency-20261108/report/REPORT.pdf),
[complete report text](../output/surface-consistency-20261108/report/REPORT.md),
or [all 159 held-out cases](../output/surface-consistency-20261108/report/gallery/index.html).
The [experimental contract](SURFACE_CONSISTENCY_EXPERIMENT.md),
[derivations](SURFACE_CONSISTENCY_THEORY.md), and
[reproduction commands](../cookbooks/surface-consistency.md) distinguish the frozen
experiment from subsequent explanations.

## What was actually tested

The pretrained PMC sampler generated four common-seed candidates for each of 183
image identities: 24 development, 72 pilot test and 87 image-disjoint replication
cases. All 732 candidates were generated. Development scales and the method menu
were frozen before inspecting test errors. There was no model training.

Artificial occlusion hides about 35% of originally observed target pixels. Every
selector receives only the occluded observation and legitimate candidate masks.
Held-out source RGB evaluates the decisions on a fixed target core, including
pixels missed by a candidate mask. The control is the exact expected error from
uniformly choosing one of the same four candidates. Source-stratified, paired
image bootstrap intervals use 5,000 resamples.

## Measured results

Positive gain means lower normalized hidden RGB MAE than uniform selection.
The intervals below are nominal 95% intervals; component comparisons are exploratory.

| Rule | Pooled MAE | Pooled gain [95% CI] |
| --- | ---: | ---: |
| Uniform selection | 0.169857 | 0 |
| Full boundary + color + texture | 0.165547 | +0.004310 [-0.000512, +0.009074] |
| Local seam variant | 0.162670 | +0.007187 [+0.002183, +0.012296] |
| Visible-pixel preservation | 0.164305 | +0.005552 [+0.000499, +0.010812] |

The local point estimate is a **4.23% relative reduction** in hidden RGB error,
about **1.83 intensity levels** on a 0-255 scale. This is an exploratory pooled
result, not evidence that local surface is superior to every alternative.

| Cohort | Full-surface MAE gain [95% CI] | Local-surface MAE gain [95% CI] |
| --- | ---: | ---: |
| Pilot test, 72 images | +0.002972 [-0.004722, +0.011277] | +0.011096 [+0.003282, +0.019391] |
| Replication, 87 images | +0.005418 [+0.000044, +0.011186] | +0.003951 [-0.002079, +0.009743] |

The replication full-surface interval narrowly excludes zero; the pooled interval
includes zero. The local MAE gain is smaller and inconclusive in replication.
Report both cohorts rather than treating the pooled result as two independently
significant confirmations.

Masked spatial LPIPS provides supporting perceptual evidence on all 159 test
pools. Full-surface gain is +0.008947 [-0.001027, +0.019221]; local-surface gain is
+0.018738 [+0.009265, +0.028777]. Local LPIPS intervals exclude zero in both cohorts.
This metric masks both RGB images outside the held-out core; feature receptive
fields still interact with the mask boundary. It is not human preference.

Direct paired comparisons remain inconclusive: local minus full improvement is
+0.002876 [-0.002400, +0.008486], and local over visible-pixel preservation is
+0.001634 [-0.005020, +0.008491]. A difference in whether separate intervals cross
zero is not itself evidence that the methods differ.

## What explains the result

1. **The appearance prior can disagree with truth.** In 51/159 cases, the full
   score rates the erroneous selected RGB as more compatible than the actual
   source RGB even when both use the same original support mask. This post-hoc,
   evaluation-only diagnostic removes changing mask support as an explanation
   for that preference. It does not independently identify which semantic or
   material transition caused each failure.
2. **Global compatibility is a weak ranking signal here.** Full-surface pair
   ordering accuracy is 0.532 [0.493, 0.569] across 157 eligible image pools;
   handcrafted texture alone is 0.495 [0.455, 0.533]. Chance ordering is 0.5.
   Pair ordering and best-of-four decision gain weight errors differently.
3. **Filtering and ranking are distinct.** The exact full-surface gain is
   `0.004310 = 0.000731 validity-filter gain + 0.003579 appearance-ranking gain`.
   Ranking gain alone has interval [-0.001150, +0.008247]. The available oracle
   headroom is 0.031060, leaving full-surface regret of 0.026750. Most of that
   regret comes from ranking, rather than excluding the best candidate.
4. **Lower pixel error does not guarantee correct structure.** The report shows
   a boat completion with lower error but a conspicuous added protrusion, an
   incorrect dark bowl favored by global color matching, a white ceramic
   surface where appearance ranking helps, a pose ambiguity with little oracle
   headroom, and an all-invalid pool that explicitly falls back to seed 11.

These findings support a narrower hypothesis: nearby evidence can sometimes help
choose a surface continuation, while global same-appearance assumptions can fail
at part/material changes and under shape ambiguity. Constructed stripe and
material-transition examples establish those failure mechanisms, not their
natural-image frequency. The triangle-inequality argument in the theory note
shows why identical observations cannot uniquely determine every hidden surface.

## Reproducibility and next decision

Both cohorts pass exact replay of prepared pixels, raw scores, metrics, policy
decisions and summaries in the pinned scorer environment. All downloaded
experiment archives match their independently obtained cluster checksums.
All eight repeated smoke RGBs and masks match the main run exactly. The CPU
suite passed 212 tests, all five Bash regressions and synthetic smoke.

A separate numerical-portability audit found one changed boundary-plus-color
pilot decision under another NumPy runtime. Full and local decisions are unchanged
in both cohorts. Frozen scores and decisions were preserved. Exact scorer replay
uses Python 3.10, NumPy 1.23.5 and Pillow 9.5.0; the original development-analysis
runtime is recorded separately in its freeze file.

Do not deploy the full rule as a demonstrated improvement. Preserve the fixed
local rule as a promising candidate for a new, preregistered study against
visible-pixel preservation, with real human preferences and explicit material
transition/shape checks. A learned uncertainty model would need separate training
and new test data; these results do not establish such a model. No further GPU
job is required to reproduce the reported saved-artifact analysis.
