# Surface consistency: a testable compatibility prior

Started 2026-11-08 at 07:42 UTC; requested research window ends at 12:42 UTC.
This document distinguishes a theory, an implementation and the experiment that
can test it. The completed [measured results](SURFACE_CONSISTENCY_RESULTS.md) and
[illustrated report](../output/surface-consistency-20261108/report/REPORT.pdf)
are available. The full rule remains inconclusive; the local variant has an
exploratory pooled benefit. The frozen contract below was not changed after testing.
The original proposal is preserved in [SURFACE_CONSISTENT_PMC.md](SURFACE_CONSISTENT_PMC.md).

## What the theory can and cannot identify

Let the observation be X=(I_visible,V,O,c), and let Y be the unseen target RGB.
A pretrained PMC sampler produces K candidates from q(I,M | X). A compatibility
loss L(I,M;X) is an observable prior, not a measurement of Y. Reranking changes
the decision rule, not the generator or its training distribution:

    k_surface = argmin_k L(I_k,M_k;X).

The useful hypothesis is conditional: when the visible and hidden regions belong
to a sufficiently stationary material/pattern and at least one candidate preserves
that structure, a compatibility score may predict reconstruction quality. This
requires testing. Uniform appearance is not a universal property of objects.

There is an identifiability limit. Two objects can have identical observed pixels
and different hidden colors, patterns, text or parts. Any deterministic selector
using only X and the same pool makes the same decision for both possible truths.
It cannot be guaranteed correct for both. Smoothness and surface stationarity
are assumptions, and uncertainty remains even with a perfect implementation.

Three specific failure mechanisms follow:

1. Color distributions discard spatial arrangement. A phase-shifted striped
   completion can have exactly the same histogram and mean boundary color.
2. Matching global visible appearance can reject a legitimate new material or
   semantic part behind the occlusion. A uniform copy can score better than truth.
3. A candidate-dependent hidden mask lets the score ignore difficult generated
   pixels. Missing/small-region guards help, but do not prove correct geometry.

Local seam comparisons address some spatial ambiguity. They do not resolve
unobserved material transitions, perspective, shading or repeated-pattern phase
far from the seam. These are limits to measure, not reasons to label every
discontinuity an error.

## Implementation contract

`amodal_benchmark/surface.py` accepts observed uint8 RGB, observed visible support,
an occlusion mask, candidate RGB and its predicted target mask. It has no hidden
truth argument. All arrays share one pixel frame. The hidden score support is
`H_k = M_k & O & ~V`.

Boundary compares the mean original-observation CIELAB color in V's seam band
with candidate color in H's seam band. The bands use a square radius-5 dilation
at 512 pixels; Lab uses floating-point sRGB/D65 conversion and distance/100.
This is explicitly different from the source example's 8-bit OpenCV Lab scaling.
Color uses a joint 8x8x8 Lab histogram and Hellinger distance. Texture averages
Hellinger distances between an eight-bin unsigned gradient orientation histogram
(plus a flat bin) and fixed gradient-strength bins. Features use eroded support
so gradients do not read pixels across the mask boundary.

The local variant compares original-visible/candidate-hidden pixel pairs across
the interface at distances 1..5 along either axis, weighted by inverse distance.
Its texture features come from width-16 seam bands. This is a separately declared
variant, not an assumed improvement. Candidate visible-RGB change is recorded
independently; it is not allowed to alter the original observation anchor.

Guard conditions are at least 64 visible and predicted-hidden pixels, predicted
retention of at least 50% of visible target pixels, and at least 16 pixels on both
seam sides. Missing individual terms invalidate methods requiring those terms.
All-invalid pools fall back to the first seed and retain a fallback flag. Equal
losses keep input seed order. No missing score becomes zero.

Seven fixed methods are boundary, color, texture, boundary+color,
boundary+texture, full surface, and local surface. Full weights are .40/.25/.35;
two-component weights are renormalized, and local weights are .60/.40. Each
component's positive median on development candidates normalizes its scale;
the scale is frozen before test analysis. No test-dependent weights or thresholds
are permitted. A separately reported development-selected method may choose
among those seven and the fixed-first control using development hidden MAE;
its identity is frozen before test evaluation and is exploratory.

## Frozen controlled experiment

The configuration is [surface-consistency-v1.json](../configs/research/surface-consistency-v1.json).
Ninety-six cases are selected by a fixed SHA-256 ordering from the four benchmark
sources (24 each). There is one target per source image identity; the first six
eligible cases per source are development and the next 18 are test. Selection
does not inspect historical completion quality, metrics or appearance labels.
It requires a cached visible SAM mask with at least 4,096 native pixels and
bounding-box dimensions at least 80 pixels, and adequate resized support.

The source image and its visible mask are cropped to a square whose side is 1.35
times the longer visible-mask bounding-box side, then resized to 512. A hash-chosen side is covered by a neutral half-plane,
with a threshold hiding approximately 35% of the visible target. RGB is replaced
*after* resizing, preventing resampling from leaking held-out pixels into inputs.
The actual occlusion fraction and crop are recorded. Remaining visible support
is V; held-out target support H_GT is saved only for evaluation. The principal
error mask is the pre-occlusion visible mask eroded by two pixels, intersected
with the artificial occlusion, to reduce SAM boundary contamination.

This tests recovery of **artificially hidden, originally observed target pixels**.
It does not declare the original object fully visible or claim complete-object
amodal ground truth. Native SAM contamination can remain. Crop construction uses
pre-occlusion support, a declared shared experimental prior. No output-based
exclusions or replacement of difficult cases are allowed.

Generation uses the pinned SD2 Mixed Context pipeline at step 35 of 50, the
existing LaMa background construction, the annotation query as prompt, and
explicit CUDA generators with seeds [11,22,33,44]. Each of four workers owns
one allocated Quadro RTX 6000. Grounding DINO queried with the target label and
SAM supply candidate masks; visible-mask IoU selects the target instance.
Ordering, initial occluder prediction and the progressive loop are bypassed
because the artificial occluder and shared one-step state are already known.
The artificial half-plane is passed directly without `run_iteration`'s three
5x5 occluder-dilation passes. This makes the removed observation support explicit
and identical for every candidate; it is a declared difference from full released
preprocessing, not a reproduction of that end-to-end path.
The released Mixed Context function is called directly; no diffusion retraining
or denoising modification is made. This is a mechanism experiment, not a new
full Open-World scored benchmark run.

Inputs and generation manifests are separate from `evaluation_only/`. The GPU
generator does not load held-out RGB or masks. Candidate PNGs, mask PNGs, raw
scores, seed records, failures, runtimes, peak allocated VRAM, input hashes,
source hashes, upstream pipeline hash and package versions are preserved.

## Evaluation and explanation

The primary metric is RGB MAE/255 over H_GT core, measured on **every** hidden
ground-truth pixel, regardless of the candidate's predicted mask. A missing
generated output receives error 1. This prevents low-error statistics from
silently excluding pipeline failures or shrinking the evaluation support.
Visible RGB MAE, held-out mask recall and available independent perceptual
metrics are supporting diagnostics. No metric on this synthetic subset proves
quality on true occlusions or on the full progressive benchmark.

Primary control is the exact expected error of uniform random selection from
the same four candidates, computed as their mean error. This removes random
control-seed noise and gives identical generation cost. Fixed seed 11 is the
second control; within the same one-step protocol it corresponds to K=1 output.
An oracle choosing minimum hidden MAE is an unavailable-at-inference ceiling,
never a deployable method or a selection input.

For a pool with errors e_1..e_K, define:

    selection_gain = mean(e_k) - e_selected
    available_gain = mean(e_k) - min(e_k)
    oracle_regret = e_selected - min(e_k)

Thus `selection_gain = available_gain - oracle_regret`. A poor result can be
explained by insufficient candidate quality/diversity, an uninformative score,
or both. Gain can be negative. Per-case score components, predicted areas and
counterexamples explain *why* a rule makes its choice; they do not establish
a causal generative mechanism.

Report development and test separately, all 72 test cases, per-source summaries,
paired image-bootstrap 95% intervals, win/tie/loss counts, fallbacks and errors.
The full-surface comparison against uniform selection is the primary hypothesis;
the ablations and selected-development variant are exploratory. A confidence
interval crossing zero is inconclusive, not proof of equivalence. Qualitative
panels include deterministic representative gains and losses, all four candidates,
observed inputs, held-out truth, scores and masks. Independent human preference
remains a separate unperformed evaluation unless real responses are supplied.

## Prespecified replication and supporting checks

At 08:00 UTC, before inspecting pilot test errors, a second image-disjoint cohort
was declared. The intended 24 additional cases per source exhausted eligible
LAION support after 15 cases. This was amended before test analysis to **87
replication cases**: 24 VG, 24 COCOA, 24 free-images and 15 LAION. Eligibility,
scoring, seeds, generation and metrics remain identical; no thresholds were
relaxed. The incomplete 96-case preparation is retained and was never submitted.
The replication has no development split and must reuse the pilot's frozen scales
and selected development method. Report its own result as well as a pooled
159-test-image estimate; the latter is weighted by cohort case counts, not by
the official benchmark's source proportions. Predeclaration, amendment and
excluded-image hashes are retained with the artifacts.

Supporting perceptual evaluation uses LPIPS AlexNet v0.1 in spatial mode. Both
truth and candidate are set to neutral RGB 127 outside H_GT core; the spatial
distance map is averaged only on H_GT core in the native 512 frame. Mask edges
affect feature receptive fields; this declared diagnostic is not a direct human
preference metric. It is never used to tune or select candidates. A missing RGB
has no LPIPS value; perceptual comparisons must report common complete-pool
denominators instead of silently changing coverage by method.

An exploratory audit also scores the 1,390 previously saved expanded-vocabulary
outputs against their pre-existing single-assistant labels. Inputs are aligned
with recorded integer geometry, cropped to the union of visible/predicted support
plus a 10% square margin and nearest-resized to 512. Here the unknown support is
the complement of visible support, so expansion is not true occlusion. Report
validity coverage, missing support, descriptive AUC/ordinal association and a
confound-filtered sensitivity analysis. These outputs offer one candidate each
and no hidden RGB truth; this audit cannot establish a reranking improvement or
independent human agreement. It does not tune the frozen controlled experiment.

Before inspecting pilot test errors, a simple observable preservation control
was also declared: select minimum candidate RGB MAE on the original visible
support, using the same mask-validity guards and fixed-first fallback. This
checks whether the surface terms add value beyond preserving known pixels.
It is a supporting control and does not change the previously specified
development method-selection menu. The declaration timestamp is retained.

A second diagnostic control, also declared before test errors, uniformly selects
among exactly the candidates with valid full-surface terms (falling back to seed
11 when none is valid). Comparing full surface with this control separates the
mask-validity filter's contribution from the ranking loss's contribution. The
original primary comparison against all-four uniform selection remains reported.
