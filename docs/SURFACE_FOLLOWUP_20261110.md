# Surface-PMC improvement study, 10 November 2026

Latest completion,11 November12:01 UTC: the full progressive runs and all final-RGB
diagnostics are finished and audited. Full corrected CLIP/LPIPS/VGG/SSIM gains are
1.123/0.835/2.789/1.782%, below3%. Raw-final correction removal worsens three
reconstruction metrics. Separate raw-pool oracle ceilings also miss3% for CLIP,
LPIPS and SSIM; this rules out final raw RGB reranking alone on existing fixed
masks/trajectories. Read the [combined diagnostic report](../output/surface-lastpass-review-20261111/REPORT.md).
The independently confirmed128-image raw selector remains successful within its
controlled artificial-occlusion scope. Earlier execution notes below are historical.

04:17 UTC continuation: the unchanged selection-only component is now being
independently tested on 128 additional image-disjoint VG/COCOA cases. Protocol,
sources, ranker and seeds were frozen before generation; Slurm **37843** is
queued. See [the confirmation record](SURFACE_CONFIRMATION_20261110.md). This
does not alter the primary/secondary interpretation below or the full
progressive run. No confirmation results exist yet; the deadline has passed.

Latest measurement (2026-11-10 02:55 UTC; native replay passed at03:13 UTC):
the frozen method improves hidden MAE by **9.09%**, object LPIPS by **12.56%**,
DreamSim by **18.55%**, and CLIP cosine by **2.42%** on all 166 fresh images.
The corresponding paired 95% intervals are [6.44, 11.91], [9.52, 15.64],
[14.57, 22.87], and [1.93, 2.95] percent. No generated candidates failed.
**The all-metric 3% objective is not met: CLIP is below the target.** This is
controlled artificial occlusion, not the full progressive benchmark.

Job37785 completed scoring but stopped during exact primitive-score replay on
Quadro node gebru. CPU dispatch changes small floating-point RGB-to-Lab values
near hard histogram boundaries; merely pinning NumPy/Pillow is insufficient.
The original generation node gus reproduces all1,328 candidate scores exactly
under the pinned environment. CPU-only Slurm37799 passed all166 image decisions,
selected RGB/masks and3,984 hidden-MAE checks; unsuccessful floyd replay37798 is
retained. No tolerances were relaxed, scores replaced, or parameters changed.
Four-GPU37812 now runs the final progressive smoke followed by the matched full
pair; CPU37814 will verify, compare and register that pair. Old37795 was superseded.

The pre-specified selection-only component (`trained_surface_raw`) has relative
gains MAE5.845% [2.920,8.873], LPIPS6.754% [3.751,9.843], DreamSim12.764%
[8.639,17.190], and CLIP3.075% [2.630,3.568]. All four point estimates exceed3%,
but MAE/CLIP lower intervals do not. This is an already measured secondary
component using the exact same frozen choices, without correction. Do not
replace the frozen primary after seeing it; a newly preferred primary would
need independent confirmation. The gallery shows raw and corrected selections.

The authorized research window is 2026-11-09 18:07 through 23:07 UTC (local
10 November 01:07 through 06:07). Target: 3–5% relative improvement in hidden
RGB MAE and relevant perceptual metrics. This is an objective, not a result.

The earlier 183 cases are now development data for this follow-up, including
their previously published test cases. Their historical claims and frozen files
remain unchanged. A new image-identity and image-byte-disjoint cohort supplies
the test. New method parameters are frozen before its truth metrics are opened.

Hypotheses under development:

1. Local surface scores plus visible preservation can rank shared candidates.
2. A regularized small ranker can learn this combination from development pools.
3. Visible RGB residuals can calibrate a generated surface's low-frequency color
   while retaining generated gradients and texture. This uses only observations.

All candidate policies receive observation RGB, visible mask, occlusion mask and
generated RGB/mask. Hidden truth and its support enter evaluation only. Fixed
hidden-core MAE includes pixels missed by predicted masks. Uniform selection
and seed 11 remain controls; failures and denominators are explicit. Paired,
source-stratified image bootstrap intervals accompany relative point gains.

The existing Open-World protocol uses CLIP text logits, LPIPS, VGG feature
similarity and SSIM. The original PMC paper additionally uses DreamSim and CLIP
cosine. These are distinct protocols; metric names and references must remain
explicit. Artificial occlusion results are not the full progressive benchmark.
The original PMC paper (local `CS331_paper.pdf`, p.6, Metrics paragraph) specifies
category-text CLIP cosine and complete objects placed on **black** backgrounds
for DreamSim/LPIPS. The new controlled metric script follows that background
convention. The established Open-World white-background scoring remains unchanged.
Any full-benchmark postprocessing comparison retains its original references,
geometries, failures and weights, and records its cache ancestry.

The accepted baseline, old studies and unrelated cluster jobs are preserved.
No detector vocabulary or reference-protocol change counts as a surface gain.

Development update, 18:44 UTC: the 183-image observation-only consensus-feature
ranker reaches 7.54% MAE and 5.34% hidden LPIPS cross-validation gains. Bounded
affine calibration adds about 0.5 percentage points of development MAE gain.
These estimates are used for model selection, not as held-out evidence.
The provisional policy is saved before complete-object development measurements;
no fresh evaluation truth has been opened. Development metrics run in steps
37566.2 (96 images) and 37566.3 (87 images). Observation-only deep features run
in step 37566.4, alongside generation inside the existing four-GPU allocation.
The metric smoke passes, including DreamSim, after isolated dependency setup.

Additional candidate features compare the generated and observed **visible**
pixels with spatial LPIPS and DreamSim, compare generated versus observed partial
objects, and measure category-text CLIP cosine. They receive only the observation
manifest, observed RGB, visible support and generated candidates. The small ridge
ranker may use these pretrained features; the generator and perceptual networks
are frozen. Multi-metric model selection uses source-held-out development folds
and a balanced MAE/object-LPIPS/DreamSim/CLIP objective. Fresh-test decisions and
parameters will be frozen before test measurement.

18:50 UTC design extension, still before any fresh truth metrics: four-candidate
development CLIP headroom is only 3.15% even with evaluation's best calibrated
CLIP choice, which hurts MAE. Add seeds 55,66,77,88 under the same generation
settings for all 183 development and 166 fresh images (job 37568, dependency
37566). The primary candidate budget becomes eight; record four-candidate
results as a budget ablation. All methods will use common pools, and semantic-only
selection remains an explicit control. This extra sampling is not attributed
to surface scoring. Additional development images or fresh-test tuning are not
introduced by this extension.

A separate full-run calibration policy was frozen at 18:50 UTC in
`full-calibration-freeze.json`: affine visible residual, strength 1, maximum
channel shift 32, exact visible-pixel restoration, unchanged generated alpha.
It derives the accepted 951 completions and preserves 1,614 failures. This full
comparison does not use the candidate ranker and cannot establish hidden MAE
without hidden RGB ground truth.

Observation-only feature audit passes on four development images: randomizing
every unobserved input pixel has exactly no effect on any extracted feature;
identical visible appearances have zero distance; DreamSim cached-embedding
distance agrees with its official forward method. A further developmental
surface-consensus correction transfers only bounded, smooth RGB residuals from
mask-supported candidate medians while retaining each chosen mask and texture.
Its four CPU invariance/bound/shape regressions pass; efficacy remains unmeasured.

## Full cached calibration result (19:05 UTC)

All 2,565 targets pass provenance-required artifact verification and are recorded
in the separate design experiment ledger. The 951 saved outputs and 1,614
failures are unchanged in count, geometry and alpha. Baseline and derived RGB
were scored together on the same L40S runtime with the existing Open-World
reference/metric protocol. Source-image cluster bootstrap retains correlated
targets together.

| Metric | Paired baseline | Calibration | Relative gain | 95% interval |
| --- | ---: | ---: | ---: | --- |
| LPIPS, n=2,565 | 0.735009 | 0.732160 | +0.388% | [+0.340%, +0.439%] |
| VGG feature similarity, n=2,565 | 0.242636 | 0.245108 | +1.019% | [+0.908%, +1.127%] |
| SSIM, n=2,565 | 0.303397 | 0.306268 | +0.946% | [+0.770%, +1.128%] |
| CLIP logit, n=951 | 28.393099 | 28.298934 | -0.332% | [-0.484%, -0.177%] |

**This approach does not meet the requested 3-5% full-benchmark objective.**
Do not promote it as a successful result or change its frozen calibration after
inspection. There is no hidden ground truth in this real-occlusion protocol,
so no hidden MAE claim is possible here. The subsequent controlled method is
selected exclusively on the designated old development images.

An observation-support audit finds mean original-visible recall 0.954 across 951
outputs, with 82 below 0.9 and several near zero. All original visible support
lies within the saved completion frame. This documents remaining segmentation/
identity errors; it does not justify changing masks or deleting hard cases in
the frozen calibration experiment.

19:21 UTC design update, before fresh-test measurement: the final eight-pool
model-selection utility will standardize development source means to the known
new cohort counts (64 VG, 64 COCOA, 38 free images, zero LAION). All 183 old
images remain available to fit source-held-out rankers; LAION contributes no
weight to the deployment utility because its fresh eligibility is exhausted.
The unweighted development search remains an explicit sensitivity analysis.

Four-pool consensus correction on out-of-fold choices improves development MAE
from 0.165210 to 0.157371 (strength 1, sigma 4), but reduces CLIP cosine from
0.287249 to 0.285628. LPIPS and DreamSim also improve slightly. This informs
development only. The primary eight-pool policy is not yet frozen.

An opt-in progressive runtime bridge and four focused CPU regressions now exist.
It anchors every pass to the original measured support using exact crop geometry,
explicitly seeds each candidate, and retains the upstream stopping procedure.
It has not yet received GPU validation and is not a demonstrated full-run gain.

19:27 UTC: eight-pool development will measure the predeclared consensus variants
(strength/sigma 0.5/4, 1/4, 1/12) for every candidate, allowing joint correction
and ranker selection with the same fixed balanced utility. These measurements
are restricted to the old 96/87 cohorts. Fresh-test analysis includes equal-budget
controls receiving the identical correction, to isolate selection from correction.
Runtime feature-provider audit passes all eight compared candidates exactly
(maximum absolute feature difference zero), without evaluation-data access.

19:30 UTC selection clarification, still before fresh measurement: feasible
development configurations (all four primary relative gains at least 3%) rank
ahead of infeasible configurations; the previously declared balanced utility
orders candidates within either group. This directly reflects the user target
and prevents a large DreamSim gain from outweighing a missed MAE/CLIP threshold.

19:34 UTC: a Quadro four-GPU progressive smoke test-only Slurm estimate is
23:16 UTC, after the work window. No job was created by --test-only. The smoke
will instead follow the current four-GPU controlled experiment. A provisional
development-only four-pool smoke policy is saved separately and cannot authorize
fresh-test evaluation.

Pre-test decoded-image split audit: 183 development and 166 fresh images have
no exact decoded cross-split or within-fresh duplicates. No aligned near-copy
flags pass the predeclared aspect/thumbnail test; minimum cross-split thumbnail
MAE is 0.1365 versus the 0.005 threshold. This is an identity audit, not generated
image evaluation, and does not certify absence of cropped or transformed clones.

19:45 UTC integration update: progressive smoke in allocation 37568 failed during
DreamSim initialization because DINO and InstaOrder both import top-level utils.
The runtime now temporarily isolates DINO imports and restores the original
module cache even on failure. Six focused CPU regressions pass. Original failed
artifacts remain; corrected smoke runs in pdmc-surface-progressive-smoke-v2-20261110.
Fresh four-pool download is complete, extracted and independently hash-verified
(b9d7c71f9f2bb550741fe8d0eda81081ae013657829dc156295c116053d6a7cf).

19:52 UTC: progressive development smoke v2 and a separate matched-runtime
L40 control smoke both finish and pass provenance verification. Each has the
same eight targets and one saved completion. Surface generates four candidates
in the true progressive sampler and scores its selected output. This tiny smoke
establishes runtime behavior only. Full comparisons will use a newly generated
L40 control with identical runtime and frozen reference bytes.

19:54 UTC pre-test model-family extension: compare a limited nonlinear ranker
grid on the same old development images: 100 gradient-boosted depth-2/3 trees,
minimum leaves 24/48, learning rate .05, and semantic weights 0/2/4/8. All
features and targets are centered within each candidate pool. Use the same
leave-source-out folds, source weights and feasibility-first balanced utility.
The portable export has no sklearn inference dependency. This extension is made
before fresh metric evaluation; no new test outcomes have been examined.

2026-11-10 02:37 UTC continuation: the original five-hour window has elapsed.
Job37568 completed successfully at 2026-11-09 20:43:57 UTC (cluster timestamp
15:43:57 EST), including all extra candidates and fresh observable features.
The final model had not been frozen and fresh quality metrics had not been
computed. Continuing under the explicitly active goal to finish evaluation;
any final improvement must NOT be described as completed within five hours.
No new method families are added after this continuation.

2026-11-10 02:42 UTC: final eight-candidate method frozen before fresh scoring.
The predeclared utility selects ridge100, all-appearance loss plus CLIP weight16,
affine correction strength1, and consensus strength0.5/sigma4/maxshift32.
Source-weighted leave-source-out development gains: MAE11.415%, LPIPS14.378%,
DreamSim19.042%, CLIP3.150%, hidden spatial LPIPS6.199%. These are development
estimates; no fresh quality metrics have yet been read. The nonlinear tree grid
did not win. Final-method-freeze.json is immutable and includes source snapshots.
Four Quadro GPUs are available now; controlled metrics and matched full baseline/
surface will run on the same allocation and explicit library/runtime configuration.
