# Surface-PMC with annotated vocabulary: frozen transfer experiment

The user designated annotated vocabulary as the team's baseline on 12 November
2026. The work window is 06:46–11:46 UTC (13:46–18:46 Asia/Saigon). This changes
the comparison population/protocol; it does not replace earlier released-vocabulary
results. No positive outcome or 3% success threshold is assumed.

## Research question and intervention

Does the existing trained, corrected, eight-candidate progressive Surface-PMC
pipeline improve appearance scores relative to single-candidate PD-MC when
**both** use the annotated detector vocabulary? Both arms use semantic matcher
v2, released boundary checks, automatic background sizing, seed 0, inclusive
maximum iteration 5, Mixed Context timestep 35, pinned model weights, and the
same metric environment and Quadro RTX 6000 GPU model. The isolated CUDA 11.8
cuFFT library already validated in the previous study is used by both arms.

The generator and feature networks remain pretrained and frozen. The 22-feature
ridge ranker, normalization, eight candidate seeds, affine/consensus corrections,
original-visible restoration, and progressive stopping rules transfer unchanged
from the final 10 November method. No new fitting is needed to isolate transfer
to annotated vocabulary. Previous artificial-occlusion claims are not repurposed
as evidence for this natural-occlusion experiment.

The adapter now permits annotated vocabulary and explicit frozen cohorts.
Candidate segmentation captures the original target-only segmenter before the
annotated scene wrapper is installed. This preserves the original selection
mechanism: the scene wrapper otherwise ignores the requested target-only query
and detects every category. Initial detection and progressive occlusion checks
continue using the annotated vocabulary in both arms. The old source and freezes
remain in prior provenance; a new parent-linked method freeze records the repair.

## Cohort and decision rules

The first analysis covers **192 source images / 204 targets**: VG 107 images /
109 targets; COCOA 58/67; free_images 14/14; LAION 13/14. All associated targets
are retained. Source quotas are proportional to eligible image counts, rounded
by largest remainder. Images are ordered by SHA256(`20261112:source/image_id`).

Exclude all 477 images from earlier controlled development, test and confirmation
studies by identity and source bytes. Audit decoded RGB and aligned 64-pixel
thumbnails. A duplicate flag uses aspect within 1% and normalized RGB MAE ≤0.005;
reject flagged sampled images and refill by the same deterministic rule. One
near-copy (VG/2316160 versus prior VG/1159611) was excluded before generation.
No completion outcomes, target success statuses, or metric values enter sampling.
The final audit has no exact/near cross-study flags or within-cohort duplicates.
This screen cannot rule out every transformed or cropped duplicate.

The primary endpoint is failure-inclusive LPIPS on **all 204 targets**. Report
paired absolute and relative gain and a two-sided nominal 95% source-stratified,
image-cluster bootstrap interval (5,000 resamples). Positive LPIPS gain is
`(mean_control − mean_surface) / mean_control`; positive gains for similarity
metrics reverse the numerator. Evidence for a primary improvement requires its
absolute interval to exclude zero in the favorable direction; report inconclusive
or adverse findings unchanged. There is no interim efficacy stopping or retuning.

CLIP, VGG feature similarity, SSIM, coverage and failure transitions are secondary.
CLIP uses shared successes and reports excluded pairs, alongside separate overall
success-only means. LPIPS failures score 1; VGG/SSIM failures score 0. Keep all
upstream detection, matching and completion failures. Secondary intervals are
nominal, not simultaneous claims. Per-source results and shared-success metrics
are diagnostics, not substitute primary denominators.

The full 2,565-target continuation is separately reported and includes training-
overlapping images; it is an operational full-benchmark result, not wholly unseen
generalization. A pre-recorded image-identity/source-byte exclusion inventory
supports a separate unseen-image sensitivity analysis. The near-copy screen was
applied to the primary sampled cohort, not exhaustively to this larger inventory.

## Comparability, resource use and limitations

Copy the historical annotated SAM references into each new run, verify the copies,
and require regenerated detector masks to match. Reference bytes, input bytes,
checkpoint hashes, configurations and package inventories must match between
arms. Evaluate the cohort first; preserve independently verified immutable copies
before continuing the same scientific configuration over the full inventory.
The coordinator forwards the cohort to all four workers while annotation queries
continue using all official annotations for each image.

Surface-PMC generates eight candidates per progressive pass; K=1 is the team's
operational baseline. This estimates the complete method-plus-sampling effect.
It does **not** isolate the causal benefit of the selector at equal compute.
Report candidate/iteration counts, GPU allocation and wall times. Natural hidden
RGB is unavailable, and these metrics compare against visible SAM references;
they do not establish hidden-surface reconstruction accuracy or human preference.

All runs request four GPUs and four workers. Full inference may exceed the work
window and the cluster's 24-hour allocation limit. Preserve valid jobs and resume
only unchanged configurations and output directories. A dependent finalizer
verifies full provenance before comparison and registration. No partial or cohort
run is registered as a full scored experiment.

Artifacts live under `output/surface-annotated-20261112-design/` and
`output/surface-annotated-20261112/`. The design freeze, exact submissions, job IDs,
GPU validation, cohort audit, paired target records and recovery hashes provide
the evidence trail. Measured findings will be added after the frozen runs finish.
