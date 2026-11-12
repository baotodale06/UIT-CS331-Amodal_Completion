# Validated project findings

Updated 2026-11-11.

## Insufficient fixed-output final raw RGB headroom (11 November)

The final-pass ablation retains2,565 targets/958 outputs. Removing RGB correction
and native restoration improves CLIP0.293% versus corrected but worsens LPIPS0.240%,
VGG0.650%, SSIM0.683%. It does not reproduce the artificial-occlusion benefit on
this full natural benchmark. Original-score replay drift is at most1.79e-7.

All7,664 final raw candidates were scored with the same fixed corrected masks,
geometry, preceding trajectories and failures. Separate reference-informed oracle
choices yield only CLIP2.817%, LPIPS0.740%, SSIM1.407% versus matched control;
VGG reaches3.257%. These are different outputs, not a joint attainable result.
Final raw RGB selection on these saved pools cannot meet all four3% full targets.
This does not establish limits for new generation, candidate masks, earlier
selections or changed trajectories. All results pass independent numerical audit.

With1,607 failures fixed, corrected successful-output LPIPS is0.273977. Full gains
of3%/5% require saved means0.231366/0.192009, another15.55%/29.92% reduction from
corrected. Failure floor:0.626511. This is denominator arithmetic, not proof that
surface-consistent generation cannot improve the full benchmark. See
output/surface-lastpass-review-20261111/REPORT.md.

## Completed full progressive result (11 November)

The corrected eight-candidate full method improves all four paired metrics but
misses 3%: CLIP 1.123%, LPIPS 0.835%, VGG 2.789%, SSIM 1.782%. Its saved-output coverage
is958/2,565 versus 951/2,565 for the matched control. Source inputs/references,
metric operations and installed packages match; generation compute differs.

Only966 targets reach surface sampling;1,599 stop earlier (877 label mismatches,
251 ambiguous instances,57 no detections,69 small masks,1 unknown class,344 no
occlusion detected). Shared-success LPIPS improves 4.165%, but all-target LPIPS
improves 0.835% after the required failure accounting. Neither coverage filtering
nor the controlled confirmation can turn the full result into a 3% success.

The runtime always restores native visible pixels after resizing; the independent
raw-selection confirmation does not. A new fixed-mask final-pass RGB diagnostic
tests removing those final corrections using saved candidates. It does not undo
their effect on earlier progressive passes or segmentation. Outcomes are pending.

## Independent selection-only confirmation (2026-11-11)

On a pre-generation-frozen 128-image VG/COCOA cohort excluded from all 349 earlier
images, the unchanged eight-candidate ranker improves hidden MAE 7.644%, object
LPIPS 11.328%, DreamSim 16.766%, CLIP cosine4.068% versus expected uniform selection.
All four individual nominal 95% lower bounds exceed 3%; this is not a simultaneous
confidence band. Zero candidate failures. The selector returns raw generated
RGB/masks; no calibration, consensus correction or visible restoration is applied.

Against CLIP-only selection it improves MAE 4.376%, LPIPS 6.723%, DreamSim 11.492%
but loses CLIP 0.632%. CLIP is an explicit selector input. This supports a controlled
reconstruction benefit beyond semantic-only ranking, not universal metric dominance.
SSIM improves 1.668%, below the target. One-step artificial occlusion does not
establish full progressive or original-paper benchmark improvement. Full corrected
progressive continuation 39387 remains separate and pending.

## Fresh controlled surface-selection evidence (2026-11-10)

A separate 128-image confirmation of the unchanged selection-only component
was frozen before generation at 04:12:48 UTC. It excludes all 349 prior images;
861 unused images pass native eligibility (629 VG, 232 COCOA). The selected
64+64 cohort passes decoded identity and aligned near-copy audits. This is an
eligibility/protocol finding, not another measured improvement. Job 37843 is
queued; see `docs/SURFACE_CONFIRMATION_20261110.md`.

The final ranker was frozen before scoring 166 new image-disjoint cases. Against
uniform selection from the same eight candidates, selection only reduces MAE
5.845%, object LPIPS 6.754%, and DreamSim 12.764%, while CLIP increases 3.075%.
These are pre-specified secondary point estimates; MAE and CLIP lower confidence
bounds are below 3%. The primary corrected outputs improve reconstruction more
(9.086%, 12.560%, 18.553%) but CLIP gains only 2.422%. Exact replay and report
verification pass. No full progressive improvement is established yet. See
[the complete controlled report](output/surface-followup-20261110/START_HERE.md).

## CPU dispatch matters for exact surface-score replay

The fresh L40S-generation artifacts use NumPy1.23.5/Pillow9.5.0 on AMD node gus.
Replaying them on other CPUs can move float32 Lab values across hard histogram
boundaries, even with the same Python packages. The first saved case reproduces
all primitive scores exactly on gus. Full native replay 37799 now passes all
166 images, 1,328 candidates, selected RGB/masks and 3,984 MAE checks. Preserve
the strict replay tolerance and record hardware/runtime differences separately.
User-site packages must also be disabled for research commands; the login
account otherwise supplies a different NumPy/Pillow. This is a reproducibility
finding, not a reason to modify the frozen model or its reported test outcomes.

## L40S runtime compatibility, 2026-11-10

The pinned torch 1.13.1/CUDA 11.7 environment's cuFFT 10.7.2 fails LaMa on L40S.
An isolated `nvidia-cufft-cu11==10.9.0.58` library, loaded using LD_PRELOAD,
passes the real FFT test and eight complete PMC+SAM candidate generations in
job 37565. It does not overwrite the original library. Cross-runtime smoke RGB
is not byte-identical (normalized mean absolute differences approximately
0.00064–0.00394); fresh method comparisons must share the same candidate pools.
This is measured compatibility evidence, not an assumption of numerical parity.

## Surface-PMC measured findings (2026-11-08)

- The completed [159-test-image study](docs/SURFACE_CONSISTENCY_RESULTS.md) uses
  732 generated candidates including development, with frozen normalization and
  a separate image-disjoint replication. The full boundary/color/texture rule
  has pooled hidden-MAE gain +0.004310 [-0.000512, +0.009074], inconclusive.
- The local seam variant has exploratory pooled gain +0.007187
  [+0.002183, +0.012296], or 4.23% relative error reduction. Its replication MAE
  interval crosses zero; local LPIPS improves in both cohorts. Direct local
  superiority to full surface or visible-pixel preservation is not established.
- Full compatibility prefers the erroneous selected RGB to actual source RGB
  on identical mask support in 51/159 cases. This evaluation-only diagnostic
  exposes a limitation of the prior; it does not supply hidden truth at inference.
- Full gain decomposes exactly into +0.000731 from validity filtering and
  +0.003579 from ranking. Remaining oracle regret is 0.026750, mostly ranking
  regret. Full pair ordering is 0.532 [0.493, 0.569]; texture alone is near chance.
- Both cohorts replay with zero raw-score difference in NumPy 1.23.5/Pillow 9.5.0.
  A separate NumPy 2.5.4 audit changes one pilot boundary-plus-color choice;
  full/local choices are unchanged. Do not confuse a portability rescore with
  the original frozen experiment or relax tolerances to conceal a difference.

## Surface theory and implementation facts (2026-11-08; not quality results)

- Surface compatibility is an observable prior, not hidden-RGB truth. Identical
  visible evidence can admit different legitimate hidden surfaces; no such
  selector has a universal reconstruction guarantee.
- The new scorer anchors to original visible RGB. Focused tests show its surface
  terms cannot improve by repainting candidate visible pixels or by changing
  source RGB outside the visible mask. This establishes a software property,
  not empirical completion quality.
- The controlled pilot hides originally visible SAM-supported pixels; it does
  not require or claim that the full original object is unoccluded. Crop support
  and mask contamination remain explicit experimental limitations.

## Human-review preparation facts (2026-11-08)

- The initial 160-case random audit has 85 VG, 58 COCOA, 14 free-images and three LAION outputs. Only 21 have matched true occlusion; it cannot independently validate every category or occlusion-bin finding. Selection was frozen before checking these distributions and did not use assistant judgments.
- The full 1,390-output alternative remains available for a complete human census. One independent human can supply a separate set of judgments; it cannot establish inter-human agreement or consensus. Preparation and software validation do not constitute human evaluation.

## Expanded-vocabulary appearance findings (2026-11-08)

- The full saved-output census has 652/1,390 outputs with at least one observed appearance defect. Positive/assessable counts: texture 578/1,304 (44.3%), color 398/1,345 (29.6%), pattern 311/601 (51.7%), semantic drift 6/1,350 (0.4%), hidden-region inconsistency 559/1,113 (50.2%), co-occurrence 11/1,199 (0.9%). These are one assistant's overlapping labels, conditional on assessability, not hidden-ground-truth accuracy.
- Eleven confirmed extra-object cases are confined to raw generated RGB, outside returned object masks. Six candidate positives were pre-existing source context; full originals were essential to avoid misattribution. Sixteen final-mask target-switch tags are separate from semantic drift.
- The requested true-occlusion bins contain 114 targets and 71 saved outputs, with availability 80.0%, 41.7%, 50.0%, 60.0%, 81.2%, 60.0%. These results do not support monotonic decline. Only five targets fall at 70-80%.
- Expansion-proxy versus annotated occlusion: Spearman rho=0.297, Pearson r=0.393, median absolute discrepancy 12.21 percentage points (181 paired outputs). Output-dependent expansion cannot replace missing true occlusion for the remaining targets.
- Keyboard pattern defects occur in 21/22 assessable outputs, bus graphic/text defects in 43/57, and laptop repeated-structure defects in 25/37. Category groups preserve recorded annotation fields, including attribute variants; they are not a controlled ranking.
- The strict sensitivity subset excludes mask, target-switch, low-resolution, small-target and occluder-residue tags, retaining 979 outputs. Texture remains 390/979 and hidden-region inconsistency 374/873; this selected subset is not a causal correction. See [the expanded-vocabulary report](output/appearance-expanded-20261107/REPORT.md).

## Appearance-failure findings (2026-11-07)

- In the 120-output single-assistant audit, positive/assessable counts are texture 54/112, color 51/113, pattern 25/45, semantic drift 4/114, hidden-region inconsistency 59/100 and co-occurrence 11/105. These overlapping raw frequencies describe a stratified sample, not whole-benchmark prevalence.
- Eleven co-occurrence positives split into seven affecting the returned region and four visible only in raw generated RGB. Six final-mask target switches are tracked separately from semantic drift.
- Unique image/name joins recover COCOA occlusion for 321 targets; all supplied ratios agree with independently decoded masks to within 3.66e-10. Requested bins contain 114 targets, with only five at 70-80% and no saved outputs there.
- Generated-area proxy has Spearman rho=0.133 and median absolute discrepancy 11.54 percentage points against true occlusion for 142 matched outputs. Initial mask undercoverage, overgrowth and target switching can inflate expansion.
- Across 951 outputs, median visible-support retention is 98.9%; 18 retain under 50%. Nine matched saved outputs have SAM-versus-annotated-visible IoU below 0.5. Mask quality confounds appearance interpretation.
- Results fluctuate across true-occlusion bins; sparse samples, selection and category/source differences prevent a monotonic or causal conclusion. Hidden-region appearance has no RGB ground truth. Full evidence: [appearance report](output/appearance-failures-20261107/REPORT.md).

## Accepted baseline findings (2026-11-06)

The requested full baseline is complete. Slurm job **31548** completed with exit code 0 in **4h 02m 38s** on four Quadro RTX 6000 GPUs. GPU inventory/FFT, CUDA extension build/forward/backward, full pipeline imports, all source-image checks, the four-worker scored smoke, full evaluation, provenance-required verification and registration all passed. The unchanged scientific contract is `configs/baseline.json` (SAM references, semantic matcher v2, released vocabulary, seed 0).

**2,565 targets: 951 saved completions and 1,614 failures (62.92%).** Saved outputs comprise 796 non-cap termination states and 155 iteration-limited outputs; neither category certifies visual completeness. There are zero missing images and zero reference failures. Overall CLIP is **28.393098 (n=951)**; LPIPS **0.735009**, feature similarity **0.242636** and SSIM **0.303397** each use **n=2,565**, including the configured failure penalties.

Recovered **16,814 files (2,960,571,039 bytes)**. Every file matches a fresh independently computed remote SHA-256, and the recovered run passed local `verify --require-full --require-provenance` against the existing dataset. All 36 captured project source files match the maintained source. See the [accepted report](output/baseline-establishment-20261105/REPORT.md) for per-source results, failure reasons, provenance, recovery evidence and interpretation limits.

- Dominant failure categories are 877 unmatched labels, 344 no-occlusion decisions and 251 ambiguous instance matches. Preserve these alongside the metric denominators when assessing a future change.
- The cluster and local verifiers agree on the full target inventory, input/reference identities, cached-metric hashes, summary arithmetic and execution provenance. Independent CPU verification does not rerun the learned metrics.
- Historical semantic-v2 and the new run have identical configurations, input/target identities and all 2,565 seeds. Six targets gained a saved completion and one lost it. All 945 shared completion PNGs differ, as do 32 reference masks and 19 full reference PNGs. The old environment is unknown; these are descriptive differences rather than an isolated improvement.
- A Windows directory listing initially reported zero bytes for an open transfer; opening the file handle showed the growing archive. Recovery also needed a shorter artifact root to avoid Windows path-length failures. The accepted artifact root is `output/baseline-31548/` (maximum path length 213).

## Launcher and completed smoke findings (2026-11-06)

- One-worker and four-worker smoke runs have identical eight target records, source/reference identities, completion hashes and aggregate metrics. Comparison evidence: `output/baseline-establishment-20261105/one-vs-four-worker-smoke-comparison.json`. This is measured equivalence on the selected eight-target smoke, not a full-benchmark reproducibility claim.

- Scored Quadro smoke 30436 passed with 1/8 completions, all four metrics and provenance verification; the other seven records are target-selection failures. This establishes one-worker execution, not benchmark quality. The replacement full workflow also requires a four-worker smoke.
- Job 30426 never reached inference: a generated setup stage illegally combined mutually exclusive runner flags. Tests now execute generated commands through the real Bash parser in both cache modes.
- An existing-cache workflow must skip provisioning and run explicit GPU checks before its ordinary `--run-only` inference preflight. The repair changes orchestration only. The current remote source includes the provenance modules; older absence findings below describe the pre-upload state.

## Baseline preparation findings (2026-11-05)

- **Measured hardware incompatibility:** pinned PyTorch 1.13.1/CUDA 11.7 passed basic CUDA/import checks on L40S, but the scored LaMa smoke failed with `CUFFT_INTERNAL_ERROR` (30428). An isolated FFT round trip reproduced it on L40S (30440); the identical check passed on Quadro RTX 6000 (30435). The integration test and all-device baseline preflight now test FFT, and baseline jobs are restricted to `debug-gpu`. Preserve the package pins; GPU architecture is an operational constraint.
- With the user's newly authorized remote execution, job 30425 validated the actual persistent GPU runtime: PyTorch 1.13.1/CUDA 11.7, GroundingDINO build and forward/backward operators, complete SAM/Mixed Context/InstaOrder/LaMa imports, and package consistency all passed on a Quadro RTX 6000. Evidence is in `output/baseline-establishment-20261105/runtime-preflight-30425/`; its 36 source-file hashes were checked after download. This validates the runtime, not full-benchmark quality or four-worker memory fit.
- An allocated probe found the old `/dev/shm` Python environment unavailable. The persistent project `.amodal-cache` existed but required the runner's pinned repairs (including NumPy 2.2.6 to 1.23.5 and CUDA 13 components to 11.7-compatible versions) and missing checkpoints. The repair completed successfully without touching historical experiment outputs.
- The default two-target smoke selects `VG/4_blackfloorlamp` and `VG/16_whitecar`. Their recovered semantic-v2 results are `not_detected` and `ambiguous_target`; this smoke can pass without exercising diffusion or learned metrics. The new baseline GPU smoke includes the first eight targets, including historical success `VG/43_television`, and requires an actually scored completion before the full run. This is runtime coverage, not a new quality result.
- The local RTX 3060 has 6 GB; the released pipeline is documented to need roughly 24 GB per worker. Read-only Slurm inspection reports eight Quadro RTX 6000 GPUs in debug-gpu, with a maximum 24-hour job. Direct compute-node SSH is prohibited; validate the node-local cache only inside an allocation.
- The cluster project currently lacks `provenance.py`, the configuration workflow and helper scripts. Starting its old evaluator would not establish the requested current-workflow baseline.

## Experiment artifacts recovered and verified (2026-11-05)

- Recovered 70,012 files (11.90 GB) into `output/artifact-recovery-2026-11-05/downloads/`; every file matches an independent remote SHA-256. See [the recovery report](output/artifact-recovery-2026-11-05/REPORT.md).
- Earlier missing-artifact findings are superseded by this recovery. The pre-cleanup inventory proves those gaps existed before cleanup; local backups did not contain the missing runs.
- Fresh remote and local hashes establish byte identity for 941/946 shared semantic-v2/annotated completions. The five changed images match the five changed metric tuples; 622 targets have changed reference image/mask bytes. Annotated retains all 946 successes and adds 444.
- Original masks plus the verified source RGB (or completion JPEG) reproduce accepted PNG bytes exactly in Pillow 9.5.0 / NumPy 1.23.5. Acceptance used remote SHA-256 equality, not visual similarity or regenerated segmentation.
- Six full historical and two subset runs have passing artifact checks. Three legacy LISA summaries require documented metadata-only verification views. Historical source/environment execution provenance is absent from the recovered run directories; recovery does not make these runs eligible for provenance-required registration.

## Manager review of historical reports (2026-11-05)

- Eight saved CSV summaries recompute consistently; detailed evidence is in `logs/manager-review-2026-11-05/evidence.json`.
- Annotated vocabulary versus semantic v2: 444 newly saved targets, zero lost outputs, and identical reported metric tuples for 941/946 shared outputs. Shared-output appearance means change slightly adversely; aggregate gains primarily reflect recovered coverage and reduced failure penalties. Metric identity does not establish identical image bytes.
- The annotated report records 1,390 saved outputs: 1,178 non-cap stop states and 212 iteration-limited outputs. Neither status certifies visual completeness. Its 1,175 failures include 484 no-occlusion decisions, 329 ambiguous targets and 232 unmatched labels.
- Newer semantic/annotated reports lack detailed target, completion and reference artifacts locally. SAM-bgfix artifacts pass the fresh CPU verifier, but historical execution provenance is absent. No `experiments.csv` exists locally. These limits distinguish historical report evidence from a newly accepted provenance-backed experiment.

## Retained research proposal, not an experimental finding (2026-11-05)

- [Surface-Consistent PMC](docs/SURFACE_CONSISTENT_PMC.md) proposes selecting among pretrained PMC completions using visible/hidden boundary continuity, color distributions and texture compatibility. It is a proposal in the supplied conversation, not a published PMC component or a measured improvement here.
- Its hidden-region evaluation requires appropriate complete-object ground truth; the current SAM/LISA visible-reference benchmark does not supply it. The source's example scorer also leaves original-visible anchoring and all-invalid-candidate handling unresolved. These are design considerations to resolve before implementation.

## Repository retention findings (2026-11-05)

- The evaluator records `matching.py` and `detection.py` implementation hashes in run/reference configurations. Filesystem cleanup must not casually reformat those modules because it can invalidate resume comparisons.
- The offline PD-MC fixture and archived upstream `pdmc_main.py` are byte-identical, deliberately serving different locations. Annotation fixtures likewise support tests without installed datasets. Keep these copies and the release ZIP used by an optional integration test.
- Dataset download/reconstruction scripts and output catalog/review builders are provenance-bearing artifacts, even though they sit outside the active package. They were retained in place with the research data.
- The Windows editable install has its own `amodal_benchmark-0.2.0.dist-info` under `.venv/Lib/site-packages`. Removing the generated root egg-info from the workspace did not break the installed CLI or smoke. Regenerate root metadata through a CPU editable installation when needed.

## Protocol

- Official annotation checksums establish the fixed target order and counts. Full source-image preflight passed for all 2,565 targets on 2026-11-04. This confirms availability and decodability, not authenticity of historical LAION bytes.
- SAM references originate from the pipeline's own initial detection; LISA references are independently generated visible segmentations. Appearance results from these protocols must be interpreted separately.
- CLIP excludes failed completions. Its denominator can change between experiments, so its mean alone cannot establish improvement.
- A saved completion at the iteration cap is not the same as a completed object. Existing detailed summaries retain this distinction.

## Engineering evidence

- The synthetic CPU smoke exercises the real inference adapter, matcher, reference/artifact publication, failure penalties, resume and verifier while substituting expensive model calls and metric values.
- The verifier now rejects stale failure rates and histograms before historical-summary normalization can discard them.
- Full failed-attempt registration checks exact official target identities and preserved provenance, not just a count of 2,565.
- One-factor LISA comparisons require frozen reference bytes. SAM comparisons explicitly record when their generated reference bytes differ.
- Data preflight and independent verification reject SAM references passed off as the independent LISA protocol.

## Limits

The initial CPU rebuild did not measure GPU quality. The accepted full baseline above now records actual GPU evaluation scores; no speed, memory or model-quality improvement is established. The CPU verifier checks saved learned-metric values and their input signatures; it does not recompute the neural metrics. No newly accepted best experiment exists yet.

## Annotated Surface-PMC integration findings (2026-11-12)

The existing annotated scene wrapper ignores its requested category argument and
queries all per-image categories. The surface bridge must capture the target-only
segmenter before this wrapper is installed, preserving prior candidate mask
semantics. No fitted coefficients or diffusion weights change.

The image split screen found VG/2316160 nearly identical to prior VG/1159611
(aligned 64px normalized RGB MAE 0.003209). It was excluded before generation and
replaced by the same hash-sampling rule. Final 192-image split audit passes.
