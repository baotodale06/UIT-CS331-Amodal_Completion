# Scope decisions and negative results

Updated 2026-11-11.

## Close final raw RGB reranking on existing progressive outputs (11 November)

Completed39404/39407 show that removing final corrections worsens LPIPS/VGG/SSIM
despite a small CLIP gain. Completed39411 measures every saved final raw candidate:
7,664 RGB images in958 pools, with masks/geometry/trajectory/failures fixed.
Even per-metric oracle raw choices miss3% for CLIP(2.817%), LPIPS(0.740%) and
SSIM(1.407%). VGG's3.257% ceiling uses a different choice. Do not claim combined
oracle success or deploy a reference-informed oracle. These diagnostics and
their independent audits are complete in output/surface-lastpass-review-20261111.

Do not launch another expensive raw full run assuming this final step alone
caused the shortfall. New generation, candidate masks and earlier trajectory
interventions are outside these bounds; the whole surface-consistency direction
is NOT disproven. Preserve the successful independent controlled confirmation.

## Full corrected method misses the target (11 November)

Completed full gains are CLIP 1.123%, LPIPS 0.835%, VGG 2.789%, SSIM 1.782%, with all
2,565 targets retained. Do not claim a 3% full-benchmark improvement. The4.165%
shared-success LPIPS gain is a diagnostic, not an alternative primary endpoint.
Do not change failure penalties, detector vocabulary or reference protocol to
attribute a metric improvement to surface consistency.

Do not relabel the confirmed one-step raw selector as the completed full corrected
method. The frozen final-pass ablation39404/39407 preserves the latter's masks and
trajectory; even a positive result would not establish raw progressive inference.

## Preserve independent-confirmation scope (2026-11-11)

The independent 128-image raw-selection confirmation now passes its four primary
3% point targets and individual nominal 95% lower-bound criterion. It does not
retroactively make the earlier 166-image corrected primary a success. SSIM+1.668%
still misses3%; do not promote the finding as every recorded metric passing.
Do not refit on confirmation outcomes or describe the ongoing corrected full
progressive run as the confirmed raw selector. Full job 37812 timed out; continue
39387 with unchanged policy and records, rather than replacing it with new settings.
No new metric optimization or data exclusion was performed on this cohort.

## Fresh controlled CLIP target missed (2026-11-10)

Do not relabel the selection-only secondary finding as the earlier primary.
Its 128-image independent confirmation is now frozen and queued as job 37843;
no refitting or correction search follows the 166-image outcomes. Keep the
previous corrected method and the ongoing full progressive comparison intact.
The new confirmation is separate evidence and remains unmeasured.

The frozen eight-pool method's fresh point gains are MAE9.09%, LPIPS12.56%,
DreamSim18.55% and CLIP2.42%. CLIP's nominal 95% interval is [1.93%,2.95%].
Do not claim every primary metric meets3%, or choose a new selector after seeing
these outcomes. Compared with the same corrected semantic-only selector,
surface improves MAE/LPIPS/DreamSim but CLIP falls0.71%. Exact native replay now
passes; full progressive improvement has not been measured.

The pre-specified selection-only component clears all four 3% point targets
(5.85%, 6.75%, 12.76%, 3.07%) and remains a secondary result. It does not change
the corrected primary's missed target; no new primary was selected after testing.

## Failed full calibration approach (2026-11-10)

The frozen full-run affine residual calibration preserves all 2,565 targets and
951 saved outputs, but misses the 3-5% goal: relative LPIPS +0.388%, VGG +1.019%,
SSIM +0.946%, CLIP -0.332%. The calibrated run passes full provenance verification.
Do not tune this frozen experiment after inspecting its metrics, describe it as
a successful full-benchmark improvement, or infer hidden MAE without truth.
See docs/SURFACE_FOLLOWUP_20261110.md for source-image bootstrap intervals.

## Current follow-up constraints

- Waiting for four Quadro GPUs would start after the five-hour window; pending
  job 37564 was cancelled after a successful isolated L40S compatibility smoke.
- The old 183-case study cannot serve as held-out evidence for the new ranker:
  all its cases are explicitly development now. Its prior frozen results remain.
- There are no further eligible LAION images under the unchanged mask-size
  criteria after excluding the prior study. The new 166-case cohort therefore
  contains VG, COCOA and free-images only, and must not be described as four-source.

## Surface-PMC interpretation boundaries after controlled testing (2026-11-08)

- The primary full-surface pooled interval crosses zero. Do not market the
  implemented rule as a proven improvement or replace it with the best test ablation.
- A significant local-versus-uniform result and an inconclusive full-versus-uniform
  result do not establish local-versus-full superiority; the paired contrast is
  inconclusive. Local-versus-visible-preservation is also inconclusive.
- Pseudo-occlusion of originally observed pixels does not establish recovery of
  naturally hidden parts, independent human preference, full amodal shape or
  progressive-loop gains. The one-step study remains separate from the full registry.
- Constructed counterexamples and post-hoc diagnostics expose proxy limitations;
  they do not establish causal natural-image failure frequencies or a trained fix.
- Training a material-change/uncertainty model is deferred until separate training
  and validation data are defined. No training run was needed or started for this
  completed five-hour request. A fresh confirmation study is a future recommendation.

## Human-review interpretation boundaries (2026-11-08)

- Do not send the earlier report, labels or highlighted failure examples with the independent-review packet; these would prime the rater. Only the allowlisted reviewer ZIP is intended for sharing before submission.
- Do not treat the prepared 160-case audit as human review of all 1,390 saved outputs, replace random cases after seeing responses, or fill drafts/unknowns with zero.
- Do not describe human–assistant agreement as agreement between two humans. Preserve both original rating sets before any later disagreement discussion.

## Expanded-vocabulary census interpretation (2026-11-08)

- The earlier 120-case baseline audit cannot stand in for the requested expanded-vocabulary run. The new report uses all 1,390 saved annotated-vocabulary outputs and separately accounts for 1,175 failures.
- Cropped context is insufficient to establish a newly generated co-occurring object. Full-source checks removed six provisional positives; eleven confirmed cases remain, all raw-RGB-only. Existing objects and ambiguous remnants are not automatically regeneration.
- Full census coverage removes sampling weights, not reviewer uncertainty. Single-assistant panel judgments require independent human adjudication for consensus claims; unknown labels remain explicit.
- Heavy occlusion is a stratification condition, not a texture/semantic failure type. True ratios remain unavailable for 2,244 targets; model expansion is not substituted. The five-target 70-80% bin cannot establish a smooth degradation curve.
- No generator improvement, surface-guidance benefit or isolated vocabulary effect was established. Historical provenance remains absent and was not reconstructed. The task required analysis of saved outputs, not a new GPU experiment.

## Appearance-analysis shortcuts rejected (2026-11-07)

- Generated-area fraction is not ground-truth occlusion; its weak observed correlation prevents substituting it for the matched COCOA subset or binning failed outputs by it.
- Hidden-region consistency labels do not establish correctness of unseen texture/color. There is no hidden RGB ground truth in this evaluation.
- Empty saved-output bins do not have zero appearance error: conditional metrics remain missing, while all-target LPIPS can be 1 solely through penalties.
- A changed final target mask is not by itself semantic drift; existing context or ambiguous occluder residue is not automatically a newly generated object.
- Raw frequencies from balanced expansion strata cannot be promoted to whole-benchmark failure rates. The report includes conditional weighted estimates and unknown/NA denominators, without claiming independent human consensus.
- Small high-occlusion/category groups do not establish monotonic performance decline, significance or causal mechanisms. Proposed surface/context controls remain unexecuted.

## Baseline interpretation and recovery limits (2026-11-06)

- The net gain of five saved completions over historical semantic-v2 is not evidence of a controlled model improvement: there are six gains, one loss, changed references and no historical environment snapshot.
- A byte-identical rerun of historical completions is not established. The new full run is accepted on its own preserved source, runtime, inputs, artifacts, metrics and provenance.
- Slow bulk transport was superseded by exact-byte recovery. Local duplicates and reconstructed PNG encodings were accepted only when they matched fresh remote hashes; no historical scores or records were relabelled as a new execution. The partial archive is retained as transport evidence.

## Launcher mode conflict resolved (2026-11-06)

Combining `--setup-only` with `--run-only` is rejected by the shared runner and caused job 30426 to fail before any inference. Existing-cache workflows now skip the provisioning stage. Generated-command parser regressions cover both modes; weakening the runner guard is unnecessary.

## Baseline acceptance shortcuts rejected (2026-11-05)

- Basic CUDA matrix/operator tests and successful LaMa imports did not establish FFT support on L40S. Scored smoke and an isolated FFT diagnostic both failed under CUDA 11.7. The same FFT check passes on Quadro; restrict the baseline to that hardware and retain the pinned runtime.
- Two selection failures do not validate completion generation and learned metrics. The baseline workflow requires a successful, fully scored target in a separate GPU smoke.
- Existing recovered historical scores cannot be relabelled as a new provenance-backed baseline. The prepared full output directory is new, and the workflow rejects adoption of historical targets without execution provenance.
- A passing CPU suite and a 6 GB local GPU do not establish the cluster runtime or full baseline. Actual runtime validation subsequently passed in job 30425; full-baseline quality/verification remains pending.

## Recovery findings and rejected shortcuts (2026-11-05)

- Equal reported metrics alone were insufficient to restore completion bytes. Fresh remote SHA-256 comparisons now establish identity for 941 shared images; the artifacts were recovered with exact-byte checks.
- Initial local hash lookup found no matches for 3,817 missing LISA reference-hash uses. Recovery used the genuine remote masks and records instead of substituting SAM references.
- Full bulk transfer was too slow for the one-hour window. Exact local duplicates plus downloaded masks/JPEGs and hash-checked PNG encoding recovered all 70,012 files without changing the experiment.
- Current source snapshots cannot replace missing historical execution provenance. No historical provenance was reconstructed or registration attempted. See [the recovery report](output/artifact-recovery-2026-11-05/REPORT.md).

## Deliberately omitted from the starter workflow

| Idea | Reason | Revisit when |
| --- | --- | --- |
| Kaggle submission and leaderboard tooling | This is a research evaluation on released images | A real external submission contract is introduced |
| Tabular trainers, classification targets, ROC-AUC and random folds | PD-MC is evaluated as a pretrained image-completion pipeline | A separate training task has a documented dataset and validation design |
| Automatic best-model promotion | Reference protocols, exclusions and conditional denominators can differ | Comparable full experiments have verified evidence and an explicit selection rule |
| Moving every module into a new `src/` tree | Existing Slurm entry points, source uploads and runtime paths already work | A concrete packaging need justifies a compatibility migration |
| Unrequested remote deployment or submission | Remote work must serve the user's requested project task; agents are now authorized to execute needed transfers and jobs | A new project task requires deployment or computation |

These are scope decisions, not empirical claims that an algorithm failed. No new scientific negative result was produced in this infrastructure rebuild.
# Surface follow-up: calibration alone misses the goal (2026-11-10)

Frozen affine visible-residual correction and exact observed RGB restoration on
the accepted full baseline preserve all 951 generated masks and 1,614 failures.
Paired full-protocol gains are LPIPS +0.388%, VGG +1.019%, SSIM +0.946%, with
CLIP -0.332%. All 2,565 artifacts pass provenance-required verification. This
does not satisfy 3-5% and is not a successful full-benchmark improvement. Keep
the experiment and its negative CLIP result; do not tune it after inspection.
See `docs/SURFACE_FOLLOWUP_20261110.md`. Candidate ranking and controlled
surface-consensus experiments remain active and use separate development data.


## Deferred for annotated transfer experiment (2026-11-12)

Do not retrain or tune the selector on the new 204-target evaluation cohort.
The first question is transfer of the existing frozen method to the team's
annotated baseline. New fitting would require a separately frozen development
protocol and independent evaluation. Do not count the vocabulary expansion itself
as a surface gain, or K=8 versus K=1 as an equal-compute selector effect.
