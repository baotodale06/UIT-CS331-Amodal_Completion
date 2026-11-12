# Active plan: annotated Surface-PMC, 12 November 2026

1. Complete copied-reference and GPU runtime verification.
2. Run frozen annotated control and Surface-PMC on 204 targets, preserve snapshots.
3. Compare all targets with LPIPS primary, secondary CLIP/VGG/SSIM and failure transitions.
4. Audit decisions/artifacts, report qualitative gains and failures, and recover evidence.
5. Continue both unchanged configurations over all 2,565 targets; verify/register only full results.
6. Deliver measured cohort findings within 06:46?11:46 UTC; retain valid longer jobs.

No new training is required for the primary transfer test. Historical methods and
results are retained; a source repair keeps candidate segmentation target-only
when annotated scene segmentation is active.

---

# Open-world amodal research plan

Updated 2026-11-11.

## Scientific reporting revision completed

The user's request for formulas and scientific depth is delivered in
[the 75-slide research briefing](output/research-briefing-2026-11-11/presentation.pdf),
with 57 rendered equations, derivations, exact fitted parameters, statistical
definitions, detailed notes and source links. The earlier manager briefs remain
preserved. This changes the explanation, not any experiment or measured result.

## Final-RGB diagnostics completed; full target remains unmet

Jobs39404/39407 and the subsequently frozen four-GPU headroom39411 completed and
passed independent local numerical audits. Read
output/surface-lastpass-review-20261111/REPORT.md. Removing final correction worsens
three reconstruction metrics. Even separate per-metric raw-pool oracle choices
reach only CLIP2.817%, LPIPS0.740%, SSIM1.407% (VGG3.257%) versus matched control.
These different reference-informed choices are bounds, not a learned result.
They close final raw RGB reranking as a route to all four3% full gains on existing
fixed masks/trajectories. No new diffusion or ranker fitting was performed.

All remote jobs, collectors and audits are done; do not restart them. A later
improvement study must change generation/trajectory or another explicitly isolated
mechanism and freeze a new evaluation design. Do not change vocabulary, failure
penalties or reference protocol to claim a surface gain. The independently
confirmed128-image controlled result remains successful and separate.

## Latest continuation: completed full comparison and frozen diagnostic

Full 39387/finalizer37814 completed. All 2,565 targets were verified and registered;
CLIP/LPIPS/VGG/SSIM gains are1.123/0.835/2.789/1.782%, below the full 3% target.
Read output/surface-progressive-review-20261111/REPORT.md. The independent
controlled raw-selection confirmation remains successful and separate.

Measurement39404 and finalizer 39407 now evaluate the existing final selected raw
RGB while fixing masks, geometry, statuses and preceding trajectories. All three
controls are rescored on the same runtime, with every target retained. This
post-hoc diagnostic is frozen before its new metric measurement and cannot be
called an independent confirmation or end-to-end raw progressive result.

Continue monitoring local session 62314 and
output/surface-lastpass-ablation-20261111-recovery/status.json. After completion,
inspect metric replay drift and all full-denominator contrasts before deciding
whether another progressive run is justified. Preserve all completed results.

## Current continuation

The independent 128-image raw-selection confirmation is complete and locally
audited: MAE/LPIPS/DreamSim/CLIP gains are 7.644/11.328/16.766/4.068%, with all
individual nominal 95% lower bounds above 3%. SSIM is below target (+1.668%).
Read output/surface-confirmation128-20261110/report/REPORT.md. Preserve the
distinction from the earlier corrected primary and full progressive experiment.

Full job 37812 timed out after 2,356 targets. Same-configuration continuation 39387
is running; finalizer 37814 now depends on it. Existing read-only collector watches
39387/37814/37843 and will recover full comparison metadata after completion.
Next: validate the full comparison, denominators, failures and recorded provenance;
report its result separately. Do not resubmit completed confirmation or retune it.
The original five-hour window is already exceeded; valid inference remains running.

## Active Surface-PMC improvement follow-up

2026-11-10 05:17 UTC: remaining work awaits Slurm results. Local collector20571
automatically retrieves independently checksum-verified finished evidence;
see output/surface-job-recovery-20261110/README.md. Review the confirmation and
full progressive reports when they exist, preserving scope and all failure
counts. Do not infer completion from a running job or from successful recovery.

2026-11-10 04:17 UTC: follow the unchanged selection-only component with an
independent, pre-generation-frozen 128-image confirmation (64 VG, 64 COCOA),
excluding all 349 earlier images. Job **37843** is submitted with four GPUs and
four workers; see `docs/SURFACE_CONFIRMATION_20261110.md`. No refitting, RGB
correction, or new method search. Report all four primary point gains and
confidence intervals after native replay; do not combine this cohort with the
previous 166 to obscure a failure. The original deadline has already passed.
The full progressive 37812/37814 comparison remains a separate pending result.

Current execution, 2026-11-10 03:25 UTC: controlled scoring, exact native replay,
report verification and all 166 gallery cases are complete. The original window
has elapsed. Preserve the final pre-test freeze. Job **37812** passed the final
progressive smoke and is running the matched full control, followed by the full
surface method. Dependent job **37814** performs verification and paired reporting.
Do not tune on the fresh 166 outcomes or describe controlled gains as full-benchmark
gains. The corrected primary misses the CLIP 3% target; the pre-specified
selection-only component clears all four point targets as a secondary result.

Within 18:07–23:07 UTC on 9 November (10 November local), develop on the previous
183 cases, freeze a surface ranker/calibration policy, and evaluate on 166 new
image-disjoint cases. Compare uniform and fixed-seed controls with MAE, LPIPS,
DreamSim and CLIP. Separately derive and score a full Open-World surface
postprocessing run against its immutable baseline, preserving failures,
references and generated masks. Verify provenance and report any unmet targets.
See `docs/SURFACE_FOLLOWUP_20261110.md` and the active section of `session.md`.

## Surface-PMC research completed (2026-11-08)

The requested five-hour study produced 732 candidates, a frozen 24-image
development split, 159 held-out pilot/replication cases, supporting LPIPS,
exact replay, theoretical derivations and a complete explanation gallery.
Read the [handoff](output/surface-consistency-20261108/START_HERE.md) and
[measured findings](docs/SURFACE_CONSISTENCY_RESULTS.md).

The full surface rule remains inconclusive: pooled MAE gain +0.004310
[-0.000512, +0.009074]. The local variant has exploratory pooled gain +0.007187
[+0.002183, +0.012296], but local superiority to full surface or visible-pixel
preservation is not established. Preserve the separate cohort results and
post-hoc explanation labels. No trained model, human preference or full
progressive benchmark improvement is claimed.

All requested generation/metric jobs completed. No further inference is pending.
Do not rerun the completed study or tune its frozen test decisions. A future
local-versus-visible-preservation study needs a new cohort and independent human
preferences; it is a recommendation, not an active or submitted experiment.

## Independent human rating pack prepared (2026-11-08)

The user requested preparation for one independent human rater. Delivered [the handoff](output/human-review-20261108/README.md): a fixed random 160-output audit plus a separately packaged full 1,390-output census, both from the expanded-vocabulary run. Default preparation is the manageable random audit; scope should be selected before collecting ratings. The packs hide prior judgments and metrics, preserve native evidence, and collect six severity labels, uncertainty, confounds and notes.

Preparation is complete. Actual human review remains pending. No human ratings, adjudication, consensus or generator improvement have been claimed. Use [the frozen protocol](docs/HUMAN_VISUAL_REVIEW.md) and the new response importer when a human returns their file; preserve existing assistant reports.

## Expanded-vocabulary appearance census completed (2026-11-08)

The requested expanded-vocabulary Person 2 deliverable is complete in [the full census report](output/appearance-expanded-20261107/REPORT.md): all 2,565 target records, all 1,390 saved outputs visually reviewed, six appearance failure types, frequency/severity, required true-occlusion bins, category comparisons, explicit uncertainty and source-checked examples. The earlier 120-output baseline report remains a distinct analysis. No inference rerun or additional annotation task is pending under this request. The final census labels and source outputs are frozen; independent human adjudication and controlled hidden-RGB experiments remain proposed follow-ups.

## Appearance-failure task completed (2026-11-07)

The requested Person 2 taxonomy, frequency/severity analysis, occlusion plots, category comparisons and visual examples are delivered in [the report package](output/appearance-failures-20261107/REPORT.md). The sample and final labels are frozen, with reproducible summaries and verification. No inference or additional annotation task is pending under this request.

Possible next experiments are independent human adjudication, category-balanced pseudo-occlusion with hidden RGB, and equal-budget surface/context ablations. They are proposals, not measured improvements or newly scheduled runs. Preserve the distinction between 114 targets in true-occlusion bins and the 120-output expansion-stratified visual sample.

## Baseline task completed (2026-11-06)

The requested full baseline is complete. Slurm job **31548** completed with exit code 0 in **4h 02m 38s** on four Quadro RTX 6000 GPUs. GPU inventory/FFT, CUDA extension build/forward/backward, full pipeline imports, all source-image checks, the four-worker scored smoke, full evaluation, provenance-required verification and registration all passed. The unchanged scientific contract is `configs/baseline.json` (SAM references, semantic matcher v2, released vocabulary, seed 0).

**2,565 targets: 951 saved completions and 1,614 failures (62.92%).** Saved outputs comprise 796 non-cap termination states and 155 iteration-limited outputs; neither category certifies visual completeness. There are zero missing images and zero reference failures. Overall CLIP is **28.393098 (n=951)**; LPIPS **0.735009**, feature similarity **0.242636** and SSIM **0.303397** each use **n=2,565**, including the configured failure penalties.

Recovered **16,814 files (2,960,571,039 bytes)**. Every file matches a fresh independently computed remote SHA-256, and the recovered run passed local `verify --require-full --require-provenance` against the existing dataset. All 36 captured project source files match the maintained source. See the [accepted report](output/baseline-establishment-20261105/REPORT.md) for per-source results, failure reasons, provenance, recovery evidence and interpretation limits.

The baseline is now the explicit reference for future compatible experiments. Preserve its accepted artifacts and registry entry; use a new output directory for any changed experiment. No inference, scoring, transfer or registration work remains for this baseline.

## Experiment artifacts recovered and verified (2026-11-05)

Recovered 70,012 files (11.90 GB) into `output/artifact-recovery-2026-11-05/downloads/`; every file matches an independent remote SHA-256. See [the recovery report](output/artifact-recovery-2026-11-05/REPORT.md). Artifact verification now covers all six full historical runs and both 20-target runs. Preserve the recovered originals and metadata-only legacy verification views. Historical execution provenance remains unavailable. The new provenance-backed baseline above now supplies the current reference; do not register historical runs by manufacturing provenance.

## Objective

Maintain a reproducible research workflow for the released PD-MC method on the Open-World benchmark. The setup guide is adapted to this image-inference project; competition submissions and training infrastructure are outside scope.

## Dataset and protocol

Use the verified four-source contract: 2,379 image identities, 2,565 targets and 553 labels. Preserve original image bytes and LAION provenance. SAM and LISA appearance references define separate protocols. Report failure counts, exclusions, per-metric denominators and iteration-limit states.

## Baseline

`configs/baseline.json` declares SAM references, semantic matcher v2, released detector vocabulary, automatic backdrop sizing and four workers. The accepted run is `/home/daoqm/Vu_Hung/cv_nangcao/output/pdmc-baseline-provenance-20261105` (Slurm 31548). Its verified local copy is `output/baseline-31548/output/pdmc-baseline-provenance-20261105`. This establishes the reference baseline; it does not select a best model.

## Annotated vocabulary experiment already completed

The expanded annotated vocabulary was already evaluated on all 2,565 targets in `output/artifact-recovery-2026-11-05/downloads/pdmc-open-world-sam-annotated`: 1,390 saved completions and 1,175 failures. Its recovered artifacts passed independent verification. Reuse these historical results; no routine rerun is needed or planned. See the [recovery report](output/artifact-recovery-2026-11-05/REPORT.md).

The paired historical comparison against semantic v2 retains all 946 saved completions and adds 444. Historical execution/source/environment snapshots are absent, so these results cannot receive provenance-required registration or establish an isolated vocabulary effect against the newly validated baseline. A fresh run would serve a separate, explicit need for a comparison under the current runtime and provenance workflow; it is not unfinished baseline work.

## Validation

Run `bash scripts/check.sh` in Linux/WSL. Before a full evaluation, run the GPU extension/import checks and the eight-target scored smoke inside the allocated cluster runtime. Independently verify full artifacts and provenance before adding the run to the research ledger. See `README.md` for exact commands.

## Retained research direction

The user requested understanding and durable storage of [Surface-Consistent PMC](docs/SURFACE_CONSISTENT_PMC.md): generate multiple PMC completions and select by visible-object appearance consistency. The proposal and focused source are preserved for future questions. Implementation has not been requested by this knowledge-capture task. The baseline task and historical annotated-vocabulary evaluation are complete; no further experiment is scheduled by this plan.
