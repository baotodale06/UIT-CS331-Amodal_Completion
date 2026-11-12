# Project progress

Updated 2026-11-11.

## Scientific briefing revision delivered (11 November)

- Responded to the user's criticism that the manager deck omitted the mathematics:
  [75-slide research PDF](output/research-briefing-2026-11-11/presentation.pdf), offline
  HTML, 57 equations, approximately 19,300 words of notes, and full-precision
  parameters for the frozen 22-feature ranker.
- Added implemented loss definitions, theoretical derivations, fitting objectives,
  correction equations, statistical estimands, failure denominators and scoped
  oracle arguments alongside the preserved project history and results.
- Checked equation rendering, layout, controls, PDF contents, evidence links,
  parameter identity, numeric claims, source preservation and portable archive.
  Existing studies and both previous manager briefs remain unchanged.

## Updated manager briefing delivered (11 November)

- Recreated the previous manager-brief format as a 40-slide offline presentation
  and PDF, with approximately 10,300 words of detailed notes and evidence links.
- Covers the full project history through the completed final-RGB and candidate-pool
  diagnostics; controlled confirmation and full-benchmark outcomes remain separate.
- Includes the unchanged historical-results CSV, 28 scoped numerical comparisons,
  source hashes, delivery validation and a portable package.
- Browser/layout, PDF and archive checks pass; the original 5 November brief and
  maintained scientific sources remain unchanged. No new scientific run was made.
- Start at [the briefing handoff](output/manager-briefing-2026-11-11/README.md).

## Final-RGB ablation and pool ceilings completed (11 November, 12:01 UTC)

Recovered39404/39407: all2,565 targets,958 fixed-mask raw images. Raw-final gains
versus matched control: CLIP1.423%, LPIPS0.596%, VGG2.121%, SSIM1.087%. It loses
three reconstruction metrics against corrected output; original-score drift is
at most1.79e-7. Independent audit verifies2,577 recovered members,7,047 original
files, target lineage, failures, means and image-cluster bootstrap intervals.

Frozen headroom39411 completed in23m36s with four Quadros/four workers. Scored all
7,664 saved raw candidates without new diffusion. Separate per-metric oracle
ceilings: CLIP2.817%, LPIPS0.740%, VGG3.257%, SSIM1.407%. Each oracle uses evaluation
references and a different choice; no deployable or joint gain is claimed.
Selected RGB hashes and all metrics exactly reproduce the prior ablation. Local
headroom audit passes2,582 members,2,565 parent links,3,832 oracle choices and all
means/intervals. Seven new focused tests and both cluster GPU checks pass.

Combined report, audits and verified569,403-byte portable package are in
output/surface-lastpass-review-20261111/. All jobs/collectors are done. Full3%
objective remains unmet; controlled128-image confirmation remains separate.

## Full run complete; new diagnostic submitted (11 November, 11:00 UTC)

Recovered and independently audited full 39387/37814 results: all 2,565 targets,
951 control and958 surface successes,12 gained/5 lost. CLIP+1.123%, LPIPS+0.835%,
VGG+2.789%, SSIM+1.782%; full 3% objective missed. Remote full/provenance verification
and registration pass. Local audit checks 14,123 recovered files,4,921 pre-resume
records, every paired identity, metric means and image-cluster bootstrap intervals.
One-page report and verified145,543-byte reading ZIP are in
output/surface-progressive-review-20261111/.

Surface sampling was reached by 966 targets, with 1,599 unchanged earlier failures.
Shared-success LPIPS+4.165% is diagnostic only. Completed evidence recovery is
finished; the independent 128-image confirmation keeps its original scope.

Frozen final-pass RGB ablation39404 is pending on four Quadros/four workers;
39407 finalizes after success. New read-only collector session 62314 retrieves its
hashed evidence. No new diffusion or fitted parameters;958 saved images retain
the corrected run's alpha/frame/trajectory, and all failure penalties remain.
Four CPU derivation regressions pass locally and on pinned WSL; remote source
hash/real-image preflight passes. Full neural metrics will be rescored together.

## Confirmation delivered; full benchmark resumed (2026-11-11)

Completed confirmation 37843: 128 independent VG/COCOA images, 1,024 candidates,
zero generation failures. Frozen raw selection improves MAE 7.644%, LPIPS 11.328%,
DreamSim 16.766%, CLIP 4.068%; all individual nominal 95% lower bounds exceed 3%.
SSIM+1.668% remains below 3%. Audit independently reproduces selected-image MAE,
all summary means/intervals and all decisions; native all-candidate replay passed.
Report PDF, complete 128-case gallery and numerical audit are under
output/surface-confirmation128-20261110/report/. Browser checks pass every image
and source/outcome/search filter; PDF layout is reviewed.

Timeout37812 preserved 2,356/2,565 surface target records. Unchanged continuation
39387 runs four workers on four Quadro GPUs; finalizer 37814 depends on it. At
07:42:53 UTC the surface count is2,383; control is complete. Collector now tracks
the replacement job; five focused recovery regressions pass. Exact resume receipt:
output/surface-followup-20261110-design/resume-20261111-submission.json.
Full progressive quality remains unmeasured. Original five-hour deadline missed.

07:49 UTC health check: 2,394/2,565 surface records; two workers completed and
both remaining workers have fresh logs without a recent traceback. Frozen hashes
match and finalizer dependency is correct. Recovery regressions also pass on
pinned WSL Python 3.10. Portable delivery is surface-confirmation-delivery.zip:
443 members, 5,341,427 bytes, all 442 payload hashes and CRC verified.

## Surface-PMC improvement follow-up in progress

05:17 UTC: automatic read-only evidence collector started as local session20571.
It waits for successful jobs37843/37814, checks independent remote archive/member
hashes, and recovers evidence into output/surface-job-recovery-20261110. Four
focused safeguards pass on Windows and pinned WSL Python3.10; the unchanged
confirmation freeze passes preflight. Full-run metadata recovery will retain
native PNGs on the cluster and cannot substitute for image-level replay. All
remaining quality conclusions depend on pending external results.

05:07 UTC: full control has 963/2,565 records with four active workers and no
recent traceback. Confirmation 37843 remains pending for an estimated
09:45:49 UTC start. Read-only alternative-partition scheduling predicts a later
start; local 6 GiB GPU cannot run the validated configuration. Existing jobs,
allocations and scientific freezes are preserved. Current receipts are in
`output/surface-confirmation128-20261110-design/status-latest.json`.

04:17 UTC: prepared and submitted independent selection-only confirmation job
**37843** (four GPUs/workers), with 128 new VG/COCOA images and all 349 earlier
images excluded. Frozen inputs, policy, source snapshots, overlap audit, upload
hash verification, and remote preflight are complete. The unchanged ranker
returns unmodified generated RGB/masks. No confirmation outcomes exist yet.
248 CPU tests discovered: 242 passed and six optional dependency skips; shell
regressions/smoke pass. The two new confirmation regressions also pass in the
pinned replay environment. See `docs/SURFACE_CONFIRMATION_20261110.md`.

03:52 UTC: monitored control reaches256/2,565 targets, approximately10%. All four
worker logs are active (updated within9-20seconds) without a traceback in their
last200 lines.37812 remains live;37814 waits. No full comparison metric exists
yet. Read-only monitor session77472 remains available for the next continuation.

03:36 UTC: independent final-smoke audit verifies eight derived seeds, all19
saved artifact hashes, rank selection and exact preservation of81,241 visible
pixels. This verifies one progressive sampler pass only. Job37812 advances to
151/2,565 control targets; storage is sufficient and37814 remains dependent.

03:32 UTC: live full control advances to127/2,565 target records. Added actual
overall metric means/denominators and coverage tables to the queued full report,
alongside paired shared-success CLIP. Fixed undefined relative bootstrap intervals
for zero-baseline samples without discarding draws; no method or scored result
changed. Nine focused regressions pass locally and on the cluster. Deployment
hashes match and all17 frozen scientific files are unchanged.37812 is running;
37814 remains pending. The controlled reading ZIP remains immutable.

03:26 UTC: verified controlled report, PDF, all166 gallery cases and portable
reading ZIP are delivered under output/surface-followup-20261110. The package
contains555 independently checked members (8.84MB). Report validation reproduces
all primary/secondary means and332 selected-artifact hashes; all panels and
gallery filters pass browser checks. Full frozen progressive smoke passes;
37812 is now running the matched control (82 target records at03:25UTC), then
the full method.37814 remains queued for final verification and paired analysis.
No full progressive result is established, and the five-hour deadline was missed.

03:14 UTC: all 166 fresh cases measured and independently verified. Exact native
replay 37799 passes 1,328 candidate scores and 3,984 MAE checks, with unchanged
pre-scoring choices and corrected RGB/masks. Corrected primary gains:
MAE 9.09%, LPIPS 12.56%, DreamSim 18.55%, CLIP 2.42% (CLIP misses 3%). Pre-specified
selection-only component gains 5.85%, 6.75%, 12.76%, 3.07% respectively; secondary
point targets pass, but this does not replace the primary freeze. Full progressive
job 37812 is running its final smoke; finalizer 37814 waits for the complete pair.
Evidence export 37813 completed; local transfer/report validation remains.

2026-11-10 02:51 UTC: the final eight-candidate method is frozen before fresh
scoring (SHA-256 `a0a626fcfddec095a1d7b677b496309fccb79ff6b4e446f5127b69368f3358e1`).
All 166 observation-only decisions finished at 02:48:59 UTC; four-worker metric
evaluation is running in Quadro Slurm job **37785**. This job subsequently runs
independent replay, the finalized progressive smoke, and a matched full control
and full Surface-PMC evaluation. Dependent CPU job **37795** verifies, compares
and registers the complete progressive pair. No fresh improvement is claimed yet.
The original 18:07–23:07 UTC research window has elapsed; do not claim on-time
completion. The final CPU check passes **241 tests**, all shell regressions and
synthetic smoke. Final Quadro checks pass real CUDA build/forward/backward, FFT,
SAM/custom Diffusers/full pipeline imports, and exact observable-feature replay.
The earlier updates below are chronological history, not current job status.

19:33 UTC: first 96 development images have all eight generated candidates and
complete observable feature extraction. Joint correction metrics are running;
87-image generation and 166-image extra-seed generation follow in job 37568.
Full calibration-only experiment is verified but misses the goal (LPIPS +0.388%,
VGG +1.019%, SSIM +0.946%, CLIP -0.332%). Fresh truth remains unopened.
Progressive runtime bridge, CLI and Slurm runner are implemented; five anchor/seed/
freeze tests pass. GPU feature-provider equivalence is exact on eight candidates.
Full CPU suite: 233 passed, all shell regressions and synthetic smoke; added
source-weight/control arithmetic tests pass separately. No full progressive
GPU run has been launched yet.


- Prepared 166 fresh cases, excluding all 183 earlier image identities and hashes.
- Implemented observed-residual color calibration, a regularized within-image
  candidate ranker, sampling-agreement features, and compatible full-run derivation.
- L40S smoke 37565 passed CUDA build/forward/backward, real FFT and full imports,
  with eight valid candidates. A study-only cuFFT 10.9.0.58 preload fixes the
  earlier LaMa runtime failure without replacing the pinned environment library.
- Four-GPU/four-worker generation 37566 is active. Quadro job 37564 was cancelled
  because its estimated start was beyond the research deadline.
- DreamSim ensemble is cached and its identity-distance check passes in an
  isolated metric environment. Nine focused refinement tests pass on Windows;
  optional field test skips when scipy is absent from the minimal CPU environment.
- No fresh-test improvement is claimed yet. Development choices are not frozen.

## Surface-PMC controlled experiment completed (2026-11-08)

- Delivered the [13-page report and complete gallery](output/surface-consistency-20261108/START_HERE.md),
  component/selection tables, theory derivations and reproducible source.
- Generated all 732 candidates: 24 development, 72 pilot-test and 87 disjoint
  replication images, four fixed seeds each. No model training was needed.
- Full surface is inconclusive on pooled held-out MAE: gain +0.004310
  [-0.000512, +0.009074]. Local seam gain is +0.007187 [+0.002183, +0.012296],
  with supporting pooled LPIPS; replication MAE and direct superiority to the
  other observable rules remain inconclusive.
- In 51/159 cases the full compatibility score prefers erroneous selected RGB
  to source truth on identical support. Exact filtering/ranking decomposition,
  within-image pair ordering, counterexamples and explicit losses explain limits.
- Smoke **36891**, pilot **36894**, pilot LPIPS **36895**, replication **36897**
  and replication LPIPS **36898** all completed successfully. Cluster replay
  submission **36958** failed before checking due to a missing helper; both
  experiments subsequently passed local exact replay. Separate cluster
  explanatory output agrees with the local pinned calculation.
- All 212 CPU tests, five Bash regressions and synthetic smoke passed. All 732
  candidate metrics, 1,989 policy decisions and 2,745 artifact checks replay;
  maximum raw-score difference is zero. The eight repeated smoke RGB/mask pairs
  match the main run. Archive checksums match remote values.
- Report checks cover 636 test candidates, 732 perceptual input identities,
  318 gallery panels, all table summaries and PDF links/layout. Browser filters
  return 159 total, 84 gains, 75 losses and one fallback. Numerical portability
  changes one pilot boundary-plus-color ablation choice; full/local choices are
  unchanged. Frozen scientific decisions and historical benchmarks are preserved.

## Human reviewer handoff prepared (2026-11-08)

- Prepared a [160-case random audit ZIP](output/human-review-20261108/audit-160/human-review-160-cases.zip) (226,870,269 bytes) and [full 1,390-case ZIP](output/human-review-20261108/census-1390/human-review-1390-cases.zip) (1,920,708,510 bytes), with neutral case IDs, blank six-criterion forms, guides, native source/output images and JSON/CSV export.
- Preserved private identity keys, selection seed/probabilities, source hashes and delivery receipts outside the reviewer archives. No prior assistant labels, report conclusions or numeric performance scores are included.
- Added builder, packet validator and strict human-response importer. Missing/draft responses cannot become negative labels; U/NA retain separate denominators; human–assistant agreement is not inter-human reliability.
- All 15 focused tests pass, including synthetic end-to-end response import without modifying prior labels. Both archives pass exact member-hash checks; 800/6,950 images decode, 640/5,560 native artifact hashes match, and 960/8,340 rating cells are blank.
- Browser checks passed for offline file opening, image loading, blank startup, completion validation, confound exclusivity, co-occurrence details, autosave/reload, JSON import, CSV serialization and wrong-packet rejection. Synthetic browser state was cleared. No human responses have been collected or messages sent.

## Expanded-vocabulary appearance census (2026-11-08)

- Delivered a [13-page report](output/appearance-expanded-20261107/REPORT.pdf), six-type taxonomy, CSV/JSON tables, PNG/PDF/SVG plots and an offline viewer for all 1,390 saved outputs.
- Analyzed all 2,565 targets, retaining 1,175 pipeline failures. Recorded 652 outputs with at least one appearance label; retained per-label unknown/NA counts, severity and confound sensitivity.
- Completed all visual batches and targeted full-source checks; archived the intermediate sample and initial census labels. Confirmed eleven raw-RGB co-occurrence cases, all outside returned masks.
- Independently decoded 321 COCOA ratios; required bins contain 114 targets and 71 saved outputs. Kept expansion-proxy plots separate; included all recorded category groups.
- Fifteen focused tests passed; numerical and artifact checks reproduce census coverage, metric-bin statistics and labels, with 9,300 source-artifact hash checks. PDF layout and viewer filters were checked. Scientific outputs and configurations remain unchanged.
- Packaged all 1,390 panels with the report, tables, source-check evidence and analysis-source snapshots into a CRC-verified 169,950,750-byte [delivery ZIP](output/appearance-expanded-20261107/appearance-expanded-delivery.zip); file hashes and receipt are retained.

## Person 2 appearance-failure deliverable completed (2026-11-07)

Delivered a [12-page report](output/appearance-failures-20261107/REPORT.pdf), six-type [taxonomy](output/appearance-failures-20261107/TAXONOMY.md), exportable performance/occlusion and severity plots, category analysis, and a searchable [120-case audit](output/appearance-failures-20261107/review/index.html). All counts and evidence notes are downloadable as CSV/JSON.

Used the accepted 2,565-target baseline and audited a deterministic, expansion-stratified sample of saved outputs. Recovered COCOA ratios for 321 unique exact-name matches; 114 lie in the requested 20-80% bins. Mask decoding independently confirms the ratios. Generated expansion correlates only weakly with ground-truth occlusion and is reported separately. Appearance labels are a single assistant's judgments, with unknown/NA evidence and sampling weights retained.

The isolated WSL CPU check completed 167 Python tests, five Bash regressions and synthetic smoke. Final report validation verifies input hashes, tables, sampling, taxonomy, links/images and PDF layout; browser filters pass. No new GPU execution, learned-metric recomputation or experiment registration occurred.

## Full baseline accepted and recovered (2026-11-06)

The requested full baseline is complete. Slurm job **31548** completed with exit code 0 in **4h 02m 38s** on four Quadro RTX 6000 GPUs. GPU inventory/FFT, CUDA extension build/forward/backward, full pipeline imports, all source-image checks, the four-worker scored smoke, full evaluation, provenance-required verification and registration all passed. The unchanged scientific contract is `configs/baseline.json` (SAM references, semantic matcher v2, released vocabulary, seed 0).

**2,565 targets: 951 saved completions and 1,614 failures (62.92%).** Saved outputs comprise 796 non-cap termination states and 155 iteration-limited outputs; neither category certifies visual completeness. There are zero missing images and zero reference failures. Overall CLIP is **28.393098 (n=951)**; LPIPS **0.735009**, feature similarity **0.242636** and SSIM **0.303397** each use **n=2,565**, including the configured failure penalties.

Recovered **16,814 files (2,960,571,039 bytes)**. Every file matches a fresh independently computed remote SHA-256, and the recovered run passed local `verify --require-full --require-provenance` against the existing dataset. All 36 captured project source files match the maintained source. See the [accepted report](output/baseline-establishment-20261105/REPORT.md) for per-source results, failure reasons, provenance, recovery evidence and interpretation limits.

The local watcher and slow bulk transfer were retired after cluster completion. Recovery reused only hash-identical bytes and restored 470 PNG encodings with exact remote-hash matches. All artifacts are preserved under `output/baseline-31548/`; earlier progress entries below describe historical states.

## Baseline submitted; GPU runtime validated (2026-11-05)

- Added a staged cluster workflow with runtime validation, separate eight-target/four-worker scored smoke, full baseline evaluation, independent provenance verification and gated registration. A valid job has no three-hour process timeout.
- All 155 Python tests pass in WSL without skips; all five Bash regressions and synthetic smoke pass. Six workflow regression tests establish failure gating; Windows plan paths were also checked.
- All 2,565 targets pass local source-image preflight with zero missing inputs. Prepared an 88-file source-only upload manifest and resolved contracts.
- The user authorized agent-operated remote work and AGENTS.md now records it. Uploaded all 88 files, verified every SHA-256, and backed up the old remote source. Runtime job 30425 completed successfully in 8m04s: real CUDA extension build/forward/backward, full imports and pinned package checks passed (PyTorch 1.13.1, CUDA 11.7, Quadro RTX 6000). Downloaded evidence includes 36 verified source snapshots. The L40S smoke 30428 failed in cuFFT; isolated diagnostic 30440 reproduced this compatibility failure. Added an FFT integration check, which passed with the other GPU checks in Quadro job 30435. New Quadro-only smoke 30436 is queued for resources, followed by full four-worker baseline 30426. Failed artifacts and updated validation are downloaded. Full scoring/registration has not occurred.
- Preliminary validation 30424 found no executable Python in the node-local `/dev/shm` cache and failed before inference. Replaced pending baseline 30423 with the persistent-cache workflow. The four-GPU allocation is resource constrained; unrelated jobs are unchanged.
- Evidence and runnable commands: [baseline launch package](output/baseline-establishment-20261105/LAUNCH.md).

## Experiment artifacts recovered and verified (2026-11-05)

- Recovered 70,012 files (11.90 GB) into `output/artifact-recovery-2026-11-05/downloads/`; every file matches an independent remote SHA-256. See [the recovery report](output/artifact-recovery-2026-11-05/REPORT.md).
- Verified six full historical runs and two 20-target runs; normalized only absent metadata in separate legacy LISA verification copies.
- Recovered 275 additional older SAM target records (82 successful completions) through verified reuse lineage.
- Confirmed 941/946 shared annotated/semantic-v2 completion PNGs are identical and 622 targets have changed reference artifacts.
- Preserved original reports/artifacts and scientific code. No new GPU experiment or ledger registration occurred; historical execution provenance remains unavailable.

## Manager briefing completed (2026-11-05)

- Prepared a 16-slide offline presentation and PDF, speaker notes, an eight-run CSV appendix and evidence index in `output/manager-briefing-2026-11-05/`.
- Connected the architecture and runtime repairs to historical coverage gains, remaining failures, protocol limits and the retained Surface-PMC proposal.
- Reused the completed eight-CSV audit and fresh 149-test / five-shell-regression / CPU-smoke checks. Confirmed slide fit, image loading, links, PDF page count and unchanged maintained source/config/test/helper hashes.
- Recommended recovering missing detailed artifacts before claiming full verification of newer runs; proposed an equal-budget candidate-selection experiment. No scientific implementation or remote action was performed.

## Surface-Consistent PMC knowledge retained (2026-11-05)

- Read the supplied `ChatGPT - Assess Reproduction Feasibility.html` and retained the surface-selection idea in [a detailed research reference](docs/SURFACE_CONSISTENT_PMC.md), with a [focused source extract](docs/research/SURFACE_CONSISTENT_PMC_SOURCE.md) preserving the original Vietnamese wording and formulas.
- Recorded boundary/color/texture scoring, candidate generation and alignment, example weights, ablations, hidden-ground-truth evaluation, resource assumptions, optional guidance and unresolved failure cases. Distinguished the proposal from measured results and the existing visible-reference protocol.
- Added a local `AGENTS.md` recall pointer and documentation-index link. Preserved the original HTML. This was documentation only; no implementation, inference or experiment was performed.

## Conservative repository cleanup (2026-11-05)

- Reviewed maintained Python modules, entry points, Bash/PowerShell helpers, configuration and regression tests; inventoried dataset/output helpers and retained their provenance context.
- Added a documentation index, package/entry-point map, test guide and archive index. Moved `paper_summary.md` to `docs/PAPER_SUMMARY.md`, completed plans to `docs/archive/plans/`, and the historical root execution log to `logs/run_amodal.log`.
- Removed 76 generated bytecode files and six generated root egg-info files from the workspace by moving them into an external rollback directory. Preserved `.venv` and its installed metadata; no permanent deletion was performed.
- Verified all 63 original code/configuration/test files are byte-identical. The 29,318-file inventory of `datasets/`, `output/`, `input/` and `.venv/` retained the same paths, sizes and modification times.
- WSL Python 3.10: all 149 tests, all five Bash regression scripts and CPU smoke passed. Windows baseline: 149 tests ran with nine expected skips; installed CLI and smoke also passed after cleanup. See `session.md` for evidence and rollback location.

## Completed

- Adapted the setup workflow to this fixed amodal evaluation task while retaining established root entry points and cluster paths.
- Added editable package installation, an `amodal` CLI, inherited JSON configs, and four-GPU Slurm command planning.
- Added execution/source/environment snapshots, CPU artifact verification, and an explicit full-run experiment ledger.
- Added an offline synthetic smoke that exercises the evaluator, failure bookkeeping, metric caches and resume.
- Removed test dependence on locally downloaded inputs by supplying pinned annotation/source fixtures.
- Added source/result transfer helpers with dry runs, reusable cookbooks and current research notes.
- Tightened checks for stale failure statistics, changed attempt configs, failed-run target identities, concurrent registration, and frozen LISA comparison references. SAM references cannot be relabelled as LISA.
- Validated 149 Python tests on Python 3.10 and Windows Python 3.14 (nine expected Windows skips), all five Bash regression scripts, and the offline smoke.

## Running experiments

No GPU experiment or remote job was started during the rebuild. Existing datasets, model caches and historical experiment outputs are preserved.

## Next steps

1. Review the generated cluster plan and transfer the updated source yourself.
2. Validate the pinned GPU runtime and complete a new two-target smoke.
3. Run and verify a full baseline, then register it explicitly.
4. Evaluate the annotation-vocabulary hypothesis in a separate output directory.

Final local validation evidence and commands are recorded in `session.md`.

## Annotated-vocabulary Surface-PMC started (2026-11-12)

Frozen 204 targets from 192 images, excluding 477 previous controlled-study images.
Source quotas and every target are preserved. Deployed 54 checked source/design files
with backup; canonical annotated-reference copies are being verified. CPU suite
passed 267 tests with six optional-dependency skips; repaired candidate-segmenter
regression passed. Final checks with the research replay environment are running.

07:13 UTC: jobs 39473-39479 submitted. Canonical reference copies match all
10,261 hashes each. Final CPU suite: 268 tests, no skips; shell regressions and
smoke passed. GPU extension, real FFT and pipeline import checks passed.
Control cohort: 89/204 records, no worker tracebacks. Pool diagnostic was frozen
before surface generation and changes no primary decisions.
