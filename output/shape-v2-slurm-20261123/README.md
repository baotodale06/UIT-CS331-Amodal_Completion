## Shape-only progress (2026-11-23 13:04 UTC)

The >=3% full-benchmark goal remains ACTIVE and unachieved. Only shape-aware
work belongs to this continuation; the other study was left untouched.

Frozen v2 has 222/2,565 terminal records: 112 success, 57 ambiguous_target,
38 no_completion and 15 not_detected. Job 45658 timed out at 12:56:30 UTC after
48m25s; controller 45659 completed in 2s and export follower 45660 completed.
The unchanged four-Quadro continuation is 45673, controller 45674 and export
follower 45675. They are pending across active maintenance until 25 November
13:00 UTC / 20:00 Vietnam. Independent maintenance-continuation-replay.json
verifies all 222 retained records, the earlier 166 records and all 299 frozen
source files. No frozen inference or control/reference change occurred.

Fixed-prefix scoring 45668 completed in 132s on one Quadro. Independent local
replay of the pre-existing 193-target VG snapshot gives CLIP +1.520%, LPIPS +0.882%,
feature similarity +3.142%, SSIM +0.605%, with 96 saved outputs in both arms and
all 193 failures/successes retained for appearance metrics. This is a nonrandom
partial diagnostic, not a full benchmark or goal completion. Same-GPU control
rescoring differs by at most 1.1921e-7. Report:
output/shape-v2-slurm-20261123/snapshot-quality-recovery/REPORT.md.

Robust fine-tuning 45663 and native replay 45664 completed. Epoch 7 was selected;
independent local replay verifies all 1,264 objects / 217 images. Full IoU
0.811374->0.793666 (-2.182%); hidden IoU 0.370333->0.391652 (+5.756%). The fixed
SAM-support union gives full 0.795808 and hidden 0.389935, so it does not resolve
the full-shape regression. Reports are under
output/shape-sam-robustness-20261123/report/ and support-clamp-ablation/.
Its collector finished normally. Best checkpoint SHA-256:
3bb1a47ddd952824a960fb82867159c366c1c3e4a53a53bab1e06585adbd64a9.

The next isolated experiment freezes the original v2 encoder/BatchNorm buffers,
trains only the decoder at LR 1e-5 for 20 epochs, and includes epoch zero in selection.
An epoch must retain initial native full IoU before it can win on hidden IoU.
SAM support is restored, with no appearance loss or RGB correction. No new
holdout/benchmark checkpoint selection is allowed. The observed partial VG
benchmark is disclosed as iterative research, not untouched confirmation.

Initial constrained preflight 45671 FAILED before optimization; dependent
training 45672 was automatically CANCELLED. Diagnostic 45676 completed in 11s
and isolated a cuDNN arithmetic difference: 79 objects replay exactly using
cudnn.deterministic=False; deterministic=True changes one object's thresholded
mask geometry. Preserve the exact replay guard. The repaired validation helper
matches baseline arithmetic and restores the training flag afterward.
Old source snapshots/logs remain under output/shape-sam-constrained-20261123/.

Repaired isolated chain output/shape-sam-constrained-v2-20261123/:
preflight 45677 -> training 45678 -> native replay 45679, all pending. Three focused
CPU tests pass in pinned cluster Python, including encoder/BatchNorm retention,
checkpoint eligibility and arithmetic-flag restoration. Real repaired GPU
preflight is still unvalidated. Current collector exec 34999 / PID 2860 is alive;
v2 collector PID 14576 remains alive. Check before restarting either.

Next actions: preserve both queued chains; inspect real preflight/training/native
outcomes after maintenance; let collectors recover artifacts and independently
replay statistics. Run scripts/build_shape_constrained_report.py only after
native-replay.json and final accounting exist. An epoch-zero result is not a
new model and must not be promoted. No constrained benchmark adapter or full
constrained benchmark has been submitted. Complete all 2,565 frozen-v2 targets,
full verification/registration, native audit and scripts/build_shape_v2_report.py
before assessing the four 3% gates. The goal stays open if any gate fails.

## Shape-only training and benchmark audit (23 November 2026, 12:40 UTC)

The isolated SAM robustness preflight 45662 passed in 74 seconds on four Quadros;
fine-tuning 45663 is RUNNING and has completed 16/20 epochs. It started at
12:27:14 UTC with a 30-minute limit. Native checkpoint replay 45664 waits for
successful training. Local recovery observer exec 3756 / PID 26664 runs
`scripts/collect_shape_robustness.py --watch`; check it before restarting. It
recovers the selected checkpoint and native/training masks, then independently
replays native geometry and bootstrap intervals. Training source and policy are
fixed; neither training completion nor a benchmark improvement is established yet.

The full Linux CPU check passed 389 tests (five optional skips), six Bash
regressions and CPU smoke. See the isolated study's `validation.json` and
`check-linux.log`. New report preparation is
`scripts/build_shape_robust_report.py`, usable only after verified final recovery.
`scripts/audit_shape_support_clamp.py` prepares a separate exploratory CPU
counterfactual on the same selected checkpoint (raw prediction OR SAM support).
It has not run and does not change the raw-policy training/native experiment.
No holdout, RGB repair, Surface or combined method is involved.

Frozen v2 CPU audit 45665 COMPLETED in 24 seconds. The 193-target snapshot
contains 96 saved outputs and verifies 195 pools / 1,560 candidates / two valid
slot-zero fallbacks. Native score arithmetic, selected RGB, target seeds and
original references passed. See `partial-audit-45665.json` under the v2 study.
This is an engineering check, not full-benchmark verification or scoring.
The existing v2 collector (PID 14576) and continuation/export jobs remain active.
The latest dated recovery is `recovery/20261123T123620.054892Z`, with newer counts
in `collection-status.json`. All four >=3% full-benchmark gates remain open.

## Full shape-only inference and final report readiness (23 November 2026, 10:48 UTC)

Job 45636 remains RUNNING on four Quadros; the dated recovery has 34/2,565
terminal records, including 10 saved outputs. No full quality scores exist.
All nine inspected no-completion records stopped through existing released
pipeline rules (six no-occlusion detections, three masks too small), with no
error fields. This diagnosis does not change the frozen inference or failure
accounting. The four worker log tails showed diffusion progress without recent
tracebacks or CUDA out-of-memory messages. Collector PID 14576 / exec 69625 was
confirmed live; continuation/export/backfill helpers remain dependency-pending.

`scripts/build_shape_v2_report.py` is ready for the completed audited recovery.
After `final-recovery.json` and `full-metadata-replay.json` exist, run
`python scripts/build_shape_v2_report.py`. It binds the report to the frozen
policy, native audit, registered full score table, local statistical replay and
final allocation accounting. It writes full/unseen metric tables, PNG/SVG
confidence-interval figures, native benchmark and prior geometry including
occlusion strata, failure/CLIP coverage, cost and a checksum receipt. The 3% point gates and uncertainty remain
distinct. Two focused tests passed; attempting to publish this still-incomplete
real study correctly refused before creating any report. No final report or
v2 gain claim has been published. See `report-readiness.json` under the study.

The goal remains ACTIVE. Continue the existing live jobs; do not start duplicate
inference or retune the frozen policy. The pending final report remains required
after the full collector completes.

## Shape-only v2 smoke verified; full benchmark running (23 November 2026, 10:43 UTC)

Smoke job **45635 COMPLETED, exit 0:0**, in 18m29s (1.232222 allocated
GPU-hours). All four sources saved a completion. Cluster selection audit and
local native-image replay agree on **9 pools / 72 candidates / 0 fallbacks**.
The 41,134,956-byte archive and 292 files are recovered; see
`output/shape-v2-slurm-20261123/smoke-native-replay.json`. Native mask/prior scores,
selected RGB, seeds, completion hashes and exact original reference bindings
passed. These are engineering checks, not benchmark improvement measurements.

Full benchmark **45636 RUNNING** on four Quadros, started 10:37:34 UTC.
Slurm backfill reduced its effective limit to **90 minutes**, ending 12:07:34 UTC.
The attempt to extend the running job to 140 minutes was rejected by Slurm
(`Access/permission denied`); no frozen source or scientific setting changed.
At the dated 10:43 recovery there were 12/2,565 terminal target records: one saved
output, six ambiguous targets, three no-completion records and two non-detections.
There are no full quality scores yet; the all-four >=3% goal remains ACTIVE.

Existing continuation controller **45637** preserves completed records and resumes
infrastructure interruptions. New CPU export follower **45642** follows each
controller and exports the full audited benchmark from the separate postprocessing
source; it never resubmits inference. New one-time helper **45643** can shorten
the first pending continuation to use remaining time before the confirmed
Nov 23 13:00 UTC-Nov 25 13:00 UTC maintenance. It only changes a pending job's
allocation duration, records Slurm's response, and leaves already allocated jobs
alone. Subsequent unchanged 24-hour continuations remain the existing controller's
responsibility. Application failures still require review.

Local collector **exec 69625 / PID 14576** runs
`python -u scripts/collect_shape_v2.py --watch`, logging to the study's
`collection-watch.log`. Check process/log/`collection-status.json` before starting
another observer. It polls without resubmitting, tolerates observation outages,
and waits for final export plus terminal accounting before checksum recovery,
safe extraction and local full metadata/statistical replay. A failed 3% gate is
retained as failure to meet the goal. The cluster export persists if the local
collector is interrupted. The old training observer has completed; do not restart it.

Twelve distinct local operational tests passed without skips; five follower tests
also passed in pinned cluster Python. See `operational-validation.json`,
`benchmark-launch-milestone.json`, `finalization-jobs.json`, `backfill-job.json`
and `latest-recovery.json` under the study. Recovery now uses UTC Slurm timestamps
and includes the frozen 2,017-target unseen inventory and all operational jobs.
Next finish all 2,565 targets, score, verify/register, audit native geometry and
full/unseen intervals, recover the final archive, and report all four gates with
complete failure/CLIP coverage and final allocation accounting. No tuning of this
frozen model or its evaluated holdout is authorized by these operational changes.

## Shape-only v2 training complete; frozen model smoke running (23 November 2026, 10:19 UTC)

Job 45632 COMPLETED successfully in 39m56s on four Quadro RTX 6000 GPUs
(2.662222 allocated GPU-hours). All 60 epochs completed. Validation selected
**epoch 47**. `training/best.pt`, `training/last.pt` and the final training receipt
are recovered locally and hash/contract verified. The local recovery observer
(exec 28679 / process 37252) completed successfully; do not restart it.

Native geometry job 45634 COMPLETED. Validation: 1,264 objects / 217 images,
full IoU 0.888731 and hidden IoU 0.436647. Reserved holdout: 1,183 objects /
195 images, full IoU 0.874936 and hidden IoU 0.421346 (166 images with hidden
truth). These are image-grouped native shape-prior metrics, not benchmark gains.
Policy freeze SHA-256:
`6078e500c236095a537000d2d3f90838253ab7542d39c69d0fafe67db150523b`.
Best-checkpoint SHA-256:
`f3caa75ac40f901a7595fffa2c199221b83e7fe5a57192f34e2b2317ffe937b7`.
No tuning after reading holdout; preserve this frozen experiment.

Real four-GPU smoke 45635 is RUNNING; full benchmark 45636 and continuation
controller 45637 remain dependency-pending. Job 45636 retains the documented
150-minute maximum / 90-minute minimum before Nov 23 13:00 UTC maintenance.
The >=3% all-four-full-metric goal is ACTIVE and has not been demonstrated.

New isolated `output/shape-v2-slurm-20261123/postprocessing-source/` on the
cluster contains the updated v2 metadata exporter and local replay verifier.
Its 302 packaged files passed hash checks; ten focused metadata/export tests
passed in pinned cluster Python without skips, plus three local replay tests.
The training/evaluation source snapshots were not edited. Final export must use
this postprocessing source, preserving the frozen inference source. No complete
v2 benchmark or native/full metadata replay exists yet.

Read `training-recovery-verification.json`, `geometry-recovery-verification.json`,
`completed-training-milestone.json`, `postprocessing-validation.json` and
`latest-recovery.json` under the study. Next inspect smoke, continue full scoring,
verification/registration/audit, export metadata, and run
`scripts/verify_shape_v2_metadata.py` against the safely extracted full archive
and dated final recovered snapshot. Finish report/visuals and final allocation
accounting. Preserve all targets, failure penalties, CLIP coverage, 2,017 historical
unseen targets, 321-target native amodal geometry and all four improvement gates.

## Operational update, 10:07 UTC

Training has completed 44/60 epochs and is still running; no quality result is
available yet. Job 45636 now has a 150-minute maximum and 90-minute minimum to
use available time before maintenance. `allocation-amendment.json` records the
scheduler change; all scientific settings and frozen source files are unchanged.

`latest-recovery.json` points to dated recovered evidence and provisional cost.
The live local observer `watch_training.py` (exec session 28679) will recover
`training/best.pt`, `training/last.pt` and the final receipt after job 45632 exits
successfully. Read `training-recovery-watch.log` before restarting an observer.
Manual recovery: `python scripts/recover_shape_v2.py --checkpoints`.

New local recovery/export checks passed: 356 tests, 11 optional dependency skips,
six Bash regressions and CPU smoke. See `recovery-export-validation.json`.
The updated v2 metadata exporter is local postprocessing code and is not part
of the already queued evaluation source. It must be transferred separately for
final export; do not edit the policy-frozen source to add it.

# Shape-only Slurm continuation — 23 November 2026

The current user explicitly authorizes Quadro, L40S or Kaggle training for speed.
This supersedes the previous Colab-only platform restriction. The original Colab
bundle remains intact. The requested >=3% relative gain on all four existing full
benchmark metrics remains unachieved; training readiness is not a quality result.

## Current jobs

- GPU preflight: **45631**, completed successfully on four Quadro RTX 6000 GPUs.
- Training: **45632**, running on four Quadros, 60 epochs; over 25 epochs finished
  at the latest live check. Consult `training-progress.json` for the recovered snapshot.
- Geometry validation and policy freeze: **45634**, waits for completed training.
- Four-worker real inference smoke: **45635**, waits for geometry checks.
- First full-benchmark slice: **45636**, waits for smoke; 90-minute allocation.
- Timeout continuation controller: **45637**, waits for the benchmark slice.
  It preserves result hashes and queues unchanged 24-hour continuations only for
  timeout/preemption/node failure. Application failures require diagnosis.
- Source snapshot: remote `~/Vu_Hung/cv_nangcao/output/shape-v2-slurm-20261123/source`.
- Dataset: verified original 297,331,784-byte training ZIP; 9,494 training and
  1,264 validation objects. The 1,183-object / 195-image holdout is transferred in
  a separate evaluation-only ZIP and directory. It is never read by the trainer.
- Cluster maintenance begins 13:00 UTC / 20:00 Vietnam time on November 23 and
  ends 13:00 UTC / 20:00 Vietnam time on November 25. Training has a 2h45 limit;
  resumable epoch checkpoints preserve completed progress if interrupted.

These are initial submission statuses; consult `squeue`, `sacct` and the saved
training receipt for current progress. Existing combined work is outside this
continuation and has not been modified.

## Validation

The final local check passes 352 Python tests (five optional torch skips), six
Bash regression scripts and CPU smoke. All 29 focused tests pass without skips
in the cluster's pinned torch environment. The four-GPU training preflight
verified actual FFT, optimization, validation and checkpoint serialization.

Job 45633 passed the real GroundingDINO CUDA build/forward/backward, cuFFT and
full pipeline imports, then exposed two test failures: missing fixture files in
the isolated package and PyTorch 1.13's experimental weights-only loader rejecting
floating-point validation metadata. Both were repaired before the passing
29-test replay. Its original failed job/log is preserved; it is not labeled a
successful overall job. The loader verifies checkpoint, training-source and data
hashes before using the pinned runtime's compatible loading path.

[validation.json](validation.json) records evidence hashes. Benchmark results and
the 3% improvement are still pending.

## Training contract

[The separate Slurm configuration](../../configs/research/shape-v2-slurm-20261123.json)
preserves the architecture, data splits, loss, seed, 60 epochs, global batch64,
checkpoint selection and benchmark endpoint. DDP uses four GPUs with per-GPU
batch16 and per-rank encoder BatchNorm. FP16 with gradient scaling supports the
Quadro architecture; the Colab specification uses BF16. These are recorded
implementation differences, not asserted equivalent training trajectories.

The trainer checks real FFT on every GPU, hashes every training asset, checks
that holdout is absent, records actual devices, and saves optimizer, scheduler,
scaler and per-rank torch/CUDA RNG state. The preflight exercises optimization,
validation and checkpoint serialization. Preflight checkpoints cannot pass the
completed-training gate. Final checkpoint selection uses validation geometry
only; benchmark scores are never used for selection.

`source-manifest.json` records transferred code; `submit.sh` retains exact
resource requests. Existing cluster packages, datasets and historical runs are
reused without overwriting them. No installation into the pinned GPU environment
is required.

## Queued evaluation

Recover completed `training.json`, `best.pt` and `last.pt`. The queued geometry
job evaluates native-image validation, freezes the selected model and all source
hashes, and evaluates the reserved holdout. The smoke copies and verifies the
original PMC's exact scoring references and checks four sources on four GPUs.
Full evaluation uses a new output directory, retains all 2,565 targets, then
verifies/registers and independently replays selections, full/unseen metric
intervals and the 321-object amodal geometry subset. CLIP bootstrap retains each
arm's established successful-output denominator. Allocation cost must be added
from final accounting; model quality cannot be inferred from a partial run.

The first benchmark slice is intended to use time before maintenance. Unfinished
inference continues with identical sources, checkpoint, seed, references and
configuration. The controller does not repeat completed benchmarks merely because
the four-metric gate fails. A failed gain gate leaves the research goal open.

No Surface ranker, appearance loss, RGB correction or combined run belongs to
this experiment. Any remaining work stays open until measured results pass the
requested endpoint; this launch does not establish an improvement.
