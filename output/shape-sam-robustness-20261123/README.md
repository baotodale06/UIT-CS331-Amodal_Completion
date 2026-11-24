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

# Isolated shape-only SAM robustness development

The validation-only SAM diagnostic in `../shape-validation-sam-20261123/REPORT.md`
shows a 0.077358 full-IoU and 0.066313 hidden-IoU absolute reduction from noisy
input masks. This separate experiment addresses that input mismatch. It does
not modify `shape-v2-slurm-20261123`, its checkpoint, holdout or benchmark.

Training-mask job **45661 completed** in 5m58s on four Quadros. It produced all
9,494 training masks from 1,554 images, with zero empty masks. Four shard
inventories verify image separation and every output hash. No validation,
holdout or benchmark objects were included in this mask-preparation job.

Training preflight **45662** is running; **45663** is dependency-pending as of
12:26 UTC on 23 November. Consult scheduler accounting for current status.
`training-jobs.json` preserves both exact submission commands, including the
30-minute maximum/20-minute minimum training allocation before maintenance.

The immutable training snapshot and configuration specify 20 epochs from the
completed v2 epoch-47 checkpoint, global batch64 on four Quadros, AdamW learning
rate 0.00005, seed 20261124, FP16 training and FP32 native validation. Inputs mix
75% SAM masks and 25% annotated masks; the shape prior can correct false-positive
support because inference does not hard-restore its input mask. True annotated
visibility defines hidden supervision and scoring. RGB remains restricted to
input support; there is no appearance ranker, RGB repair or combined method.

Each epoch selects on the existing 1,264-object/217-image validation split using
native image-mean hidden IoU, with full IoU as the tie-break. Rank-zero BatchNorm
buffers are synchronized before distributed native validation and checkpointing.
This is exploratory development informed by validation. No pristine confirmation
or 3% benchmark success may be claimed from these results.

Five new data/geometry tests passed locally and in pinned cluster Python; the
four-GPU preflight must verify real optimization, FFT, native validation and
checkpoint round trip before training can start. Subsequent work must recover
the final receipt/checkpoint, independently replay the selected native masks,
and evaluate any frozen revision with full failure/coverage and provenance
accounting. The original v2 full benchmark remains in progress.
