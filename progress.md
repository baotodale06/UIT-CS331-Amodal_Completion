## SSH restored; GPU maintenance still active (24 November 2026, 12:35 UTC)

SSH and fresh Slurm accounting work again. Existing collectors14576/2860/22844
are identity-verified alive and have each resumed successful polling after12:33UTC.
This supersedes the preceding SSH/DNS failure and stale-accounting statements.
All12 shape-only jobs remain PENDING. Roots45673(v2 full),45677(constrained
preflight),45680(calibrated native) wait for reserved maintenance; downstream
jobs have normal dependencies. Both GPU nodesgebru/gus and all CPU nodes remain
in maint. The active all-node reservation ends25November13:00UTC/20:00Vietnam.
No training/evaluation job or collector was restarted, duplicated or canceled.

V2 retains222 historical target records; no new full benchmark output is ready.
Private Kaggle147113223/version1 remains QUEUED; observer23064 is alive.
The completed local probability training and fixed-offset diagnostic remain closed
with no eligible new checkpoint. Continue the preserved Kaggle study and Quadro
chains; full2,565-target >=3% on allfour metrics remains unachieved.

Evidence: `output/shape-cluster-restored-20261124/restoration.json`.

## Fixed final-epoch diagnostic completed: no retained offset (24 November 2026, 12:31 UTC)

Recovery completed all four phases at12:30:09UTC: real GPU preflight, independent
preflight replay, full diagnostic, and independent full native-mask replay/report.
All1,264 objects/217images passed, covering8,848 exported masks (six variants and
the separate zero-offset reference). Original Quadro pixels and final zero-offset
repeats match exactly; recorded epoch20 geometry, hashes, identities, support, crop
bounds, empty masks, monotonic offsets and summaries all pass. Processes5580/28752
have exited. Original failed attempt, policy/source snapshots and parent remain intact.

None of offsets[0,0.5,1,1.5,2] passes both unchanged geometry floors. All fail hidden
IoU: the largest is0.3609508314 at+1, below0.3703334757; that offset also misses the
full floor. No new checkpoint or offset is promoted. Better unshifted Brier/NLL
does not establish hidden geometry retention; probability replay checks bounds and
aggregation only. This is reused validation, not a full-benchmark improvement.

Report: `output/shape-probability-final-recovery-20261124/full-replay.md`.
Assessment/chart: `assessment.json` and `diagnostic.html` in that directory.
Independent replay SHA256:6685eadd221a1ad4d4e849d8325173c15ea6581aab7abc6470d2ac1afea3d254.

Kaggle147113223/version1 remains QUEUED as of 2026-11-24T12:31:38.670884+00:00;
existing observer23064 and three cluster collectors14576/2860/22844 are identity-
verified alive. Current quota:1.492533 GPU-hours used of30, zero reserved. SSH still
cannot resolve the cluster host; an independent public resolver also returned no
host address. Last successful cluster accounting remains23November19:10UTC.
No jobs were resubmitted, canceled or changed. Preserve all three Quadro chains
and the calibrated-initialization Kaggle study. Maintenance is scheduled to end
25November13:00UTC/20:00Vietnam. The >=3% all-four-metric full2,565-target goal
remains unachieved; only eligible verified training may proceed to Quadro admission.

## Local probability run closed; fixed diagnostic recovered (24 November 2026, 12:24 UTC)

All seven schema8 parent phases completed successfully at 12:21:19 UTC. All 20 epochs
completed, with 189,880 sampled objects and zero skipped optimizer steps. Independent
selection replay finds no trained epoch meeting both unchanged floors: maximum trained
hidden IoU 0.3476279268 is below 0.3703334757. Selected epoch remains zero. The complete
1,264-object/217-image native replay reproduces original Quadro masks with zero differing
pixels. Parent3532 and trainerWSL502 exited normally; do not restart them. Report:
`output/shape-probability-workers4-20261123/report/REPORT.md`.

Existing diagnostic observer36128 started after parent closure and failed before inference:
its support-only training bundle omitted sam-prediction.png. The failed attempt is preserved.
A separate recovery uses the complete original native validation export, whose result hash
and all 2,528 required mask artifacts match the frozen references. Policy, sources, epoch20
checkpoint and offsets [0,0.5,1,1.5,2] are unchanged. Three-object GPU preflight and its
independent replay passed; full diagnosis/replay is proceeding under WindowsPID5580.
See `output/shape-probability-final-recovery-20261124/execution.json`. Do not duplicate it.

Kaggle147113223/version1 remains QUEUED; observer23064 is alive. Cluster collectors
14576/2860/22844 are alive; fresh SSH fails DNS resolution. No jobs were duplicated or
changed. Maintenance ends 25November13:00UTC. No new benchmark admission or verified
>=3% four-metric full-benchmark result exists. Continue the live diagnostic, Kaggle observer
and preserved Quadro chains. This is iterative research on reused development data.

## Windows execution gap confirmed; final local epoch active (2026-11-24T12:08:06.345673+00:00)

Windows Power-Troubleshooter event1 records sleep at23November19:18:48UTC
and wake at24November11:37:41UTC, with Power Button as the wake source.
A separate Kernel-Power event42 reports Button or Lid. This replaces the earlier
statement that machine suspension was only plausible. The low-power interval
explains the local training/observation gap; it does not prove the remote Kaggle
session's terminal state or recover its missing final checkpoint. No system power
policy was changed. The saved-batch recovery addresses remote output persistence.

Local training completed19/20epochs, selected0;
epoch20 has started. Epoch19 native full/hidden IoU is
0.822886756879688/0.3446480982736047, below the hidden floor0.3703334756537925.
TrainerWSL502, parentWindows3532 and diagnostic watcher36128 remain live.
Kaggle batch147113223/version1 isQUEUED at2026-11-24T12:07:13.352352+00:00, observed by
worker23064. Existing v2/constrained/calibrated cluster collectors14576/2860/22844
are alive and retrying after SSH/DNS failures; no duplicate was started. Preserve
the separate combined collector26540 and its jobs. This is a verified wait for
training, not evidence that the >=3% full-benchmark goal has been achieved.

Evidence: output/shape-local-execution-gap-20261124/milestone.json and
windows-power-event.json. The full goal remains ACTIVE and unachieved.

## Private saved-batch recovery submitted (2026-11-24T11:57:36.645845+00:00)

Kaggle accepted private notebook ez4kacchan/shape-calibrated-batch-20261124,
version1, kernel147113223, at24November11:53:19UTC. Latest observed state is
QUEUED at2026-11-24T11:57:00.037427+00:00; this does not establish optimizer execution.
Server verification confirms the exact submitted source, T4 allocation request,
private visibility and only the two intended private input datasets. The new
parent-assets dataset13635545 is private/version1; the existing isolated data
bundle is reused. No benchmark or reserved holdout data were uploaded.

This is recovery of the frozen schema9 training after session373179451 became
unreachable and was explicitly retired. All training sources, seed, loss,
20epochs and admission policy are unchanged. The driver changes only its
execution identifier. A synchronous batch wrapper writes terminal status and
a remote SHA256 inventory before Kaggle saves outputs. Synthetic success and
exit7 cases both preserved logs and verified all exported file hashes.
Observer launcher36700/worker23064 automatically downloads the
terminal version, checks remote hashes, and runs the unchanged independent
native-mask/selection replay with batch-specific identity bindings. All13
operational files are frozen in observer-manifest.json. Do not rerun preparation,
upload, submission or activation blindly. Revalidate handles before recovery.

Local original-initialization training has18/20epochs,
selected0; its fixed diagnostic still waits for all parent
phases and process exit. No newly trained checkpoint has passed admission.
The >=3% all-four-metric full2,565-target goal remains ACTIVE and unachieved.
See output/shape-calibrated-batch-20261124/launch-milestone.json and README.md.

## Kaggle session recovery and current evidence (2026-11-24T11:52:21.233279+00:00)

Supersedes the earlier live-session and queued-batch status. Following the
observation gap after 23 November19:18UTC, session373179451 could not be reached:
proxy requests timed out and its saved direct endpoint returned404. Kaggle's
session-specific archive was a valid empty22-byte ZIP; persisted logs were empty.
Quota showed1.492533 GPU-hours used of30 and zero reserved. These facts do not
establish completed training. Only two completed epochs were recovered, selected0;
no final checkpoint or terminal training export was recovered. Machine suspension
is plausible but unconfirmed. Kaggle accepted cancellation of this specific
unreachable session at11:50:10UTC on24November, before a replacement run.
The verified local collector7732 was stopped; the success-only release watcher
was given needs_review and must exit without a second cancellation. Preserve
all old observations and negative evidence. See unreachable-retirement.json.

Independent initial-mask replay completed24November11:37:55UTC: all1,264
objects/217images reproduce the calibrated local parent, zero differing pixels.
Full/hidden IoU are0.8132978430449074/0.38787438514648803. This checks the initial
parent only. Full CPU verification completed11:39:07UTC:485 Python tests with24
optional skips, six Bash regressions and CPU smoke passed. No rerun is needed.

Older probability batchv1 and constrained batchv2 are ERROR, recovered and
not eligible. Both diagnostic.log files fail torch.cuda.is_available() before
optimization. Do not restart their collectors as queued jobs. The library-path
issue is a plausible cause, not a newly proven explanation for each batch.

Local probability training is active: 17/20epochs,
selected0. All54 frozen operational
files still match. Its fixed final-epoch offset diagnostic remains queued behind
the parent's seven phases and independent replay. A separate private saved-batch
recovery is prepared at output/shape-calibrated-batch-20261124. It preserves the
schema9 training code, seed, data, loss,20epochs and admission policy; only the
execution identifier changes. Consult that directory's submission receipt for
actual submission; preparation alone is not a launch. No new checkpoint has
been admitted. Earlier datasets and validation/holdout observations remain
iterative research, not untouched confirmation.

Current SSH attempts fail DNS resolution for the cluster, so queued chains have
not been freshly confirmed. No cluster jobs were changed. Preserve the last
accounting and maintenance window ending25November13:00UTC/20:00Vietnam.
The all-four-metric3% improvement on2,565 targets remains ACTIVE and unachieved.
Evidence: output/shape-calibrated-training-20261123/recovery-milestone.json.

## Kaggle optimizer training established (2026-11-23T19:16:27.008499+00:00)

Fresh session373179451 is executing the separate schema9 calibrated-initialization
study on two T4s. CUDA FFT and all four focused tests passed in the real pinned
runtime. The two-step optimizer preflight passed with zero skipped steps.
Full initial validation repeated exactly and matched all1,264 local calibrated
parent pixel hashes. Full/hidden floors are0.8132978430/0.3878743851.

Actual training: 1/20epochs complete; epoch1 took
116.91seconds. Latest full/hidden IoU is
0.8203456954/0.3576445141.
Selected epoch remains0; a higher full IoU alone is
not eligible. Retain the fixed20-epoch run. Frozen driver SHA256
4864b70a5f1ffe4f3b60f38a8bdde7405ccfd20c162849a36decb3db9099f348;
training manifest8530f782ab5a18c965caf2668acb361b2e395dab4b75084c4f53fa304ca67139.

Observer7732 (launcher12064) monitors remote wrapper150/child151, exports terminal
artifacts and invokes the independent CPU mask/selection replay. Success-only
release watcher launcher24928 verifies all recovered files and no live owned
remote processes before releasing this session. It leaves failures for review.
Do not rerun allocation/upload/preparation/launch/activation helpers blindly.

Prospective admission-policy.json requires a fixed, nonzero selected checkpoint,
exact repeats within Quadro, explicit cross-device differences and both unchanged
training geometry floors on Quadro. No reselection from Quadro or benchmark
scores is allowed. Policy was frozen before full training results; the immutable
pre-result snapshot is admission-policy-observed-state.json. Verification receipt
documents replacing the original mutable observation reference with that snapshot;
the policy hash586549a215110cef96fc053384a7b0db536f3799f4db6a124192f29ecfd94c14 is unchanged.

Local original-initialization probability study: 14/20
epochs complete, selected0; final diagnostic still queued
behind it. All54 parent operational files remain unchanged. Four focused tests
and a clean30-file package passed. Full CPU/shell/smoke checks are
running; inspect exec41211 and cpu-validation.json for the
actual completion rather than restarting them. Full Quadro benchmark chains are
preserved through maintenance ending25Nov13:00UTC/20:00Vietnam.

Evidence: output/shape-calibrated-training-20261123/training-start-milestone.json
and README.md. The all-four-metric3% full-benchmark goal remains active and
unachieved. No newly learned checkpoint has been admitted or benchmarked.

## Fresh calibrated-initialization Kaggle study (2026-11-23T19:10:33.134764+00:00)

Completed the fixed original-prior/calibrated-prior mask-complementarity audit:
intersection full/hidden IoU 0.8186210992/0.3738354673, union
0.8065072613/0.3843748158, versus calibrated 0.8132978430/0.3878743851.
Intersection keeps the original floors, but loses hidden IoU to the calibrated
parent (paired absolute interval -0.0222844 to -0.00598556). Its hidden gain over
the original is inconclusive. Union fails the full floor. Neither dominates
the calibrated parent. All 1,264 saved validation masks/217 images were checked;
no probability field, benchmark result or model promotion follows from this.
See output/shape-mask-complementarity-20261123/cpu/REPORT.md and verification.json.

A distinct schema9 experiment is launched on fresh Kaggle two-T4 session
373179451, remote wrapper150/child151. Driver launch19:07:56 UTC. Its observer
worker reports7732 (launcher12064). Current disposition: running
at 2026-11-23T19:10:18.067785+00:00; inspect observation.json for the actual execution phase.
No optimizer run is established by a setup-only observation. Inputs and all
uploaded assets were rehashed. Four focused tests and a clean30-file package
passed; full CPU/shell/smoke checks are running.

This is a new training protocol, not a repair or reinterpretation of the old
failed strict T4 experiments. It starts robust epoch7 with the fixed native-2
logit correction, freezes its encoder/BatchNorm, and trains its decoder with
proper BCE outside observed support for20epochs,seed20261126,global batch64,
LR1e-5. Same-device native pixels must repeat exactly. Cross-device initial
differences are explicitly recorded. Both selection floors are the maximum
of the recorded local calibrated parent and its measured T4 parent. Epoch0
is not a newly learned model. Separate Quadro admission remains mandatory;
existing admission schemas do not automatically accept schema9.

Original local probability training: 13/20epochs,
selected0; latest full/hidden IoU
0.8228143154/
0.3449616649. Its54 operational
files still match. Final diagnostic remains waiting_parent_processes.
All previous jobs and the released Kaggle session373165445 are preserved.
Latest Slurm read-only accounting exit0; see milestone receipt.

New study: output/shape-calibrated-training-20261123/README.md. One-time
allocation/upload/preparation/launch helpers must not be rerun blindly.
Credentials remain outside the repository. Training and independent export
replay are still required, followed by the full2,565-target four-metric audit.
The unchanged all-four3% goal remains ACTIVE and unachieved.

## Verification complete; local shape-only training remains active (2026-11-23 18:49 UTC)

The interactive recovery change passes481 Python tests with24 optional skips,
six Bash regressions and CPU smoke. Ten recovery tests also pass from a clean
temporary source tree containing no generated output or connection helper.
The first PowerShell wrapper returned1 after classifying normal unittest stderr
as NativeCommandError, although its complete log reports all checks passed.
Direct native recapture confirms all six Bash scripts and smoke exit0; this
transport issue and exact logs are recorded in
[cpu-validation.json](output/shape-kaggle-interactive-recovery-20261123/cpu-validation.json).

Local training has completed11/20 epochs, selected0; the next epoch
is active. Latest full/hidden IoU is0.8226324812/0.3476279268.
It remains below the original hidden-region floor, so no learned model is
eligible. All54 parent operational files, six archived original observer
sources and10 recovery observer sources verify. Diagnostic observer36128 is
still waiting for the parent run and its audit. Do not restart the trainers.

Kaggle session373165445 was released after both failed attempts and native
diagnostics were recovered. At18:42 UTC quota reports0h reserved,0.413676h used
of30h; probability batchv1 and constrained batchv2 remain QUEUED. The cancelled
interactive session must not be treated as active training. Both native replay
gates remain unchanged. No new model or benchmark admission exists.

Next decisive work is the existing local20-epoch completion/native audit and
fixed final-epoch offset diagnostic. Existing Quadro chains remain pending
through maintenance ending25Nov13:00UTC/20:00Vietnam. The2,565-target,
all-four-metric3% goal remains ACTIVE and unachieved. Do not call create_goal.

## Kaggle interactive attempts recovered; local training continues (2026-11-23 18:41 UTC)

Obtained a real two-T4 Kaggle interactive allocation, session373165445. The
original frozen notebook failed before optimization because LD_LIBRARY_PATH
hid Kaggle's NVIDIA driver libraries. A separate retry preserved those paths,
kept the installed packages and scientific suffix unchanged, and passed CUDA
and a real FFT round trip. Its strict native gate still failed: independent
local replay confirms one removed pixel on object000000006773-005 among79
objects in each of three diagnostic passes. Decoded RGB and input tensors match
the cluster exactly; CPU/GPU runtime-report differences are recorded explicitly.
Four cuDNN backend settings and CPU/double sigmoid/direct-logit probes did not
remove the discrepancy. No optimizer, training, new checkpoint or benchmark ran.

Implemented separate observer/export/replay tools. Both failed attempts were
fully recovered and all35 exported members rehashed. Kaggle accepted release
of only owned session373165445 at18:36:29 UTC after all owned processes exited.
Do not relaunch the completed attempts or reuse this released session. The
original six observer sources are archived against their frozen manifest;
the repository collector subsequently moved its transport import into main
so offline tests need no connection helper. Ten focused tests pass; the full
CPU/shell/smoke check is still running in exec44227. Record its actual completion.

Local probability training remains live at10/20 epochs, selected0.
Latest full/hidden IoU: 0.8227111319/0.3406713982;
original floors0.8113735902/0.3703334757. All54 operational files are unchanged.
Observer3532 / WSL trainer502 continue; final diagnostic observer36128 is
waiting_parent_processes. Preserve the complete20-epoch run and audit.
Fresh18:37 UTC accounting keeps45673-45675,45677-45679 and45680-45685 PENDING.
Maintenance ends25November13:00UTC/20:00Vietnam. No cluster job was changed.

Next: complete local native replay, then its already prepared fixed epoch20
offset diagnostic; evaluate all results without automatic promotion. An
eligible learned candidate still needs separate Quadro admission and the full
2,565-target audit. The all-four-metric3% goal remains ACTIVE and unachieved.
Evidence: [interactive recovery](output/shape-kaggle-interactive-20261123/README.md),
[independent pixels](output/shape-kaggle-interactive-recovery-20261123/native-comparison.json).

## Post-training diagnostic ready and waiting (23 November 2026, 18:06 UTC)

Finished the fixed epoch20 diagnostic and independent native-mask replay/report.
Eighteen pinned focused tests passed without skips; full checks passed 471 tests
(26 optional skips), six Bash regressions and CPU smoke. Isolated staging and
fresh hash checks preserve all 54 parent operational files and 22 training files.

Observer36128 is live and waiting for the existing parent training and its
independent audit to finish. Its four phases will run automatically afterward:
three-object GPU preflight, independent preflight replay, full diagnostic,
independent full replay/report. No diagnostic GPU work has started yet.

Parent observer3532 / WSL trainer502 remain live at5/20 completed epochs, with
epoch6 running and selected0. Full/hidden IoU remains0.8221570681/0.3403761707,
so no learned checkpoint is eligible. Kaggle remains queued; Quadro benchmark
jobs remain pending under maintenance. Full-benchmark3% on all four metrics
remains unachieved. [Evidence](output/shape-probability-final-diagnostic-20261123/README.md).

## Completed local result audited; probability training live (23 November 2026, 17:44 UTC)

All 20 local constrained-loss epochs and seven execution phases completed with
exit 0. Independent native replay and the report confirm epoch zero won; no
learned checkpoint is eligible. Selected masks match all 1,264 original Quadro
objects exactly. Training took 186.52 minutes. The completed collector exited
normally; preserve the result and do not restart the finished processes.
[Completed local report](output/shape-local-20261123/report/REPORT.md).

The probability-loss observer then repeated the predecessor audit, checked
Kaggle probability version 1 QUEUED and launched its frozen four-worker chain at
17:26:48 UTC. All 1,264 original masks matched exactly; independent calibration
replay and optimizer preflight passed. Training began 17:29:23 UTC and is live
under WSL PID 502 / Windows WSL PID 2980, with observer 3532. Two of 20 epochs
are complete, selected epoch zero; epoch 3 is running. Neither learned epoch
passes both geometry floors. Calibration processes 28772/424 are terminal.
[Actual launch](output/shape-probability-workers4-20261123/launch-milestone.json).
No new benchmark result or 3% gain is claimed; the goal remains active.

## Four-worker fallback verified and activated (23 November 2026, 17:10 UTC)

Passed 458 CPU tests (26 optional skips), six Bash regressions and CPU smoke;
all 19 focused pinned-environment tests passed without skips. Activated observer
PID 3532, then retired only unlaunched predecessor observer PID 21512 after
verifying both identities and the live original trainer. Replacement polling
now reports `waiting_predecessor`. All 54 new operational files, 22 staged files
and original two-worker operations pass hashes. The old observer exit is expected.

Original local training completed 18/20 epochs, selected epoch 0; epoch 19 is
running. Both Kaggle versions remain queued, as do relevant cluster chains.
The new package changes only data-loader workers from two to four. The 1.578x
input-throughput diagnostic and two exact optimizer updates do not establish
end-to-end speedup or model gains. No fallback GPU phase or benchmark admission
has started. The goal remains unachieved. See
[milestone](output/shape-probability-workers4-20261123/milestone.json) and
[protocol](output/shape-probability-workers4-20261123/README.md).

## Live shape-only work revalidated (23 November 2026, 16:47 UTC)

Local schema6 completed16/20 epochs; epoch17 is progressing, selected epoch0.
Epoch16 full/hidden IoU is0.792458570/0.370160236 versus original
0.811373590/0.370333476. No eligible learned checkpoint or full benchmark gain.
All eight existing observers and the actual WSL launcher/trainer are live.
The56 operational files and22 staged probability-source files pass exact hashes.
The schema8 fallback observer remains waiting_predecessor; no GPU phase started.

Fresh explicit Kaggle constrained version2 and probability version1 both report
QUEUED without a failure message. Quota is30h total,0.082019h used and0h reserved
by running sessions; exhausted GPU quota does not explain the queue. No replacement
submission or experiment change was made. Cluster45673-45685 relevant shape jobs
remain PENDING; maintenance ends25Nov13:00UTC/20:00Vietnam. Existing chain and
export collector remain intact. Goal stays ACTIVE and unachieved. Next decisive
work is the current local native replay, then the already prepared probability
fallback if Kaggle remains queued; only an independently eligible nonzero epoch
can justify new admission work. Evidence: [output/shape-monitoring-20261123/20261123T164710Z/receipt.json](output/shape-monitoring-20261123/20261123T164710Z/receipt.json).

## Local probability-loss fallback activated (2026-11-23T16:31:44.911939+00:00)

The shape-only >=3% goal remains ACTIVE and unachieved. Prepared a separate
schema8 local probability-loss experiment to use the GPU after the existing
schema6 run completes, only if fresh Kaggle probability version1 is still QUEUED.
Observer PID21512 /exec64500 is live; its initial poll confirms predecessor
launcher328 and trainer495 are active. Existing local training has completed
14/20 epochs, selected epoch0. Preserve its full protocol.
Both private Kaggle notebooks remain queued; calibrated jobs45680-45685 remain
pending. No existing run, source, collector, benchmark policy or job was changed.

Frozen22-file local source archive SHA256:
26183ee5be888feb4eb2841febcf2ac8ea64d5d6aac76e4fe0e7041ec880d61b.
Staged at /home/hungvudao/.cache/shape-local-gpu-20261123/probability-source;
all20 predecessor source files were preserved. The original trainer, native
verifier and report were copied with explicit schema/loss changes; originals
remain byte-identical. Operational54-file manifest SHA256:
c802b8a43266fa43fa19eabf263d5a8d8d5e935c0a7b34c9d2192bde83f787f5.

Per-object BCE outside input support replaces the Dice/hidden-positive loss.
Original initialization, isolated data, seed20261125,20epochs,LR1e-5,batch64/
microbatch8, frozen encoder/BatchNorm and GroupNorm decoder are retained.
A fresh independent1264-mask exact Quadro pixel replay must pass before any
optimization; two actual optimizer preflight steps and selected pixel replay
remain mandatory. Both full/hidden geometry floors and epoch-zero exclusion
remain unchanged. No identical learned trajectory across GPU platforms is claimed.

All18 focused pinned-environment tests passed without skips. Full CPU checks
passed451 tests with25 optional-dependency skips,six Bash regressions and smoke.
The observer waits for real predecessor process termination, successful complete
execution and independent native replay, then checks Kaggle again. RUNNING or
COMPLETE skips this fallback; failure/cancellation needs review. Observation
failure never launches work. A recorded request prevents duplicate automatic
launch. On admission it runs calibration, preflight, training, native validation,
independent replay and reporting without further interaction.

No schema8 GPU work or new benchmark admission has occurred. Existing schema4
admission does not accept schema8. Inspect an independently verified nonzero
selected epoch before implementing its separate Quadro admission. The calibrated
full pipeline remains ready for maintenance ending25Nov13:00UTC/20:00Vietnam.
Evidence: output/shape-probability-local-20261123/README.md, milestone.json,
cpu-validation.json, staging.json and queue-status.json.

## Calibrated postprocessing staged and attached (2026-11-23T16:14:34.181407+00:00)

The shape-only >=3% goal remains ACTIVE and unachieved. No calibrated admitted
package or new benchmark target exists. Jobs45680->45681->45682->45683->45684
remain pending; maintenance ends25 November13:00UTC /20:00Vietnam.

Finished the calibrated metadata export, independent native/selection/statistical
replay, terminal allocation recovery and four-metric report. The complete WSL
check passed:445 Python tests,24 optional-dependency skips,six Bash regressions
and CPU smoke. All19 focused cluster tests passed without skips. The separate
400-file postprocessing manifest is
b2cae653449e4228bc2505962502f614341f4c4b4735de9ad99ee77753f0320b.
The395-file inference snapshot and all earlier299 v2,6 constrained training,
2 constrained native and392 calibrated validation files remain byte-identical.

Submitted CPU export follower45685 once at16:11:22UTC, afterany45684. Direct
scontrol confirms its dependency and separate operational directory. It follows
infrastructure continuation decisions and exports only a completed full audit.
The updated collector automatically recovers the export, runs the immutable
metadata verifier and builds the report with failure penalties, CLIP coverage,
historical unseen/native geometry and explicitly scoped Slurm accounting.
Replaced only old collector PID34848; replacement PID22844 /exec94047 is live,
and its initial successful poll includes all six calibrated jobs. Other observers
and training remain intact. Do not restart any process without rechecking identity.

Local constrained training has completed12 of20 epochs;
selected epoch remains0. Preserve the complete protocol.
Both private Kaggle kernels remain QUEUED (constrained v2, probability-loss v1).
No new eligible learned checkpoint or >=3% benchmark gain is claimed.

Next: inspect completed Quadro replay and downstream GPU gates; retain exact
pixel guards on failure. Recover all2565 targets and finish the independent audit
before evaluating the four >=3% thresholds. The export/report path is now staged;
the earlier unstaged-postprocessing note is superseded.
Evidence: output/shape-calibrated-benchmark-20261123/postprocessing-milestone.json,
postprocessing-validation.json and finalization-jobs.json.

## Calibrated shape-only benchmark chain submitted (2026-11-23T15:55:52.875511+00:00)

Goal remains ACTIVE and unachieved. Complete corrected-prior local native replay
passed: 1,264 objects / 217 images / zero differing pixels. Quadro native job45680
is maintenance-pending. Candidate SHA256 remains
4e65f4a3525db3b1e6f591c5e5bee0061f817d8d39ae1b4d15ba7929f5dc96cf.
No admitted calibrated package or new benchmark output exists yet.

Implemented strict native admission, shape-only evaluator integration, independent
selected-mask geometry replay, frozen reference/policy preparation, reserved native
geometry evaluation, scored four-worker smoke, all-target full/unseen/native audit,
and infrastructure-only continuation preserving every closed-target artifact.
Source is staged separately at output/shape-calibrated-benchmark-20261123/evaluation-source
on the cluster: 395 files, manifest SHA256
f3bca610d9b9018ba2aa5515e39f811c0df57e90e396c7f0eba4be55a16ae600.
All prior v2/constrained/native source manifests remain unchanged.

Windows integration timeout did not recur in WSL with the existing 30-second
subprocess limit: all five tests passed in20.237s on /mnt/c and4.356s on ext4.
This supports an environment-related timeout but does not identify its exact
Windows cause. The first two ext4 full suites failed because the export omitted
datasets/open_world_amodal/official metadata used by an existing regression.
After including that metadata, the complete check passed:440 Python tests with24
optional-dependency skips, six Bash regressions and CPU smoke. Nine calibrated
tests also passed in local pinned Torch without skips. All25 focused tests passed
in the staged cluster environment without skips; Bash syntax passed.

Actual chain submitted at15:54:24 UTC:45680(native validation) ->45681(admission,
CUDA extension/import checks and reserved geometry,1Quadro) ->45682(scored smoke,
4Quadro/4workers) ->45683(full2565,4Quadro/4workers) ->45684(continuation controller).
Full job includes verification, registration, four-metric gate,2017 historical
unseen targets,321-target native geometry and failure transitions. A failed native,
preflight or smoke prerequisite prevents full inference. No application-failure
or completed-bad-result retry. Initial preflight/smoke each request2h; full24h.
All confirmed queued. Maintenance ends25Nov13:00UTC /20:00Vietnam.

Collector output/shape-calibrated-benchmark-20261123/collect.py --watch is live in
exec13362. It only observes and recovers checksummed evidence; never resubmits.
Check collection-control/status.json and its process before restarting. Existing
v2 and constrained collectors/chains remain live. Both private Kaggle notebooks
remain QUEUED. Local training has completed10 epochs, selected epoch
0; preserve its complete20-epoch protocol and existing observer.

Next: inspect recovered Quadro native validation and the dependent GPU chain;
retain exact pixel guard on failure. Recover full outputs/provenance and complete
independent metadata/statistical replay, final report and GPU cost accounting when
available. Calibrated metadata export/report helpers are not yet staged. Never
claim >=3% from validation or partial benchmark. All four complete-benchmark
metrics, failure penalties and CLIP coverage remain mandatory.

## Shape probability diagnostic verified; separate candidate under runtime validation (2026-11-23T15:32:03.070031+00:00)

Independent replay passed 7,584 native masks / 1,264 objects / 217 images. Only fixed
offset -2 retains both geometry floors (full 0.8132978430, hidden 0.3878743851 versus
original 0.8113735902 / 0.3703334757). A separate validation-selected candidate is now
frozen in output/shape-calibrated-20261123/candidate. No benchmark admission or
benchmark gain is claimed. Cross-GPU uncorrected adapted replay differs by 4 pixels
on 3 objects; original v2 matches exactly. Brier/NLL probabilities were not exported.

New external schema 5/6/7 validation bridge and corrected-prior runtime passed 9
focused tests in pinned Torch locally and on the cluster, no skips. 392 files staged
in a new cluster subtree; 299 frozen v2 + 6 training + 2 native sources unchanged.
Corrected candidate three-object GPU preflight passed exact pixels. Full 1,264-object
local replay is running (exec 62233 / Windows WSL PID 37048); conditional watcher exec 30805 (Python PID 21432)
will submit exactly one Quadro native validation only after complete local success.
See local-execution.json,quadro-queue-control/status.json,then quadro-job.json.
Do not duplicate either process. No Quadro job or full benchmark admission yet.

Both Kaggle notebooks remain queued; local training completed 7 epochs,
selected epoch 0. Existing observers and chains 45673/45674/45675 and
45677/45678/45679 are preserved. Maintenance ends 25 Nov 13:00 UTC / 20:00 Vietnam.
Full pipeline GPU checks, adapter admission, smoke,full 2,565scoring/registration,
unseen/native audit and >=3% on all four metrics remain pending. Goal stays ACTIVE.
Detailed evidence and next steps: output/shape-calibrated-20261123/README.md.
The complete offset comparison was rendered after HTML preview; PNG/SVG retained.

## Separate proper-loss Kaggle experiment submitted (2026-11-23T15:11:58.997198+00:00)

The previous turn made progress: completed native error attribution and launched
the fixed-offset diagnostic. This continuation verified all live handles and
submitted a distinct shape-only training hypothesis motivated by excess mask
pixels outside observable support. No existing run, frozen source or selection
rule was changed. No new model has been admitted to benchmark inference.

Private kernel146908842, version1:
https://www.kaggle.com/code/ez4kacchan/shape-only-probability-training-20261123
Submitted15:07:11 UTC, acknowledged15:07:27 UTC; current state QUEUED
at 2026-11-23T15:11:04.175221+00:00. Source pulled back byte-identically and private/two-T4
metadata verified. Kernel SHA256 f3621020ba9120c6476683445d3af14c0594796a62fbead8c0b459b3aa8fab16.
Four-hour wall limit. Reuses the unchanged private training inputs and138 exact
conda packages; no new data upload or credentials were needed.

Separate schema7 uses per-object binary cross entropy outside input support,
with true amodal targets, replacing Dice and the hidden-positive bonus. Original
v2 initialization, seed20261125,20epochs,LR1e-5,2T4/global batch64, support mix,
augmentation and full-IoU retention/hidden-IoU selection remain unchanged. Epoch
zero cannot be promoted. Decoder uses GroupNorm; encoder and BatchNorm remain
frozen and are hashed every epoch. Exact original GPU diagnostic remains required
before optimization; two real optimizer preflight steps precede full training.
Selected native masks must also match training pixel hashes, including cases where
different masks have identical IoU. Local recovery independently replays those
pixels, native scores and selection. No benchmark or reserved holdout access.

Five focused tests passed without skips in pinned Torch; offline validation ran
the actual source overlay, verified7 added files and16 unchanged inherited files,
and confirmed existing-source overwrite rejection. The unsubmitted first prepared
package is preserved; final uploaded source and receipts are authoritative. Real
Kaggle GPU execution and any learned benefit remain unproven while queued.

Watcher36192/exec8359 and collector11972 are live.
Logs/evidence: output/shape-proper-loss-20261123/collection-watch.log,
submission.json,upload-verification.json,offline-package-validation.json,
probability-source-manifest.json and milestone.json. Terminal recovery targets
explicit version1 and preserves failures; successful replay triggers the report
builder. Existing schema4 admission does not accept schema7.

Original constrained Kaggle version2 remains QUEUED at2026-11-23T15:10:56.278893+00:00.
Local training has completed5/20 epochs; selected epoch remains
0. The independent fixed-offset diagnostic remains live
at850/1264 objects. Existing training/observers and
cluster chains remain intact. Full2,565-target inference, scoring, registration,
unseen/native audits and the all-four-metric3% goal remain incomplete and ACTIVE.

## Shape error diagnosis complete; fixed-offset diagnostic running (2026-11-23T14:54:22.249333+00:00)

Independent native error decomposition completed on all 1,264 validation objects /
217 images. Restored-support robust epoch7 loses 1.557 full-IoU percentage points:
false positives contribute -3.410 points, recovered visible pixels +0.840, and
recovered hidden pixels +1.014. False-positive changes are entirely outside the
unchanged SAM input. This is an arithmetic attribution, not a causal experiment.
Evidence: output/shape-error-modes-20261123/REPORT.md and result.json. The earlier
unclamped regression remains distinct; no checkpoint or policy was reselected.

New fixed-checkpoint diagnostic tests logit offsets 0/-0.5/-1/-1.5/-2 on epoch7,
with the original v2 checkpoint as reference. Geometry and Brier/NLL are measured
on validation only; no benchmark or holdout access. CPU three-object preflight
passed with exact native masks but was slow. Capped RTX3060 preflight passed with
zero differing pixels against both original native references and peak allocation
249.995 MiB. The first GPU preflight lacked the WSL CUDA-library path and exited6;
the repaired launcher uses the same explicit library environment as training.
The first CPU attempt used a minimal bundle lacking reference predictions; all
failed evidence remains preserved. Four focused tests passed in pinned Torch.

Full diagnostic exec48303 is running with one object per pass, two CPU threads,
20% GPU-memory cap, FP32 and TF32 disabled. Source archive gpu-source.zip and
execution receipts live under output/shape-probability-20261123. After completion,
the launcher independently replays all 7,584 exported masks and emits REPORT.md.
Probability aggregate arithmetic is replayed from generated rows; probabilities
are not exported. No offset is automatically admitted to benchmark inference.

Local training has completed 4/20 epochs; latest full/hidden IoU is
0.7894979492/0.3686593635.
Training/launcher remain live; no frozen training code or rule changed. Kaggle
version2 remains QUEUED at 2026-11-23T14:54:17.428627+00:00; all four existing observers
remain active. SSH confirmed chains45673->45674->45675 and45677->45678->45679
pending, with both GPU roots explicitly blocked by the maintenance reservation.
No job was resubmitted, cancelled or retuned. Goal remains ACTIVE and unachieved.

## Local shape-only training monitored; checks verified (2026-11-23T14:34:02.070158+00:00)

Local training is active in exec55182. 2/20 epochs
completed; selected epoch remains0. Last completed
epoch full/hidden IoU:0.7942537569/0.3669173635.
Epoch1 processed all9494 objects with149 optimizer steps and zero skipped steps,
but failed both initial geometry floors; it was not selected. Preserve the full
20-epoch protocol and baseline. Training and native recovery remain incomplete.

Validation completed:412 Python tests/nine optional skips, all six Bash regression
files and CPU smoke passed in their logs. Eight focused tests passed without skips
in pinned Torch (including loss/gradient equivalence), plus two observer regression
tests. The outer PowerShell wrapper reported exit1 because successful unittest
stderr became NativeCommandError. This was reproduced with a zero-exit stderr-only
program; direct subprocess execution returned native exit0, and an independent
native subprocess smoke also returned0. Do not describe the original wrapper as
exit0. Evidence:output/shape-local-20261123/validation.json and check-local.log.

Observer2332/exec69302 remains active; it will audit native results and build the
local report after completion. Kaggle version2 isQUEUED at2026-11-23T14:34:01.835142+00:00;
watcher22364 remains active. Last quota query showed29.92/30GPU-hours remaining.
No job was duplicated, cancelled or retuned. The separate local initial-mask
audit remains exact across1264 objects with zero differing pixels against Quadro.
No schema6 checkpoint has been admitted for benchmark inference. The all-four
full-benchmark >=3% goal remains ACTIVE and unachieved. Read training-milestone.json
and README.md under output/shape-local-20261123 for paths and next steps.

## Shape-only local training active; exact initial mask replay passed (2026-11-23 14:23 UTC)

Kaggle setup is complete, but version2 remained QUEUED at14:22:20 UTC. Existing
watcher22364/exec25827 is active. No Kaggle training result exists yet. All six
Slurm jobs45673/45674/45675 and45677/45678/45679 remain pending under maintenance;
confirmed by squeue during this continuation. No queued job was changed.

To make progress during the queue, an isolated local RTX3060 Laptop6GB fallback
was prepared and **training started14:16:26 UTC** after source validation,
full calibration and a real two-step optimizer preflight. This local use is an
assistant-selected fallback distinct from the explicitly requested cluster and
Kaggle options. Exact138 cluster conda builds, NumPy1.23.5/Pillow9.5.0 and CUDA
FFT passed. With TF32 disabled, the full1264-object/217-image initial checkpoint
replayed exactly on the local GPU. Independent native-mask replay also found
**zero differing pixels against the original Quadro diagnostic**. Initial
full/hidden IoU0.811373590207884/0.3703334756537925 is unchanged. The separate
79-object TF32-enabled diagnostic differed on29 objects; preserve those archives.

Schema6 LocalWSL is a separate frozen protocol: decoder-only GroupNorm training,
frozen encoder/BatchNorm, AdamW LR1e-5, seed20261125,20epochs, effective batch64 in
microbatches8, two data workers/CPU threads,75%GPU-memory cap. Shape-only loss and
fixed SAM/annotated augmentation; no benchmark/holdout assets. Both original
Quadro and local full/hidden IoU floors must hold before hidden-IoU selection.
Epoch zero remains ineligible for promotion. Loss/gradient equivalence tests
passed in pinned Torch. Preflight made two real updates without skipped steps,
preserved the encoder and exactly replayed native pixels after checkpoint reload.

Frozen20-file source: output/shape-local-20261123/training-source-v2.zip,
SHA256dd445099f7171e6ed853bffaf35ab8bd5adffe8ba1a4ad6d337961aa7100e2be.
Earlier unexecuted source archive is preserved. WSL source/runtime/inputs are
under /home/hungvudao/.cache/shape-local-gpu-20261123. Launch exec55182/WSL PID328,
trainer PID495 was verified live. Read execution-v2.json, train-v2.log and run-v2/.
At this observation epoch1 is ongoing; no completed epoch or learned gain claimed.

Local collector2332/exec69302 watches collection-watch-2.log and will independently
rehash/replay completed native masks, selection and statistics, then build its
report. Its first launch failed because the lock directory was absent; fixed
with two passing regression tests. The training process was never restarted.
scripts/verify_shape_local.py independently verified calibration; three focused
audit tests passed. Full CPU/Bash/smoke checks are currently running in exec81719
with log check-local.log. Do not declare that full check passed until it exits.

No local or Kaggle checkpoint is admitted by the existing schema4 Slurm adapter.
GPU adapter equivalence, smoke and full2565-target scoring/registration/native
audits remain necessary. The all-four full-benchmark >=3% goal remains ACTIVE
and unachieved. Read output/shape-local-20261123/README.md and calibration-replay.json.
Check live process handles before restarting any observer or trainer.

## Exact Kaggle runtime verified; GPU retry queued (2026-11-23 14:02 UTC)

The CPU runtime preflight COMPLETED successfully at the13:55:46 UTC observation.
Its observer PID28100/exec53951 finished normally after recovery; do not restart.
All138 exact cluster conda builds passed archive SHA256 checks. The recovered
79-object inputs.json is byte-identical to the cluster CPU receipt, including
Torch1.13.1/cuDNN8500 and its build configuration. See
output/shape-kaggle-20261123/runtime-preflight-replay.json.

Training version2 remains QUEUED at14:02:02 UTC. Its recovery watcher PID22364 /
exec25827 remains active. No optimization or GPU native replay has occurred in
version2 yet. The initial cross-GPU one-pixel mismatch from pip version1 remains
unresolved until the exact conda build executes on T4. No replay tolerance or
selection rule changed. Read collection-watch-2.log for newer observations.

All six cluster continuation/training jobs are still pending under maintenance,
confirmed14:02:16 UTC; no jobs were changed. The four-metric3% full-benchmark goal
remains ACTIVE and unachieved. Kaggle setup and private two-T4 execution access
are established; the current training retry and later benchmark work are pending.

## Kaggle recovery ready; runtime check running (2026-11-23 13:54 UTC)

Private training kernel146891234 version2 is QUEUED:
https://www.kaggle.com/code/ez4kacchan/shape-only-constrained-training-20261123
Its server source matches the submitted hash and private status. The separate
CPU runtime preflight is RUNNING at
https://www.kaggle.com/code/ez4kacchan/shape-only-runtime-preflight-20261123
No new optimization, selected checkpoint, GPU replay or benchmark gain exists.

Training watcher PID22364/exec25827 and CPU-runtime watcher PID28100/exec53951
are live. Logs: output/shape-kaggle-20261123/collection-watch-2.log and
runtime-watch.log. They recover explicit versions and preserve terminal failures;
the training watcher independently replays completed native evidence. Do not
start duplicate observers. The runtime check validates all138 conda archive
hashes, package identities and79 preprocessing records; it cannot prove GPU replay.

Recovery checks passed:406 Python tests/eight optional skips, six Bash regressions
and CPU smoke, exit0. The downloader also recovered13 actual diagnostic artifacts
byte-identically to the standard CLI. See recovery-validation.json and
downloader-validation.json. scripts/build_shape_kaggle_report.py is prepared and
syntax-checked, but no successful result exists to report yet. This report source
was added after the full check; no runtime/training source changed afterward.

Kaggle reports29.92 of30 GPU quota hours remaining. The CLI confirms retired P100
requests redirect to T4, so they do not bypass this queue. CPU runtime submission
was13:47:47 UTC; GPU version2 submission was13:40:20 UTC. Exact initial/selected
replay and the full-IoU retention gate remain unchanged. Cluster maintenance
continues; all six original queued jobs remain pending, without modification.
Goal remains ACTIVE and unachieved. Current evidence: continuation-status.json.

## Private Kaggle execution established (2026-11-23 13:42 UTC)

Kaggle setup is complete with private dataset and two-T4 notebooks; credentials
stay local. Canonical training URL:
https://www.kaggle.com/code/ez4kacchan/shape-only-constrained-training-20261123
See output/shape-kaggle-20261123/README.md and its versioned receipts.

Version1 failed before optimization at exact initial native replay. Five focused
tests and two-T4 FFT passed. The completed 79-object diagnostic confirms identical
raw images, decoded RGB and tensors across cluster/Kaggle, and exact repeated T4
predictions. One pixel in one object differs from the saved Quadro mask. This
measured count is specific to Kaggle; job45676's pixel count remains unmeasured.
Version2, submitted13:40:20 UTC, is QUEUED and tests the exact138 conda package
builds before the unchanged strict training replay. No tolerance was added.
No successful Kaggle training result or admitted checkpoint exists yet.

Recovery is implemented: scripts/collect_shape_kaggle.py --watch runs in exec25827,
logging collection-watch-2.log; check before restart. Three recovery tests passed.
Version1 artifacts retain an inaccurate descriptive reference to decoder BatchNorm;
the actual decoder uses GroupNorm, while encoder BatchNorm is frozen.

Cluster chains45673/45674/45675 and45677/45678/45679 remain pending, collectors14576
and2860 live. No scheduler or frozen-source changes. The all-four full-benchmark
3% goal remains ACTIVE and unachieved. Kaggle schema5 is not admitted by the Slurm
schema4 adapter; later GPU admission, smoke and full benchmark remain necessary.


## Shape-only adapter ready; Kaggle setup requested (2026-11-23 13:21 UTC)

Implemented and staged the isolated constrained-checkpoint adapter. Admission
requires completed schema4 training, eligible epoch>0, exact native GPU replay
and independent local replay. Preserves v2 seeds, original-observation geometry,
expected-IoU selection/fallbacks and RGB. CPU checks passed:401 tests/seven skips,
six Bash regressions and smoke; nine focused tests passed in pinned cluster Python.
See output/shape-constrained-adapter-20261123/README.md, validation.json and staging.json.
344 source files staged separately; remote frozen299 v2, six training and two
native sources unchanged. No admission package, GPU validation or new benchmark
job exists yet. The prepared GPU adapter test compares1264 native masks pixelwise.

Maintenance remains ACTIVE until25 Nov13:00 UTC/20:00 Vietnam. Existing45673/45674/45675
and45677/45678/45679 chains remain pending. Collectors14576 and2860 are live; v2
retains222 records/112 successes. No job was changed. The user now explicitly
requested Kaggle setup; proceeding with isolated CLI environment and private
shape-only training preparation. The full four-metric goal is unachieved.

## Verified checks and queued work (2026-11-23 13:05 UTC)

The repaired local check completed with exit 0: 392 Python tests (7 optional skips),
all six Bash regressions and CPU smoke passed. Three focused tests also passed
in pinned cluster Python with Torch installed. See
output/shape-sam-constrained-v2-20261123/validation.json and training-tests.log.
No repaired GPU preflight or training result exists yet: 45677/45678/45679 remain
queued behind maintenance. V2 continuation 45673 and helpers 45674/45675 remain
queued; collectors PID 14576 and PID 2860 are alive. Full scoring and the four 3%
gates remain pending.

Precision note: diagnostic 45676 establishes changed thresholded geometry for
one of 79 objects under deterministic cuDNN, and zero changed objects under the
original flags. It did not save pixelwise comparisons. Earlier wording of
"one pixel" was too specific; the number of differing pixels is unmeasured.
The exact replay guard remains intact, and the repaired native validation
uses the original arithmetic while restoring the training setting afterward.

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

## Shape-only continuation and isolated SAM development (23 November 2026, 12:26 UTC)

Frozen benchmark 45636 timed out after 1h30m25s and automatically resumed as
45658 on four Quadros. The backfill helper successfully granted 48 minutes,
starting 12:08:05 UTC. Continuation controller 45659 and export follower 45660
are pending. Independent replay verified all 166 retained records and 299 frozen
source files unchanged; see `output/shape-v2-slurm-20261123/first-continuation-replay.json`.
The 12:24 collector has 176 terminal records (85 success, 47 ambiguous_target,
31 no_completion, 13 not_detected). No complete scores or 3% gains exist.
The 48-hour maintenance reservation still begins at 13:00 UTC / 20:00 Vietnam.
A temporary local DNS outage recovered; the existing collector remains alive.

Validation diagnostic 45645 completed in 4m40s on one Quadro. Local native replay
of all 1,264 objects / 217 images verified the SAM-input drop exactly: full IoU
0.888731 -> 0.811374, hidden IoU 0.436647 -> 0.370333. Hidden scoring uses the same
177 images / 680 objects in both conditions. No empty SAM masks. This is a
validation proxy using visible-support box prompts, not benchmark improvement
or GroundingDINO selection evidence. Read
`output/shape-validation-sam-20261123/REPORT.md` and `native-replay.json`.

A separate, validation-driven shape-only robustness experiment is now prepared
under `output/shape-sam-robustness-20261123/`. Train-only SAM mask job 45661
COMPLETED in 5m58s, producing 9,494 masks / 1,554 images with zero empty masks.
GPU preflight 45662 is running; 20-epoch fine-tuning 45663 waits for it. The
fixed configuration mixes SAM/annotated inputs, permits the prior to correct
input-mask errors, preserves true hidden supervision and uses native validation
for checkpoint selection. No Surface, RGB correction, combined work, benchmark
selection or new holdout access belongs to this experiment. Preserve both
immutable source snapshots. Five data tests passed in pinned cluster Python;
real preflight/training outcomes are still pending. The active all-four 3%
full-benchmark goal remains unachieved.

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

## Shape-only training monitored and recovery ready (23 November 2026, 10:07 UTC)

Authoritative recovery at 2026-11-23T10:07:01.880066+00:00: job 45632 is RUNNING on four
Quadro RTX 6000 GPUs; 44/60 epochs complete, receipt complete=false. Jobs
45634-45637 remain queued. No v2 native geometry, policy freeze, trained-model
smoke, or benchmark result exists yet. The >=3% four-metric goal remains ACTIVE.

To use more time before maintenance, pending benchmark job 45636 was updated:
`scontrol update JobId=45636 TimeLimit=02:30:00 TimeMin=01:30:00`.
Slurm confirmed both values and the Nov 23 13:00 UTC-Nov 25 13:00 UTC maintenance
reservation. This supersedes the original fixed 90-minute allocation; its minimum
remains 90 minutes for backfill. See
[allocation amendment](output/shape-v2-slurm-20261123/allocation-amendment.json).
No training/evaluation source snapshot, model, seed, reference, or policy changed.

`python scripts/recover_shape_v2.py` saves dated, checksum-verified evidence and
allocation costs. `--checkpoints` recovers best.pt/last.pt only after successful
completed training, preserving existing local files. Latest snapshot is pointed
to by `output/shape-v2-slurm-20261123/latest-recovery.json`. A local observer,
`output/shape-v2-slurm-20261123/watch_training.py`, is live in exec session 28679,
logging to `training-recovery-watch.log`; it polls job 45632 every 45 seconds and
recovers final checkpoints automatically. Check its live process/log before
starting another observer. It never resubmits jobs or treats polling failure as
job termination.

The metadata exporter now recognizes shape-v2 and binds decisions to the closed
target's seed, trace hash and chosen slot. This is local postprocessing code;
it has NOT been installed into the isolated evaluation snapshot. Use the updated
exporter in a separate postprocessing source location after benchmark completion.
New validation: 356 Python tests (11 optional torch/sklearn/scipy skips), six Bash
regressions and CPU smoke passed; ten Windows focused recovery/export tests passed.
See `recovery-export-validation.json` and `check-recovery-export.log` under the study.
The prior cluster GPU evidence and failed runtime-job distinction remain unchanged.

Next: inspect 45632/45634-45637, verify final recovered training receipt and model
contract, inspect native validation/freeze/holdout and real smoke, then complete
all 2,565 targets and full verification/registration/audit/export/report. Keep
all four metrics, CLIP coverage, historical unseen reporting, native geometry and
cost in the final assessment. Do not infer improvement from training progress.

## Shape-only Colab handoff validated (23 November 2026)

Prepared a separate v2 ResNet34 amodal predictor, observation-only data loader,
resumable Colab trainer, notebook and hash-verified code/data ZIPs. Package holds
9,494 training and 1,264 validation objects; 1,183 holdout objects remain local.
All benchmark COCO IDs and five content near duplicates were excluded. Final
artifact and split receipts are in [the package](output/shape-v2-20261123/README.md).
333 repository tests, Bash regressions and CPU smoke pass; nine additional
focused CPU checks pass. All 12,529 archived data members verified.
Colab login works but this account has zero compute units and H100 was rejected.
The user said to prepare the package first while the H100 account is pending.
No new training or benchmark result exists. The all-four-metric 3% objective is
still active; old runtime integration and GPU validation remain future work.

## Canva Vietnamese refinement saved (16 November 2026)

User approved the reviewed 20-slide draft. Committed transaction `799911289478251733` to existing design `DAHbEx8zTOs`; verified saved title/page count. Edit: https://www.canva.com/d/45_3JDG-_oC01WZ . This supersedes the previous pending Canva transaction. Refined Vietnamese wording, hierarchy, labels, numeric conventions, tables and chart; research claims and Shape-not-implemented status preserved. All 20 Canva slides visually reviewed; no text outside canvas. Editable local PPTX and native PDF: `output/week8-vietnamese-refinement-20261116/`. PowerPoint checks 366 text shapes with zero overflow. See its README and receipts.

## Week 8 presentation delivered (16 November 2026)

Completed the requested 20-slide research story in
[output/week8-research-story-20261116](output/week8-research-story-20261116/README.md),
retaining the previous blue/white theme and adding diagrams, paper examples,
quantitative/severity/qualitative comparisons, and timed Vietnamese speaker notes.
Editable PPTX, PDF, offline slideshow, rehearsal/Q&A and delivery ZIP are ready.
Shape-Aware remains to be defined per user confirmation; combined results remain
unmeasured. All 20 slides passed native PowerPoint rendering and text-bound checks;
headline evidence, source links, viewer behavior and package integrity were checked.
No scientific pipeline or experiment state changed; the original deck is preserved.

## Annotated full continuation delivered (14 November 2026)

Recovered and audited the completed 2,565-target annotated control/Surface-PMC
comparison. Full LPIPS +1.191% [0.749%, 1.641%], paired CLIP +1.323%, VGG +2.360%,
SSIM +1.384%; frozen unseen LPIPS +0.801%. Primary 204-target cohort remains
inconclusive. Both full runs are registered. Coverage 1,394 versus 1,399 outputs;
allocation cost 4.733x. No new training, fitting, inference or job submission.

39490 timed out; planned 39491 completed, as did 39488/39492/39497/39495. Collector
finished all exports. All 16,197 metadata hashes, 2,847 selector choices, matched
provenance, preserved cohort, registry and full/unseen statistics pass locally.
Eleven focused tests and both science/reference freeze checks pass. Four-page
PDF/report/figure and all-target CSV are at
output/surface-annotated-full-review-20261114; see its audit and delivery receipts.
Native full images remain on cluster. No unfinished study job needs resubmission.

## Annotated Surface-PMC delivery completed (12 November, 10:57 UTC)

The frozen 204-target / 192-image cohort, final-pool diagnostic, independent
cluster and local audits, report, gallery and native recovery are complete.
Primary LPIPS gain: +0.250% [-0.269%, +0.810%], inconclusive. Secondary CLIP:
+1.333% [+0.258%, +2.496%]. Both arms save 87 outputs and retain 117 failures.
No new training or retuning. Annotated vocabulary is the baseline in both arms.

Start at [the delivered research package](output/surface-annotated-20261112-freshrefs/START_HERE.md).
The report has ten pages; all 204 targets and 582 thumbnail panels are included.
Local/cluster cohort and pool audits agree exactly except audit timestamps.
Browser review caught malformed dropdown closing tags; the builder and validator
were repaired, then 25 browser checks and all 582 image loads passed. Displayed
48 gains / 39 losses / 117 ties agree independently with paired records. All PDF
pages were inspected on the rendered contact sheet, with native-size checks of
the qualitative examples and equation pages. Review receipts preserve hashes.
Science and fresh-reference amendment freeze checks still pass.

Full control 39488 remains running (572/2,565 records at 2026-11-12T10:57:54.321683+00:00);
no worker tracebacks. Surface 39490, timeout continuation 39491, finalizer 39492,
full/unseen-image audit 39497 and export 39495 remain dependency-pending.
Collector 20156 is active. These are valid longer inference jobs; preserve them.
Full results/registration are unfinished and must not be inferred from the cohort.
Read docs/SURFACE_ANNOTATED_CONTINUATION_20261112.md before resuming. The original
five-hour work window is 06:46-11:46 UTC; it has not been restarted.

## Annotated cohort result and recovery (12 November, 10:20 UTC)

All 204 targets are scored in both arms. LPIPS +0.250% [-0.269%, +0.810%] is
inconclusive; secondary CLIP +1.333%. Both arms save 87 outputs and fail on the
same 117 pre-sampling targets. The 1,152-candidate Surface-PMC allocation used
1h49m34s versus the control's 24m. The pre-specified final-pool diagnostic and
both independent cluster audits passed. All 5,577 native archive members have
been recovered with independent checksums. Local replay and delivery validation
are running. The full control continues normally; dependent full Surface-PMC,
verification, unseen analysis and metadata recovery remain submitted.

## Annotated recovery preparation (12 November, 09:15 UTC)

The cohort is at 164/204 targets; quality analysis awaits completion. Independent
native CPU replay jobs 39498/39499 are submitted using the existing pinned
environment with user-site imports disabled. Four-stream archive recovery and
safe member extraction pass eight focused integrity tests. Native recovery,
local replay, scientific report and full gallery remain pending. Deadline 11:46 UTC.

## Annotated methods and report preparation (12 November, 08:24 UTC)

The transferred 22-feature ranker and exact correction/metric/statistical
operations now have a mathematical appendix. The prepared report includes a
paired method diagram and full-precision parameters. Live job 39487 has 46/204
records and 27 pools, no worker errors; quality analysis awaits the full cohort.
Science source and reference-amendment freezes remain unchanged and pass.

## 08:01 UTC execution update

Fresh annotated control 39486 completed successfully in 24 minutes: all 204
cohort records scored, verified with provenance and snapshotted. All 817 reference
files were copied with matching hashes. The historical/current reference audit
found exactly one changed mask pixel across the entire cohort, affecting one
reference and no crop geometries; no quality outcomes entered that audit.

GPU smoke 39496 is running; cohort 39487 waits for it. Resource constraints led
to an explicit scheduling amendment: full control 39488 waits for cohort 39487
and pool diagnostic 39489, prioritizing the measured cohort result by 11:46 UTC.
Full surface 39490, timeout continuation 39491 and finalizer 39492 remain valid.
CPU independent full/unseen audit 39497 follows finalizer; metadata export 39495
now follows that audit. Cohort export 39493 and pool export 39494 are queued.
Superseded pending original exports 39480-39482 were cancelled and recorded.

Local hidden collector process 20156 automatically retrieves completed cohort
metadata/native, pool metadata and full metadata exports; it never reads
unfinished quality outcomes. Logs: logs/annotated-collector-20261112.log/.err;
state: output/surface-annotated-20261112-recovery/collector-status.json.
New audit/report/gallery/PDF/ZIP builders are prepared but not yet run on results.
Three independent pool-audit regressions pass; the original science source and
method freezes still pass. WSL surface-replay-py310 matches runtime NumPy 1.23.5,
Pillow 9.5.0 and SciPy 1.10.1 for exact correction/native-image replay.

## Reference amendment and active jobs (07:36 UTC)

The historical-reference control 39473 and unchanged retry 39484 failed the
strict SAM mask check. Isolated probe 39485 found a one-pixel difference
(698,368 total pixels, IoU 0.9999971142); no quality-based adaptation was made.
Read docs/SURFACE_ANNOTATED_REFERENCE_AMENDMENT_20261112.md and the frozen
fresh-reference-amendment.json. Historical attempts and source freezes remain.

Active study: output/surface-annotated-20261112-freshrefs. Fresh cohort control
39486 generates references, verifies/snapshots all 204 targets, and copies exact
reference bytes before Surface-PMC cohort 39487. Full control 39488,
fixed-pool diagnostic 39489, full surface 39490, unchanged timeout continuation
39491, full verifier/registrar 39492. The full surface also depends on completed
full control references. Exact commands are in freshrefs-submission.json.
Only superseded pending jobs 39474-39479 were cancelled; unrelated jobs untouched.

The ranker, method freeze, cohort, primary endpoint and secondary policies remain
unchanged. No new training. Original source checks and new amendment checks pass.
Independent cohort and pool audit scripts are prepared; report/gallery remain to
be built after completed results. Monitor with scripts/monitor_annotated_surface.py.
The active-study.json routes collection and audits to the fresh-reference study.
Deadline remains 11:46 UTC. Preserve valid long-running jobs beyond that window.

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


## Repository collaboration setup (2026-11-12; separate task)

Created a separate Git checkout at `C:/Users/daovu/Documents/Project/UIT-CS331-Amodal_Completion`, based on existing
GitHub main `14834b65a09878b20fe32267919a47f16e784c7d`. Pushed branch
`setup/reproducible-development` with commit `8828d703c051e5405727974c756af30e7f431f5a`.
Branch: https://github.com/baotodale06/UIT-CS331-Amodal_Completion/tree/setup/reproducible-development

Imported maintained benchmark/research source, fixtures and configs; preserved
original model code and two proposal documents. Added hashed Python 3.12 CPU/build
locks, installer, GitHub Actions and `docs/GETTING_STARTED.md`. Clean Linux source
export passed 271 Python tests (two expected optional-artifact skips), five Bash
regressions, synthetic smoke, config/plan checks and source/wheel builds. Original
local datasets, models, outputs, active research and cluster jobs were untouched.
No new GPU validation was performed. Subsequent local source changes need explicit
commits in the new checkout; the old local workspace is not a Git working tree.

Final setup head: `3ec585c6eed9ec5d57d41e668f9779b7cb06a99b`. A fresh Linux Git clone is clean.
GitHub CI run 43177837400 passed installation, regression/smoke, build and
configuration/cleanliness steps. The follow-up commit preserves vendored CRLF
blobs and pins current Node-24-compatible GitHub Actions.


## Vietnamese project presentation completed (2026-11-14)

Prepared [the full project presentation](output/project-presentation-20261114/README.md): 41 Vietnamese slides with English technical terms (30 main + 11 appendix), editable PowerPoint, PDF, standalone offline HTML, embedded speaker notes, a Vietnamese talk-track PDF/Markdown, Q&A, numeric CSVs and a delivery ZIP. Research directly inspected and rendered the public 17-slide UIT/InSecLab KLTN template; design sources and project evidence are indexed and hashed.

The presentation retains the confirmed controlled selection result, inconclusive 204-target primary transfer, full annotated LPIPS +1.191% and 4.733x reserved GPU-hours, failure denominators, historical SAM/LISA separation and missing human ratings. All 41 slides were rendered through PowerPoint; text bounds, PDF pages/numbers, source links, offline navigation and direct all-target CSV means/transitions were checked. No scientific code, configuration, model, inference, metric cache or cluster job was changed. Full native PNGs remain on the cluster, with the retained local audit scope unchanged.
## Shape-Guided PMC · 21 November 2026

Jobs 44902 and 44903 completed on four L40S GPUs. CPU audit 44926 replayed all 1,024 silhouettes, RGB errors and decisions. Mean IoU was 0.7142 for uniform K=8, 0.7856 for Shape, 0.7684 for Surface and 0.7827 for combined. See the [report](output/shape-20261121/REPORT.md) and [slide 17 panel](output/shape-20261121/slide-17.png). The matched natural-image continuation started with a fresh L40S control after a historical SAM reference mismatch.

The full natural-image study now reuses the verified Quadro controls. The queued full PMC and both queued Surface repeats used zero GPU time. The 204-target L40S PMC snapshot had already finished. Only Shape and combined use further GPU allocations. The full report labels cross-device comparisons and reports the number of byte-identical references.

The chat's four-candidate controlled comparison raises mask IoU from 0.7128 to 0.7785 (paired gain 0.0657, nominal 95% interval 0.0515–0.0813). CPU job 44951 measured supplemental hidden-core DreamSim on the 1,024 saved candidates in 7 min 56 s. At K=8, uniform selection scored 0.2234, Shape 0.2077, Surface 0.2008 and combined 0.2045. Lower is better. The Shape gain relative to uniform was 7.01% (2.69%–11.28%). Local replay verified all 128 decision hashes, selected scores and five paired intervals. [Per-candidate evidence](output/shape-20261121/hidden-dreamsim.json)


On the frozen 204 targets, the existing Quadro and L40S PMC snapshots scored 0.685276 and 0.689272 failure-inclusive LPIPS. The paired relative gain for L40S was −0.583% (nominal 95% interval −1.853% to +0.163%). Across the 166 byte-identical references, the means were 0.729470 and 0.729226. Both runs saved the same 86 outputs by target identity, and Quadro also completed dog COCOA/COCO_train2014_000000108113_dog (LPIPS 0.208432). L40S recorded completion_not_detected and received the standard 1.0 penalty there. This one transition contributed 0.003880 of the 0.003997 absolute mean LPIPS difference. The per-target comparison checked both execution inventories, config equality, all target seeds, inputs, scoring references, metric caches and per-target CSV values. [Saved comparison](output/shape-20261121/pmc-cohort-cross-device.json)

For slide 16, CPU jobs 44959 and 44960 used the saved Quadro outputs and the 321 COCOA annotations. PMC and Surface each saved 183/321 completions. Including every failure at IoU zero, mean amodal IoU was 0.46268 for PMC and 0.46442 for Surface. The paired gain was +0.00174 (nominal 95% interval −0.01390 to +0.01693). Pasting the checksummed completion alpha into the original image independently matched the benchmark’s clipping and indexing on all 642 target/method pairs. Each available native mask matched its completion alpha. The full comparison now checks the per-target control geometry against this result, and the report shows the verified historical slide 15 rows while the L40S arms run. [Native geometry and severity groups](output/shape-20261121/cocoa-historical-controls.json)

The 204-target Shape cohort finished on four L40S GPUs in job 44914 (6,593 elapsed seconds, 7.326 GPU-hours). Job 44943 verified 158 eight-seed decisions and two fallback selections in 36 seconds. Against the completed L40S PMC control, Shape scored 0.697381 versus 0.689272 LPIPS (relative gain −1.176%, nominal 95% interval −3.083% to +0.622%) and 0.344676 versus 0.350671 SSIM. Both saved 86/204 outputs, with 85 shared successes. Shape gained the COCOA dog and lost the VG banana. On 22 COCOA targets from 20 images, failure-inclusive amodal IoU increased from 0.23021 to 0.26312 (+0.03291, interval +0.00339 to +0.08318). The extra dog contributed +0.02145 of this mean gain. Historical Surface scored 0.683564 LPIPS; the cross-device Shape relative gain was −2.021% (−3.719% to −0.865%), and −1.887% on 166 identical references. A separate local bootstrap implementation reproduced the means and intervals for all five same-device metrics and both historical comparisons. [Cohort summary](output/shape-20261121/shape-cohort-vs-pmc-summary.json) · [Paired targets](output/shape-20261121/shape-cohort-vs-pmc-pairs.json)

Native Shape cohort traces show that 117/204 targets stopped before sampling, 61 used one eight-seed pass, 10 used two, five used three, two used four, and nine used six. The total of 158 pools matches audit 44943. The combined audit now compares each method’s first saved seed list, crop, observed inputs, prior, eight RGBs and eight masks before counting selector changes. The existing bicycle and white-house pools pass all equality checks and choose different slots. Whole-cohort counts await the combined snapshot.


The Shape cohort report now uses the Surface study's additive coverage analysis. Of the overall 0.008108 increase in failure-inclusive LPIPS, 0.007881 came from the 85 targets that both PMC and Shape saved. The one gained completion reduced the overall mean by 0.003926, and the one lost completion increased it by 0.004154. The 117 shared failures contributed zero. Each contribution uses all 204 targets. The remote snapshot hashes verified the four reasons Shape stopped before any candidate selection: 47 multiple distinct instances, 27 unmatched labels, 34 no-occlusion decisions and 9 masks too small. One additional target failed after candidate generation. The same reporting calculation will be applied to the combined and full results from their audited pairs. [Cohort report](output/shape-20261121/REPORT.md)


Job 44917 and audits 44944/44945 completed on the frozen 204 natural targets. Combined scored 0.683841 LPIPS and 0.355981 SSIM, saved 87 targets and used 7.332 allocated L40S GPU-hours. The relative LPIPS gain versus same-device PMC was +0.788% (nominal 95% interval -0.099% to +2.102%). Versus Shape on identical inputs/references it was +1.941% (+0.807% to +3.568%). Against published Quadro Surface the relative gain was -0.041% (-0.300% to +0.201%). The local multiplicity-based bootstrap reproduced all five combined-versus-PMC endpoints, the two historical Surface LPIPS contrasts and the direct same-device comparison. The full Shape continuation 44946 is underway. [Cohort replay](output/shape-20261121/cohort-local-replay.json)

The frozen 204-target table now includes the Surface study's source-level diagnostic. Combined versus Shape reduced failure-inclusive LPIPS by 2.74% on 109 VG targets (nominal 95% interval +1.03% to +5.67%) and 1.75% on 67 COCOA targets (+0.36% to +4.23%). The two groups of 14 targets gave +0.17% on free_images and -0.95% on laion, with both intervals crossing zero. All of these comparisons used the same L40S references and the original image-cluster bootstrap. Source-level intervals are descriptive and unadjusted. [Cohort table](output/shape-20261121/REPORT.md)


2026-11-21 UTC. The 204-target cohort had 22 targets with independent COCOA geometry. One plate was the only match with at least 60% occlusion (82.4%). All three L40S methods rejected its small visible mask before candidate generation. The cohort cannot establish how reranking works at that severity. The full severity analysis still requires the two full L40S evaluations. [Cohort report](output/shape-20261121/REPORT.md)

2026-11-21 UTC. Published the visible-mask control for slide 18. CPU job 45150 scored the frozen candidate pools, and CPU job 45155 rendered the report. An independent local render using the recorded Slurm allocations matched the published file byte for byte. The report contains the K=4 and K=8 paired intervals for mask IoU, hidden IoU, object LPIPS and COCOA amodal IoU. At 20:49 UTC Shape had 782 of 2,565 native records. Combined had its 204 preserved cohort records and was waiting for Shape. [Week 8 report](output/shape-20261121/REPORT.md)

The 204-target natural-image table now shows all four methods together. The historical Surface job (39487, Quadro RTX 6000) had LPIPS 0.683564, SSIM 0.356430, 87 saved targets and 7.304 reserved GPU-hours. The other three rows use L40S. The scoring references match byte for byte on 166 targets across devices. CPU job 45158 published the verified table. [Week 8 report](output/shape-20261121/REPORT.md)

The 32 controlled COCOA objects with independent outlines fall into groups of 3, 14, 11, 4 and 0 across the five Week 8 occlusion bins when measured in the actual model input. The calculation counts naturally hidden or artificially covered outline pixels inside each crop. For the four objects with 60% to under 80% hidden, original-frame amodal IoU averaged 0.4490 for uniform K=4 and 0.5477 for Shape. These four observations describe mixed natural and artificial occlusion. [Controlled COCOA table](output/shape-20261121/REPORT.md#occlusion-in-the-controlled-cocoa-crops)

For the 22 natural COCOA polygon matches in the 204-target cohort, historical Surface on Quadro scored 0.2676 mean amodal IoU. On L40S, Shape scored 0.2631 and Shape + Surface scored 0.2482. The three methods saved the same seven completions. The cross-device paired differences versus Surface were -0.0045 (nominal 95% interval -0.0167 to +0.0075) and -0.0194 (-0.0564 to +0.0005). Each failed completion contributes zero. The report now prints the direct Surface geometry comparison as each full L40S arm is audited. [Cohort geometry comparisons](output/shape-20261121/REPORT.md)

On the 166 cohort targets with byte-identical scoring references, the relative LPIPS gain of L40S PMC over Quadro PMC was +0.033% (nominal 95% interval -0.173% to +0.281%). Against historical Surface, Shape scored -1.887% (-3.873% to -0.616%) and Shape + Surface scored +0.066% (-0.109% to +0.267%). The analysis includes failed completions and resamples 159 images. CPU job 45219 published the intervals. [Week 8 results](output/shape-20261121/REPORT.md#204-natural-targets)

CPU jobs 45246 and 45247 verified and published the 204-target sampling counts. On the 87 targets where each arm sampled, Shape generated 1,264 candidate images in 158 passes with two first-seed fallbacks. Shape + Surface generated 1,200 in 150 passes with one fallback. The published and retrieved reports matched the expected digest. [Week 8 results](output/shape-20261121/REPORT.md)

On the 32 controlled images with independent COCOA outlines, mean amodal IoU was 0.6626 for Surface, 0.6740 for Shape and 0.6712 for Shape + Surface. The paired differences versus Surface were +0.0114 (nominal 95% interval -0.0101 to +0.0363) and +0.0086 (-0.0091 to +0.0284). Both intervals include zero. All three selectors used the same eight candidates per image. CPU job 45259 published this comparison. [Controlled outline results](output/shape-20261121/REPORT.md#controlled-images-with-cocoa-amodal-outlines)

The 204-target compute and cross-device candidate checks are in the Week 8 report. Shape used 7.326 L40S GPU-hours for 158 passes. Shape + Surface used 7.332 for 150 passes. On 59 cohort targets with matching input images, scoring references, first-pass crops, visible masks and seeds, the mean absolute difference between the raw Quadro Surface and L40S Shape candidate images was 0.584 channel levels on a 0-255 scale. CPU jobs 45288 and 45289 independently measured it from saved PNGs. At 23:24 UTC, Shape had 1,061 full-run records and combined was waiting with its original 204. [Week 8 report](output/shape-20261121/REPORT.md)

CPU job 45291 verified all 817 reference files for each 204-target L40S cohort copy against the saved source and destination. The final combined audit will perform the corresponding comparison against the completed Shape full reference set, including copy receipts from any resumed allocations. At 23:33 UTC Shape had 1,074 of 2,565 native records and combined had its original 204. [Recorded submissions](output/shape-20261121/progressive-l40s/submissions.tsv)

2026-11-21 23:58 UTC. The [Week 8 report](output/shape-20261121/REPORT.md) now explains how LPIPS and SSIM use the visible SAM reference on a white background. For the COCOA bed in train image 009199, adding Surface reduced LPIPS from 0.4617 to 0.4549 while amodal IoU dropped from 0.7605 to 0.6940. CPU job 45293 checked the score rows and independently reprojected the native completion masks. CPU job 45294 published the two paragraphs. The full Shape and combined evaluations remain in progress and dependency-pending, respectively.

The three completed 204-target cohort metadata snapshots now pass the repository's provenance verifier locally. Each of the 612 score rows matches the native target seed, status and metric cache or failure penalties. The snapshots preserve the original code, environment and target record digests for verifying cohort reuse in the full continuations. [Shape cohort execution](output/shape-20261121/progressive-l40s/shape-cohort/execution.json) · [Combined cohort execution](output/shape-20261121/progressive-l40s/combined-cohort/execution.json)

2026-11-22 01:03 UTC. Verified that 211 of the 321 COCOA polygon targets belong to the frozen unseen split, spanning 202 images. The archived PMC and Surface runs scored 0.4231 and 0.4201 mean amodal IoU there, including failed completions. Report job 44950 will add both new methods and their paired silhouette comparison on those same targets. Shape had processed 1,258 of 2,565 targets and combined retained its 204 cohort records while waiting. [Verification details](session.md)

2026-11-22 01:10 UTC. The local 204-target Shape and combined snapshots now include the native prior, all eight candidate masks and the selection record for every sampling pass. Local replay recovered every one of the 308 candidate choices. Among 73 first-pass target pools where Shape and the frozen Surface ranking picked different candidates, the joint rule chose Shape's candidate 24 times, Surface's 27 times and a third candidate 22 times. [Transfer, hashes and checks](session.md)

2026-11-22 01:40 UTC. Both registered Quadro controls passed a local replay of their full registry entries. The queued CPU audits will apply the same check to the new Shape and Shape + Surface results. The 01:38 UTC collector recorded 1,309/2,565 Shape targets, including its frozen cohort. Combined was waiting with its 204 retained targets. [Execution log](session.md)

2026-11-22 01:46 UTC. On the same L40S cohort, adding Surface to Shape increased failure-inclusive VGG feature similarity by 4.641% (nominal 95% interval +2.078% to +8.472%) and SSIM by 3.280% (+1.118% to +6.653%). The paired CLIP difference among the 86 common successes was -0.120% (-0.927% to +0.692%). The full report will use the same denominators and resampling rules on the two completed full audits. [Cohort metrics](output/shape-20261121/REPORT.md)

2026-11-22 01:56 UTC. Corrected the earlier description of the 166-target cohort subset. Those targets matched on all three reference files. The RGBA crop used by the appearance evaluator matched on 174 targets from 167 images. On this subset Shape versus historical Surface gave -1.807% relative LPIPS gain and combined versus historical Surface gave -0.035%. Both original comparisons are retained. The queued full audits will measure both subsets using the completed records. [Cohort results](output/shape-20261121/REPORT.md#cross-device-comparison-with-identical-lpips-scoring-crops)

2026-11-22 02:00 UTC. Verified the first 1,024 full Shape records in annotation order against their source inventory. The 92 retained records matched the frozen cohort. For the 932 fresh records, local replay from native masks matched all 8,680 IoU scores and 1,085 candidate selections, including one fallback. This is a check of selection during the ongoing run. Full accuracy and coverage await scoring and both completed executions. [Recovered native masks and decisions](output/shape-20261121/progressive-l40s/shape-selector-0001-1024.tar.gz)

2026-11-22 02:11 UTC. The local and cluster cohort metadata exports matched across 503 Shape files and 496 combined files, including all 308 selector decisions and 173 success score caches. CPU jobs 45303 and 45304 will export the corresponding full records after their audits. The local collector will retrieve both archives and the final report. Shape had processed 1,360 of 2,565 targets. Combined retained 204 while waiting. [Export jobs](output/shape-20261121/progressive-l40s/metadata-export-jobs.tsv)

2026-11-22 02:19 UTC. Recovered the frozen cohort silhouettes and independently reproduced all 88 scores from 22 COCOA outlines across the four methods. Shape and combined both saved the same seven objects. The dog accounted for 84.6% of the observed mean IoU decline with combined. The interval for that group difference includes zero. Shape had processed 1,374 of 2,565 targets and combined had its 204 retained targets. [Native outline replay](output/shape-20261121/cohort-cocoa-native-replay.json)

2026-11-22 02:30 UTC. The local collector now verifies all 321 natural COCOA outline scores after each full audit by recovering native outputs and projecting completion alpha into the original frame. The same path matched every Shape and combined score on the 22 annotated cohort targets. Shape had processed 1,385/2,565 targets. Combined was waiting with 204 retained. [Local outline replay](scripts/replay_shape_outlines.py)

2026-11-22 02:35 UTC. Independently reproduced the six available slide 15 rows from their native score CSVs and Slurm allocation seconds, including the historical Surface timeout. Shape and combined took 7.326 and 7.332 allocated L40S GPU-hours on the retained cohort. Shape 44946 had 1,392/2,565 native records. Combined 44948 was queued. [Slide 15 score and cost receipt](output/shape-20261121/slide-15-cohort-and-controls.json)

2026-11-22 02:41 UTC. CPU job 45308 compared the live first 1,024 Shape results with the prior independent selector review. All 932 new and 92 retained records matched. The final CPU audit will check those digests again against the closed execution. Shape had processed 1,416/2,565 targets and Shape + Surface was queued. [Full audit](scripts/compare_shape_progressive.py)

2026-11-22 02:45 UTC. Shape had processed 1,419/2,565 targets. Actual target timings and the historical pass counts for the remaining work now project the last inference record around 22:16 Vietnam time, after the original 21:51 allocation end. Pass counts vary by method and scoring follows inference. Confirmed the queued recovery dependencies for a Slurm-reported timeout. The full scores are pending.

2026-11-22 02:54 UTC. The local collector now reconciles each full metadata export with the official target order, closed execution, selector choices and paired report as soon as the archive arrives. A trial on the completed cohort matched all 408 method-target score rows and 308 progressive choices. Shape job 44946 had 1,426/2,565 records. Combined job 44948 was waiting with its own 204 retained records. Both full registrations and both local archive reviews are pending. [Archive verifier](scripts/verify_shape_metadata.py)

2026-11-22 02:59 UTC. Checked the live 21-slide Canva presentation against the November 21 PowerPoint and corrected the report's page references. Shape methodology belongs on slides 11–12 and 15, the full results table on 16, occlusion severity on 17, the umbrella and bus on 18, the ablation on 19, and limitations and conclusions on 20–21. The 72 table lines matched the previous report, and all 44 local links resolved. CPU report job 45310 completed. Shape job 44946 had 1,435/2,565 results. Combined 44948 retained 204 while waiting. [Report for the current deck](output/shape-20261121/REPORT.md)

2026-11-22 03:10 UTC. Checked the native scoring images from both historical controls and L40S Shape for 1,136 targets. The LPIPS crops matched across devices on 948. Of the 174 matched crops in the frozen cohort, 74 had empty reference masks and the same failure penalty across all five executions. The full comparisons will use the completed 2,565-target inventory. Shape had 1,451 result records, and Shape + Surface was queued. [Per-target reference check](output/shape-20261121/progressive-l40s/shape-reference-overlap-1136.json)

2026-11-22 03:20 UTC. Prepared the independent numerical check for the final tables on slides 16 and 17. Both Quadro control rows reproduced from the native scores and allocation records. All ten historical LPIPS means and ten historical amodal IoU means matched across the five occlusion groups. Collector 15260 will check all four methods against the exported tables after the two L40S runs finish. Shape had 1,471/2,565 records and Shape + Surface was queued. [Slide table replay](scripts/replay_shape_slide_tables.py)

2026-11-22 03:34 UTC. Added the training-label comparison to the Vietnamese limitations for slides 20-21. CPU report job 45316 completed. All 72 table lines matched the previous report and all 45 local link targets resolved. Shape job 44946 had 1,491 of 2,565 result records. Combined 44948 was dependency-pending. CPU job 45320 will replay the full LPIPS intervals on the subsets with matching reference files or matching scoring crops when audits 44947 and 44949 complete. [Updated report](output/shape-20261121/REPORT.md)

2026-11-22 03:39 UTC. The report now includes Vietnamese captions for the qualitative comparison on slide 18 and the equal-budget comparison on slide 19. All 12 image tiles matched their source images pixel for pixel. Shape job 44946 had 1,494 of 2,565 inference results. Full scoring is pending. [Slide 18 examples](output/shape-20261121/REPORT.md#slide-18--matched-qualitative-examples)

2026-11-22 03:44 UTC. The result collector will retrieve and reconcile the full paired LPIPS interval receipt after both audits, in addition to checking the slide 16 and 17 tables. Shape reached 1,504 of 2,565 target records. Both new full scores are still pending. [Interval replay](scripts/replay_shape_reference_intervals.py)

2026-11-22 03:50 UTC. Shape saved 724 of the 1,294 Visual Genome targets on L40S. Both registered Quadro controls were checked against their existing executions and scored rows. PMC saved 726 and Surface saved 729. The full Shape run had 1,510 inference records, and Shape + Surface was pending. [VG coverage](output/shape-20261121/progressive-l40s/shape-vg-inference-coverage.json)

2026-11-22 03:59 UTC. Across all 1,294 Visual Genome targets, the CPU replay of the saved L40S Shape predictions gave a failure-inclusive SSIM of 0.460034. Registered Quadro PMC and Surface scored 0.463503 and 0.471548 on those targets. All 109 retained Shape scores agreed with the cohort. Shape job 44946 had 1,517 of 2,565 native records at this check. Combined job 44948 was dependency-pending. [VG SSIM by target](output/shape-20261121/progressive-l40s/shape-vg-ssim.json)

2026-11-22 04:05 UTC. The Vietnamese report now includes the Visual Genome SSIM result below the pending full benchmark table. The scheduled Shape audit will reconcile the early scores and their input hashes against the final execution. The same check passed on all 109 VG targets in the scored cohort. Shape had 1,523 of 2,565 records and Shape + Surface was queued. [Week 8 report](output/shape-20261121/REPORT.md)

2026-11-22 04:18 UTC. The report now gives hidden LPIPS and DreamSim on the same 32 COCOA images and the same four candidates used to measure independently annotated amodal IoU. Shape improved IoU and hidden-core DreamSim versus uniform selection. The paired LPIPS interval included zero. At 04:18, Shape had 1,544 target records and combined was queued. [Report](output/shape-20261121/REPORT.md#controlled-images-with-cocoa-amodal-outlines)

2026-11-22 05:17 UTC. The Week 8 report now gives Visual Genome SSIM on the 722 targets where both Shape and historical Surface saved a completion. Shape scored 0.8224 and Surface 0.8368 on those targets. Across all 1,294 VG targets the corresponding means are 0.4600 and 0.4715, including zero SSIM for failures. Cluster access recovered and the collector counted 1,648 Shape records at 05:16. Shape + Surface was dependency-pending. [Report and source scores](output/shape-20261121/REPORT.md#ssim-trên-visual-genome)

2026-11-22 05:21 UTC. On the 82.67%-occluded COCOA plate, original-frame mask IoU was 0.2061 for Shape, 0.1641 for historical Surface and 0.9809 for historical PMC. Three of the four severe cases now bound the Shape group mean at 0.3015 or less. Its exact value awaits the elephant. Both full registrations remain pending. [Per-case scores, derivation and native masks](discoveries.md#shape-on-the-four-most-occluded-natural-cocoa-targets)

2026-11-22 05:30 UTC. The Week 8 report now includes the natural plate example with links to the predicted mask and the independent COCOA outline. CPU rendering completed and the local report matches the remote copy. [Week 8 report](output/shape-20261121/REPORT.md)

2026-11-22 05:45 UTC. Shape scored 0.5984 LPIPS across all 1,294 Visual Genome targets. Registered PMC scored 0.5932 and Surface scored 0.5839. Shape's LPIPS was 0.88% higher than PMC's (nominal 95% interval 0.20% to 1.60%) and 2.48% higher than Surface's (1.83% to 3.21%). Shape predictions came from L40S, and the historical predictions came from Quadro. Both full registrations are still pending. [Vietnamese report](output/shape-20261121/REPORT.md#lpips-tr%C3%AAn-visual-genome)

2026-11-22 05:54 UTC. The Week 8 natural-occlusion section now shows why the Shape selector missed most of a severely occluded plate. The best first-pass mask in the existing pool reached 0.607 amodal IoU. The mask closest to the U-Net prior scored 0.162. CPU job 45340 replayed all six saved passes and rendered the comparison. Both full registrations remain pending. [Week 8 report](output/shape-20261121/REPORT.md) ? [Figure](output/shape-20261121/shape-plate-first-pass.png)

2026-11-22 06:05 UTC. On 1,094 Visual Genome targets from 1,056 previously unseen images, Shape scored 0.6190 LPIPS, historical PMC 0.6132 and historical Surface 0.6051. Two implementations reproduced the paired image-cluster intervals from 5,000 draws. Shape had 1,728 of 2,565 full records. Combined had its 204 retained records. [Scores and frozen subset](discoveries.md#shape-on-previously-unseen-visual-genome-images)

2026-11-22 06:15 UTC. The Vietnamese report explains the candidate budgets and historical controls for the full natural-image comparison. The report will include failure-inclusive LPIPS and COCOA outline IoU differences against both controls as each full audit finishes. CPU job 45341 published the update. Shape had 1,736 records and Shape + Surface was queued. [Week 8 report](output/shape-20261121/REPORT.md)

2026-11-22 06:20 UTC. Updated the report against the live 22-slide Canva deck and the synchronized PowerPoint. Methods belong on slides 11–13 and 16, full results on 17, severity on 18, the umbrella and bus on 19, the fixed-budget comparison on 20 and limitations and conclusions on 21–22. The newest report preserves all existing measured values. Shape had 1,753 full target records. [Report](output/shape-20261121/REPORT.md) · [Current presentation](https://www.canva.com/d/NzI2bnio6-0Ph2z)

2026-11-22. The two test objects verified to be fully visible and inside the input crop before artificial occlusion were 24.43% hidden (bowl) and 34.93% hidden (cat). Neither reaches the paper's 50–80% difficult range. Per-case amodal IoU, hidden LPIPS and hidden-core DreamSim are available for slide 21. [Counts and four-candidate results](discoveries.md#occlusion-and-hidden-appearance-on-the-two-initially-fully-visible-cocoa-objects)

2026-11-22 06:31 UTC. The report now gives the exact 50–80% hidden subgroup from the 32 controlled COCOA crops. Shape selected a mask above the mean IoU of the same four candidates on six of nine images. Mean amodal IoU was 0.6192 for Shape and 0.5682 for uniform selection. All nine fail at least one original visibility or framing criterion. At 06:31 UTC, Shape had 1,774 inference records and combined retained 204. The two full registrations are pending. [Report](output/shape-20261121/REPORT.md)

2026-11-22 06:38 UTC. Added hidden-region mask IoU on the frozen natural cohort to the Vietnamese report. Across 15 annotated objects from 13 images with genuine hidden pixels, Shape scored 0.2246 versus 0.1813 for L40S PMC and 0.2104 for historical Surface. Shape versus Surface's nominal interval includes zero. Shape job 44946 had 1,784 inference records, and combined job 44948 remained queued. [Report](output/shape-20261121/REPORT.md) · [Hidden-region measurements](output/shape-20261121/cohort-cocoa-hidden-iou.json)

2026-11-22 06:44 UTC. Verified the full hidden-region comparison inventory of 230 COCOA objects from 220 images. Its 153 unseen objects come from 145 images. Historical Quadro PMC scored 0.2596 hidden-region IoU and Surface scored 0.2663. Two CPU replays verified the archived masks. The existing Shape and combined full audits will calculate the same supplemental measure at closure. At 06:44 UTC Shape had 1,792 inference records and 86 saved score caches, and combined was dependency-pending. [Control scores](output/shape-20261121/full-cocoa-hidden-historical.json)

2026-11-22. Prepared slide 2 red-board replacement in Canva design DAHbEx8zTOs, transaction 1816599509754809146. User clarified that diving board means the existing red board before/after example on slide 3. Reused MAHbjoMOhkU and MAHbjtc1iDs in 210 x 210 square frames, reset crops to show the full images, and removed 44 old portrait annotation shapes/question mark. All other remaining slide 2 elements match the initial state exactly; 83 elements remain. Draft visually reviewed and shown inline. Canva edit-design requires explicit preview approval before commit; approval question is pending. Preview: output/bao-cao-cs331-20261122/Xem_truoc_trang_02_van_do.png. Receipt and complete draft: output/bao-cao-cs331-20261122/canva-slide2-board-draft.json. Saved presentation is still unchanged by this task.

2026-11-22. User approved saving the slide 2 red-board replacement. Committed Canva transaction 1816599509754809146 to DAHbEx8zTOs. Saved revision 32 has 23 pages. Reopened the saved design and verified its complete slide 2 element data matches the approved draft exactly: 83 editable elements, both original red-board images in square frames, old outline/question mark removed. The public thumbnail was stale; a fresh transaction render visually confirmed the saved replacement. Closed verification-only transaction 1681491522446072436 without edits. Saved preview: output/bao-cao-cs331-20261122/Trang_02_van_do_da_luu.png. Commit and verification receipts: canva-slide2-board-commit.json and canva-slide2-board-saved-verification.json in the same directory. Edit link: https://www.canva.com/d/WMkh6b6PvXbhpV3 . This completes the pending slide 2 image replacement.


2026-11-22. User explicitly requested deletion of slide 3. Deleted page PBYBhRL5bbKvX8bg from Canva DAHbEx8zTOs via merge job 15d964e7-253e-438d-92d2-b2ba667b9ad0 (success). Saved design now has 22 pages. The red board pair remains on slide 2. Draft transaction 4861032860694176687 updates 21 editable footers to the new /22 page order and the page 15 full-results cross-reference from 18 to 17. Exactly 22 text changes were verified; no other elements changed. These numbering edits await required Canva preview approval. Receipt and previews: output/bao-cao-cs331-20261122/slide3-removal/.


The cover footer was stored inside a shape and is now included in the same pending transaction 4861032860694176687. All 22 sequential footers 01 / 22 through 22 / 22 and the page 15 reference to results on page 17 are verified. Draft contains 23 text replacements in total. Final complete draft and operations: output/bao-cao-cs331-20261122/slide3-removal/final-renumber-draft.json. All 22 draft page previews inspected. Slide 3 deletion itself is already committed; only the numbering and reference corrections need preview approval.

2026-11-22. User approved the numbering corrections after deleting slide 3. Committed transaction 4861032860694176687 to Canva DAHbEx8zTOs. Reopened the saved design and verified all 22 footers (01 / 22 through 22 / 22), the removed page's absence, and slide 15's full-results reference to slide 17. Every saved page exactly matches the approved complete draft. Fresh saved renders of slides 1, 3, 15 and 22 passed visual review. Closed verification-only transaction 6375368235462951893 without changes. Evidence: output/bao-cao-cs331-20261122/slide3-removal/saved-verification.json and saved-slide-*.png. The deletion and all numbering corrections are complete; no presentation approval is pending. Edit link: https://www.canva.com/d/jGn0nMmVpGwLTXQ .


2026-11-22. Prepared the frozen surface-full-v1 human preference study (e1dd2c278789385727448373): 1,387 shared successes, 1,178 excluded pipeline outcomes, counterbalanced anonymous A/B, five native lossless images per case. All 6,935 images passed full remote and local validation. Offline recovery is ready; Start review.cmd opens the current UI. Six Python study tests, three Node core tests, 15 existing human-review tests, syntax checks, and isolated Edge browser QA passed. QA uses a separate qa- study and TEST_ONLY rater; no research responses exist. Private Sites publication retains project appgprj_6b02fe6d29848191b670dd42ed2ab89d and exact source commit 2ee883f9b4c6c9fdd5f409cbd12c15f6f2d7150b. The 1,465,517,904-byte deployment archive exceeds the native 536,870,912-byte upload limit, so publication uses the verified source-based remote build. Receipt: output/human-preference-20261122/deployment.json. No GPU jobs, inference, or retuning were started for this task.


2026-11-22 15:54 UTC. Human preference reviewer publication completed: https://amodal-preference-dv20261122.rompingsalmon2.chatgpt.site
The existing owner-private Site now serves the compact folder-based reviewer from source b0c8bd34ba624002dbe951eb56f693b3ad29cb54 (saved version 2). Open study folder must select output/human-preference-20261122/dist once per hosted session; all images stay local and are checksum-verified. The offline launcher works without folder selection. All 6,935 images are locally validated, and the updated browser test passed the folder-required state, voting, disclosures, ties, skips, backup export/restore, wrong-round rejection and mobile layout with no console exceptions. This supersedes the earlier publication-in-progress entry. The failed full-image remote-build receipt and original archive/source remain preserved in site-tools; deployment.json is the successful final receipt. No publishing, download, or validation work remains pending. The next input is the user's ratings; Shape and combined remain separate future rounds. No human preference scores have been produced.


2026-11-22 16:07 UTC. Shape continuation 45361 started on four L40S GPUs after a scheduling adjustment to fit before maintenance. It resumes the 2,516 completed records from timed-out job 44946. Combined 44948 follows Shape with a six-hour minimum allocation, and both recovery controllers remain queued. Full verification and comparison are pending.


2026-11-22 16:20 UTC. The native Shape geometry check now covers all 321 annotated COCOA targets and 230 eligible hidden regions. Both full-outline and hidden-region paired intervals versus the historical controls include zero. All retained cohort geometry scores match. The latest 24-slide report mapping is installed and checked. A 30-minute follow-up will continue this experiment from 23:50 Vietnam time, with full scoring, registration and the combined run still pending. [Native geometry evidence](output/shape-20261121/shape-cocoa-preverification.json)


2026-11-22 16:51 UTC. Shape 45361 has 2,542/2,565 inference records, with three workers finished. Verified all 2,516 pre-timeout records and 86 score caches byte for byte. Combined and full audits remain queued behind their prerequisites. The collector and recurring follow-up are live.


2026-11-22 17:21 UTC. Shape has 12 inference targets remaining and its final worker is actively generating candidates. Full scoring and registration follow inference. Combined remains queued, with the collector and scheduled follow-up active.


2026-11-22 18:00 UTC. Shape is now fully verified, registered, exported and locally reconciled. The collector verified all 7,074 metadata files, 2,565 score rows, 2,961 selector choices and all 321 native COCOA outlines. The frozen 204 cohort rows and 2,516 timeout records remain unchanged. Independent replay confirmed the four Shape LPIPS estimates on 2,048 identical reference sets and 2,167 identical scoring crops; all four favor the historical controls. Full LPIPS is 0.618641, SSIM 0.442871, saved outputs 1,393 and actual cost 109.571 allocated GPU-hours, including cohort and timeout. Both full geometry intervals remain inconclusive. Combined 44948 is running on four L40S GPUs; its full reference copy matches Shape for all 2,565 targets. Existing recovery controllers, collector and recurring schedule remain active. [Local reconciliation](output/shape-20261121/shape-full-local-reconciliation.json) ? [Preservation and combined references](output/shape-20261121/shape-preservation-and-combined-reference-replay.json)


2026-11-22 18:21 UTC. Combined has 269/2,565 completed inference records and all four workers active. All 204 frozen records remain intact; all 269 completed targets match Shape on input hashes, seeds and scoring references. Full scoring, registration and downstream audits remain pending on live inference. Collector and scheduled follow-up remain active.


2026-11-22 18:51 UTC. Combined is still generating normally, with 294/2,565 completed inference records. All four workers are active and have no Python tracebacks. Recovery and downstream audit jobs retain their dependencies. Full scores await inference completion; collector and scheduled follow-up remain active.


2026-11-23 07:16 UTC. Current cluster state is unavailable because the hostname does not resolve and the university VPN adapter is disabled. The live collector is retrying; its last successful combined count was 294 at 18:51 UTC yesterday. Do not interpret this stale count as current progress. Shape remains locally verified; combined recovery status needs a fresh cluster check. The scheduled follow-up remains enabled.


2026-11-23 07:20 UTC. VPN access and the collector are restored. Combined had safely timed out with 700 completed records. Its existing controller already submitted continuation 45377 and recovery controller 45378. All 700 saved records, 204 frozen records and 87 retained score caches were verified. Shortened the pending continuation to four hours; Slurm predicts 15:44-19:44 Vietnam time today, before maintenance at 20:00. Its CPU recovery request is ten minutes so it can fit afterward. No duplicate run was submitted.


2026-11-23 07:48 UTC. Combined continuation 45377 is running on four L40S GPUs with 707/2,565 records. It started at 14:40 Vietnam time and its configured end is 18:40 today. All four workers are active without Python tracebacks. Independently verified the 700 timeout records, 204 frozen cohort records and 87 score caches. Its reference files match Shape across all 2,565 targets. Collector and recovery controller remain live or dependency-pending. No job or experiment change was needed. [Monitoring receipt](output/shape-20261121/monitoring/20261123T074828Z-combined-continuation.json)


2026-11-23 08:14 UTC. Combined continuation 45377 has 753/2,565 completed records. All four L40S workers are active with zero Python tracebacks. All 700 timeout records, 204 frozen records and 87 score caches remain unchanged; every completed input, seed and scoring-reference set matches Shape. Recovery/audit/export/report jobs retain their dependencies. The collector is healthy. Full scoring and registration await live inference; no job change was needed. [Verification receipt](output/shape-20261121/monitoring/20261123T081446Z-combined-continuation.json)


2026-11-23 08:44 UTC. Combined continuation 45377 has 802/2,565 completed records. All four workers remain active with zero Python tracebacks, and all preserved records, cached scores and scoring references pass verification. Full scoring and registration are still pending. The collector and scheduled follow-up remain enabled; no recovery or job change is needed. [Verification receipt](output/shape-20261121/monitoring/20261123T084440Z-combined-continuation.json)


2026-11-23 09:14 UTC. Combined continuation 45377 has processed 833/2,565 targets. All four workers are active without Python tracebacks. The 700 timeout records, 204 frozen cohort records, 87 score caches and full scoring-reference inventory remain intact. No terminal failure or new recovery job is present. The collector and recurring follow-up remain enabled; final scoring and registration await inference completion. [Verification receipt](output/shape-20261121/monitoring/20261123T091450Z-combined-continuation.json)


## Shape-only Slurm continuation started (23 November 2026)

Current user explicitly authorizes Quadro/L40S/Kaggle training for speed, superseding
older Colab-only instructions. Shape-only objective remains >=3% on all four full
metrics. New isolated study: output/shape-v2-slurm-20261123/README.md.
Submitted four-Quadro preflight 45631 and dependent 60-epoch training 45632.
Same verified 9,494 train / 1,264 validation objects; no holdout transferred.
Separate Slurm configuration records DDP4, per-rank BatchNorm and FP16 GradScaler;
original Colab package and existing combined jobs are preserved. All sources and
assets are hashed. Training and benchmark success are not yet established.
Maintenance is 2026-11-23 13:00 UTC to 2026-11-25 13:00 UTC. Monitor jobs, finish
training, freeze policy, assess geometry, GPU-validate isolated runtime, then
matched-reference full evaluation/verification/registration and four-metric gate.



## Original Shape-Guided PMC follow-up (23 November 2026)

2026-11-23 09:45 UTC. Combined continuation 45377 has processed 879/2,565 targets. All four workers remain active without Python tracebacks. Preservation and scoring-reference checks passed. The separate Shape-v2 training job 45632 was left untouched; all 47 files in the original combined execution's frozen source manifest still match the live cluster project. Collector and follow-up remain enabled. Final combined scoring and registration await inference completion. [Verification receipt](output/shape-20261121/monitoring/20261123T094500Z-combined-continuation.json)


## Shape-only Slurm training and evaluation chain active (23 November 2026)

Current user explicitly authorizes Quadro/L40S/Kaggle for speed, superseding older
Colab-only instructions. Objective remains >=3% on all four full-benchmark metrics.
Read output/shape-v2-slurm-20261123/README.md; goal is ACTIVE and not achieved.
Four-Quadro preflight 45631 passed. Training 45632 is running for 60 epochs with
resumable checkpoints, same 9,494 train / 1,264 validation objects and original
model/loss/selection. Separate Slurm contract records DDP4, per-rank BatchNorm and
FP16 GradScaler. Original Colab bundle and existing combined work are preserved.

Queued chain: geometry validation/freeze/holdout 45634 -> four-GPU smoke 45635 ->
90-minute full benchmark slice 45636 -> continuation controller 45637. Controller
resumes infrastructure interruptions unchanged, never application failures or a
completed bad result. Evaluation is in its own frozen source snapshot. Reserved
1,183-object/195-image holdout is in a separate evaluation-only directory, unused
by training. No post-training geometry or benchmark result exists yet.

352 local tests (five optional torch skips), six Bash regressions and CPU smoke
passed. All 29 focused tests passed in cluster torch without skips. Runtime job
45633 passed CUDA extension/FFT/pipeline imports but then failed on a checkpoint
loader incompatibility and missing fixtures; both were fixed before the passing
focused replay. Keep the failed job log and distinction. See validation.json.

Maintenance: Nov 23 13:00 UTC to Nov 25 13:00 UTC (20:00 Vietnam). Preserve running
jobs, recover completed training, monitor the queued geometry/smoke, then benchmark
continuations, verification/registration, full/unseen intervals, native geometry,
compute cost and the four-metric gate. Do not claim gains from training progress.



## Original Shape-Guided PMC monitoring (23 November 2026, 10:15 UTC)

Combined continuation 45377 has processed 927/2,565 targets. All four workers remain active with zero Python tracebacks. Preserved results, score caches, scoring references and all 47 recorded source files pass verification. Added the source-manifest and captured-file replay to the local read-only monitor; its live execution passed. No Slurm, GPU experiment or independent Shape-v2 job was changed. Final combined scoring and registration remain pending. [Verification receipt](output/shape-20261121/monitoring/20261123T101513Z-combined-continuation.json)


2026-11-23 10:44 UTC. Original combined continuation 45377 has processed 980/2,565 targets. All four workers are active with zero Python tracebacks. Frozen results, score caches, references and all 47 recorded cluster source files passed verification. Its allocation still ends at 11:40:44 UTC, with the existing recovery controller pending. Full scoring and registration remain pending. Collector and scheduled follow-up stay enabled. [Verification receipt](output/shape-20261121/monitoring/20261123T104454Z-combined-continuation.json)


## Original Shape-Guided PMC recovery verified (23 November 2026, 12:21 UTC)

Confirmed combined 45377 TIMEOUT after 04:00:15, ending 11:40:59 UTC. Existing controller 45378 completed in five seconds and automatically submitted continuation 45653 and controller 45654. At 12:17 UTC the pending continuation was shortened to 30 minutes and its pending controller to five minutes, using scontrol update on those existing IDs. No new job was submitted by the assistant. Slurm confirmed 45653 RUNNING on gus from 12:17:21 UTC, four L40S GPUs, configured end 12:47:21 UTC (19:47 Vietnam time).

Read-only replay at 12:21:32 UTC verified execution 20261123T122049.520157Z-e221e354 and the ordered 2,565-target inventory. Its 1,085 records comprise 592 success, 233 no_completion, 156 ambiguous_target and 104 not_detected. These are the preserved prefix, not a newly scored full result. All four new worker logs were updated within seven seconds and had zero Python tracebacks. Every completed input, seed and scoring reference matches Shape. All 1,085 records and 87 caches from timeout 45377, all 700 earlier records and caches from 44948, and all 204 frozen cohort records remain byte-identical. The archived partial-target replay passed all eight directories across both timeouts: seven empty directories and the 22 files (3,771,295 bytes) for VG/2350254_bus. Original partial attempts remain archived while unfinished targets resume normally.

All 47 immutable captured source files and corresponding live cluster files match frozen digest 2d9ae393bdc7f14977147d3a364f3491540d7558a92556a079e1487aad42f302. The new transfer combined-refs-full-45653.json verifies 10,261 files across 2,565 targets with zero copies, identical to both previous full maps; SHA-256 03f6399389d700ae1a049b97fdf9634ed14e36e4f574112b87e40cbfc8caa5e0. Local and remote timeout-45377 receipt hashes agree at a4acdc71c2fc74836746e3b106206d3153190caa6478af4486c4b11fc4abc661. All seven closed Shape reconciliation inputs, including REPORT.md, remain valid.

Controller 45654 remains pending afterany:45653; audit 44949 waits afterok:45653?afterok:45654. Export 45304, reference replay 45320 and report 44950 remain dependency-pending. The terminal combined allocation subtotal is 47.358889 GPU-hours across cohort 44917 and interrupted jobs 44948/45377; this excludes ongoing 45653 and is not final cost. Maintenance root_1 remains 23 November 13:00 UTC through 25 November 13:00 UTC. Preserve the next pending continuation across maintenance and inspect the ledger/controllers before recovery. Running time-limit extensions remain disallowed by the earlier denial.

Collector PID 26540 is alive, with an empty error log and a successful 12:20:09 UTC poll; it was not replaced. Canva still has 24 pages and exactly the same research-slide text despite a newer saved timestamp. Its original-study mapping and placeholders are unchanged; no presentation edit was made. The independent Shape-v2 chain remains untouched. Added only a parameterized read-only monitoring verifier to avoid reusing hardcoded 45377 identities. This check passed against the real continuation; no production experiment source or scientific configuration was changed.

Evidence: [continuation and archive replay](output/shape-20261121/monitoring/20261123T122132Z-combined-continuation.json), [scheduling receipt](output/shape-20261121/monitoring/20261123T1219-combined-recovery.json), [read-only verifier](output/shape-20261121/monitoring/verify_combined_continuation.py), [saved slide text](output/shape-20261121/monitoring/20261123T122133Z-canva-research.json). The scheduling receipt's actual observed_at is 12:17:21 UTC despite its filename, and its immediate state was PENDING; subsequent sacct and scontrol confirm RUNNING. Combined full scoring, registration, exports, independent replays and final report closure are still pending; the follow-up remains enabled.


## Original Shape study queued across maintenance (23 November 2026, 12:48 UTC)

Continuation 45653 reached TIMEOUT at 12:47:30 UTC after 00:30:09, adding 43 completed records since its 1,085-record starting snapshot. Existing controller 45654 COMPLETED in five seconds at 12:47:37 UTC and automatically submitted continuation 45666 and controller 45667. No new Slurm submission, cancellation or duration change was made by this scheduled check. Job 45666 retains its 24-hour request on four L40S GPUs with four workers; Slurm reports PENDING, ReqNodeNotAvail/Reserved_for_maintenance, with no predicted start. Controller 45667 waits afterany:45666; audit 44949 waits afterok:45666?afterok:45667. Export 45304, reference replay 45320 and report 44950 remain dependency-pending.

The final snapshot contains 1,128 result records: 619 success, 246 no_completion, 159 ambiguous_target and 104 not_detected. All 87 score caches, 204 frozen cohort records, earlier 700- and 1,085-record snapshots, and every completed input/seed/scoring-reference identity passed independent replay. Timeout receipt combined-timeout-45653.json has matching local/remote SHA-256 2e16eb5bad124d3ce0b67ce0e9f3be53a3d325feb2dccba97dea1f36fc3565b3. Its four archived partial targets also passed native file rehashing: plate 110 files, greenboat 88, airplane 22, bicycle empty. All 12 archived partial directories across the three timeouts are preserved. All three full reference-transfer maps agree across 10,261 files/2,565 targets, and all 47 frozen captured/live source files remain unchanged. The seven closed Shape reconciliation inputs, including REPORT.md, remain valid.

Original combined allocated GPU-hours so far are 49.368889: cohort 44917 plus interrupted 44948, 45377 and 45653. This includes 2.010000 GPU-hours for the final pre-maintenance slice and is not final study cost. Combined still needs 1,437 target records and official full scoring/registration. Only the retained 22/321 COCOA records are complete, so full native replay cannot yet proceed. The existing collector PID 26540 recovered the new timeout receipt and successfully polled at 12:48:02 UTC; its error log remains empty. It was not replaced.

The same read-only verifier now supports a queued recovery with explicit terminal predecessor and completed controller IDs. Verified using `python output/shape-20261121/monitoring/verify_combined_continuation.py 45666 45667 --after-timeout 45653 --after-controller 45654`. It checks terminal states through sacct, verifies the unchanged execution belongs to 45653, and confirms 45666 remains pending before emitting its receipt. This was monitoring-only code; no production experiment or independent Shape-v2 source/job was changed. Saved Canva still has 24 slides and identical research text at 12:45 UTC; no slide edit occurred.

Maintenance root_1 remains 23 November 13:00 UTC through 25 November 13:00 UTC, and Slurm explicitly prevents the remaining original run from starting. Preserve the queued 24-hour continuation and downstream jobs. Defer the same scheduled task to 25 November 13:05 UTC (20:05 Vietnam time), restoring its 30-minute recurrence at that check if work remains. Full original-study completion remains unachieved.

Evidence: [timeout, source and archive replay](output/shape-20261121/monitoring/20261123T124815Z-combined-continuation.json), [allocation and remaining-work receipt](output/shape-20261121/monitoring/20261123T1248-maintenance-readiness.json), [saved presentation text](output/shape-20261121/monitoring/20261123T124537Z-canva-research.json).
