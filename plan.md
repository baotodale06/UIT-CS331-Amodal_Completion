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

## Current shape-only execution (23 November 2026, 18:06 UTC)

Probability training remains live: observer 3532, Windows WSL launcher 2980,
WSL trainer 502 and four loader workers. Five of 20 epochs are complete;
epoch 6 is running. Selected epoch remains zero. Latest full/hidden IoU is
0.8221570681 / 0.3403761707; the original floors are 0.8113735902 / 0.3703334757.

The separate final-epoch diagnostic is implemented, tested, staged and activated
as hidden observer PID 36128. It is `waiting_parent_processes`; no diagnostic GPU
work has started. Its fixed epoch20 and offsets [0, 0.5, 1, 1.5, 2] were frozen
after observing three early epochs. All 54 parent operational files and 22 staged
training files remain unchanged. Eighteen pinned focused tests passed without
skips; full checks passed 471 tests with 26 optional skips, six Bash regressions
and CPU smoke. Source manifest: 85fe297d9232d46e51e770967bcd1fe6dd48ff8a0953f8d8ffa98218fa2316f7.

Next: let all parent epochs and independent native audit finish. Observer 36128
then runs a three-object preflight, independent replay, full validation diagnostic
and independent native-mask replay/report. Inspect all offsets; no automatic
model promotion or benchmark admission exists. Do not restart either observer.
An independently eligible nonzero model still needs separate schema7/8 Quadro
admission and the complete 2,565-target four-metric audit.

Both Kaggle notebooks remain queued; calibrated jobs 45680-45685 and the other
shape-only chains remain pending under maintenance through 25 November 13:00 UTC.
The all-four-metric 3% benchmark goal remains active and unachieved.
[Diagnostic and activation evidence](output/shape-probability-final-diagnostic-20261123/README.md).

## Current shape-only execution (23 November 2026, 17:44 UTC)

Original local schema6 run is complete and independently audited: epoch zero
won, so no learned checkpoint is eligible. Its launcher, trainer, native process
and collector are terminal. The four-worker probability-loss chain started at
17:26:48 UTC after fresh predecessor replay and explicit Kaggle QUEUED status.
Exact 1,264-mask calibration, independent replay and optimizer preflight passed.
Training started 17:29:23 UTC: observer 3532 / Windows WSL 2980 / WSL trainer 502
are live. Two of 20 epochs are complete; selected epoch remains zero.

Next: complete the remaining 18 probability-loss epochs, then selected native
validation and independent replay.
An eligible nonzero checkpoint still needs Quadro admission, scored smoke and
the complete 2,565-target benchmark. Existing calibrated chain 45680-45685 and
other queued studies remain preserved. Full-benchmark 3% on all four metrics
remains the unchanged, unachieved goal. See
[launch evidence](output/shape-probability-workers4-20261123/launch-milestone.json).

## Current shape-only execution (23 November 2026, 17:10 UTC)

The tested four-worker probability fallback is activated as PID 3532 and has
entered `waiting_predecessor`. Only the unlaunched two-worker observer PID 21512
was retired, with a persistent startup fence. Original local training remains
live at 18/20 epochs complete, selected epoch 0. Both Kaggle notebooks remain
queued, and the calibrated full-benchmark chain 45680-45685 remains pending.

Next: finish original training and exact native replay, then allow the frozen
conditional fallback to run if Kaggle is still queued. Only an independently
eligible nonzero checkpoint justifies schema 7/8 Quadro admission. Preserve all
ongoing jobs and sources; the four-metric 3% goal remains active and unachieved.
[Activation and checks](output/shape-probability-workers4-20261123/milestone.json).

## Current shape-only execution (23 November 2026, 16:47 UTC)

Verified eight live observers, unchanged56 operational/22 staged sources,
local16/20 epochs complete (epoch17 running; selected0), both Kaggle versions
QUEUED with29.918 GPU hours unused and no reserved running quota. Cluster remains
in maintenance until25Nov13:00UTC. Preserve current runs; follow completion,
independent native replay and conditional schema8 fallback. New schema7/8 admission
waits for an independently eligible nonzero checkpoint. Full2565-target calibrated
chain45680-45685 remains queued. The all-four3% goal remains active and unachieved.
[Current evidence](output/shape-monitoring-20261123/20261123T164710Z/receipt.json).

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

## Shape-only continuation: package ready, H100 account pending (23 November 2026)

The user clarified the goal as >=3% relative gain on all four existing full metrics
over original PMC, retaining failure accounting, amodal IoU and unseen reporting.
The user requested finishing the package while H100 account access is pending.
[Package and next actions](output/shape-v2-20261123/README.md) are complete and
validated. Next: authenticate the correct Colab account; verify its highest GPU;
train the frozen v2 configuration; freeze any development changes; assess the
reserved holdout; integrate and GPU-validate an isolated shape-only runtime;
evaluate with exact frozen control references, verify/register and apply the
four-metric gate. No target achievement or model-quality gain has been established.
Do not start or modify combined runs for this task.

## Completed continuation and next research decision (14 November 2026)

The full annotated benchmark, registration, independent full/unseen audit,
metadata recovery and report are complete. Read
docs/SURFACE_ANNOTATED_FULL_RESULTS_20261114.md and
output/surface-annotated-full-review-20261114/REPORT.md. No existing study job
needs resuming or duplication. No new training or tuning was launched.

Full LPIPS improves 1.191% at 4.733x allocation time; frozen unseen LPIPS improves
0.801%. The earlier 3% full target is unmet, and the 204-target primary cohort
remains inconclusive. Keep the annotated control as the operational baseline.

Recommended next research design, not submitted: equal-budget progressive random
selection and correction-only ablation, with independent geometry/blinded human
assessment. Treat completed data as inspected; do not retune the frozen selector
on these results. A future mechanism study requires its own frozen protocol.

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

## Delivery checklist (12 November, 10:20 UTC)

Completed: both 204-target arms, frozen pool diagnostic, independent cluster
audits, native/metadata recovery and measured results note. Primary LPIPS is
inconclusive; retain its interval and all failures. No training or retuning.

In progress: existing workflow 81620 replays locally under pinned WSL, then builds
and validates the report/gallery/PDF/package. Inspect all PDF pages and gallery
filters; preserve a visual-review receipt, compare local and remote audit results,
and package once these final checks pass. Deadline remains 11:46 UTC.

Full control 39488 is running normally. Leave 39490/39491/39492/39497/39495 and
collector 20156 active. Full results and registration remain unfinished, separate
from the completed cohort. Do not start duplicate runs or tune on this cohort.

## Earlier delivery checklist (12 November, 08:28 UTC)

Completed: source/method freezes, source-image checks, GPU runtime and eight-target
smoke, current-runtime 204-target annotated control, identical-reference copy,
historical/current reference audit, mathematical methods appendix and prepared
report/gallery/PDF/package builders. The control's 24-minute run is verified.

Active: Surface-PMC cohort 39487, followed by frozen pool diagnostic 39489 and
cohort/pool exports 39493/39494. Collector 20156 is live. Run independent replay
under `/home/hungvudao/.cache/surface-replay-py310/bin/python` via WSL after native
recovery; then build with Windows Python (report dependencies are installed),
validate numeric values/links/native hashes, inspect PDF and browser gallery,
and package all 204 cases. No quality conclusion from unfinished targets.

Longer continuation: full annotated control 39488, Surface-PMC 39490 and timeout
continuation 39491, full verifier/registrar 39492, independent full/unseen audit
39497 and metadata export 39495. Full control waits for cohort and diagnostic to
prioritize the measured deliverable by 11:46 UTC; retain valid longer jobs.

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
## Shape-Guided PMC · 21 November 2026

The 128-image controlled comparison and both 204-target natural-image evaluations are complete. Jobs 44943 and 44944 audited the cohort predictions against the completed L40S PMC control. The report also compares both methods to historical Quadro Surface on all 204 targets, the 174 with byte-identical RGBA scoring crops and the 166 on which the raw masks, full RGBA images and crops all match. [Week 8 results](output/shape-20261121/REPORT.md)

As of 22 November at 18:00 UTC, Shape is complete, verified, registered and locally reconciled on all 2,565 targets. Continuation 45361, audit 44947 and export 45303 completed. Its 1,393 saved outputs, 1,172 failures, 2,961 candidate pools, four fallbacks, 204 retained cohort rows, 2,516 timeout records and 86 retained metric caches passed reconciliation. Full LPIPS is 0.618641, worse than both historical controls, including the 2,048 identical-reference and 2,167 identical-scoring-crop subsets. All 321 native outlines and 230 eligible hidden-region rows agree with the earlier independent replay. [Local reconciliation](output/shape-20261121/shape-full-local-reconciliation.json)

The remaining GPU work is Shape + Surface. At 23 November 12:48 UTC, continuation 45666 is PENDING for Reserved_for_maintenance, requesting four L40S GPUs, four workers and 24 hours. Its predecessor 45653 timed out normally after 00:30:09 at 12:47:30 UTC. Existing controller 45654 completed in five seconds, preserved 1,128 result records and 87 score caches, archived four partial targets, and automatically submitted 45666 plus controller 45667. Independent replay verifies every retained timeout record/cache, all 204 frozen cohort records, all 12 archived partial directories across three timeouts, Shape's full reference map and all 47 frozen source files. Counts are 619 success and 509 failures; official full scoring/registration remains incomplete, with 1,437 targets left. Only 22/321 COCOA records are ready, so the remaining native full replay also depends on inference. Allocated combined cost so far is 49.368889 GPU-hours including cohort and all three interrupted jobs. Controller 45667 waits afterany:45666, and audit 44949 waits afterok:45666?afterok:45667; export 45304, interval replay 45320 and report 44950 remain queued. Preserve these jobs across maintenance, confirmed 23 November 13:00 UTC through 25 November 13:00 UTC. Defer this same enabled follow-up to 25 November 13:05 UTC (20:05 Vietnam time), then restore 30-minute checks if work remains. Collector PID 26540 is alive and has recovered the final timeout receipt. Reuse registered Quadro controls and preserve independent Shape-v2 work. [Maintenance recovery verification](output/shape-20261121/monitoring/20261123T124815Z-combined-continuation.json)

The supplemental hidden-region IoU comparison covers 230 of the 321 annotated objects, across 220 images. At full closure, compare each new method with both archived controls and reconcile its retained cohort scores with the native 15-object measurement. [Full control scores](output/shape-20261121/full-cocoa-hidden-historical.json) ? [Frozen cohort scores](output/shape-20261121/cohort-cocoa-hidden-iou.json)

The latest saved 24-slide presentation was read on 22 November after its dataset-section replacement. The implemented methods belong on slides 13–15 and 18. Slide 19 needs the full benchmark and slide 20 needs results by occlusion severity. The umbrella and bus examples belong on slide 21, the equal-budget ablation on slide 22, and the limitations and conclusions on slides 23–24. The exported figures retain their original filenames. [Saved presentation](https://www.canva.com/d/j8Vzc325LOrZKtc) · [Current report](output/shape-20261121/REPORT.md)


## Human preference website (22 November 2026)

The frozen first round is full annotated-vocabulary PD-MC versus Surface-aware on all 1,387 shared successful outputs from 2,565 targets. Shape-aware waits for its completed, verified, registered full run; combined Surface + Shape is a separate later round. Do not replace this with the 204-target cohort or pool two-method responses with later three-method responses. The website, offline packet, and response importer are prepared. Hosting state is retained in output/human-preference-20261122/deployment.json. Next research input is the user's own exported ratings; no ratings have been collected. Human preference supplements the defined metrics, with separate coverage and rater disclosure. See docs/HUMAN_PREFERENCE_STUDY.md.


2026-11-22 15:54 UTC. Human preference reviewer publication completed: https://amodal-preference-dv20261122.rompingsalmon2.chatgpt.site
The existing owner-private Site now serves the compact folder-based reviewer from source b0c8bd34ba624002dbe951eb56f693b3ad29cb54 (saved version 2). Open study folder must select output/human-preference-20261122/dist once per hosted session; all images stay local and are checksum-verified. The offline launcher works without folder selection. All 6,935 images are locally validated, and the updated browser test passed the folder-required state, voting, disclosures, ties, skips, backup export/restore, wrong-round rejection and mobile layout with no console exceptions. This supersedes the earlier publication-in-progress entry. The failed full-image remote-build receipt and original archive/source remain preserved in site-tools; deployment.json is the successful final receipt. No publishing, download, or validation work remains pending. The next input is the user's ratings; Shape and combined remain separate future rounds. No human preference scores have been produced.
