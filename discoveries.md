## Fixed final-epoch diagnostic completed: no retained offset (24 November 2026, 12:31 UTC)

Unshifted proper-loss epoch20 improves reported probability scores but fails hidden
geometry. No tested positive offset restores the original hidden floor; larger offsets
eventually reduce both full and hidden IoU. The fixed grid therefore provides no eligible
candidate. This does not establish that every possible threshold or training objective
would fail. The distinct calibrated-initialization experiment remains queued.
All native masks passed independent replay; probabilities were not exported. See
`output/shape-probability-final-recovery-20261124/full-replay.md`.

## Local probability run closed; fixed diagnostic recovered (24 November 2026, 12:24 UTC)

The fixed diagnostic needs both original SAM support and prediction masks. The isolated
training bundle deliberately carries support only; training completion therefore does not
prove it can serve as a native prediction-reference directory. Use the complete checksummed
native export for this diagnostic. The original failure occurred before inference and is
preserved; a separate recovery passed GPU preflight and independent mask replay.

The completed probability-loss experiment has no eligible learned epoch; see the verified
parent report and `output/shape-probability-final-recovery-20261124/parent-closure-milestone.json`.

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

## Probability training tradeoff remains unresolved (23 November 2026, 18:06 UTC)

At epoch5, full/hidden native validation IoU is 0.8221570680505471 /
0.3403761707419017, versus original floors 0.811373590207884 /
0.3703334756537925. Full geometry improves while hidden geometry regresses;
selected epoch remains zero. These are intermediate development observations,
not a completed training conclusion or benchmark gain.

The separate final-epoch diagnostic fixes epoch20 and offsets [0,0.5,1,1.5,2]
from a policy recorded after epoch3. Positive offsets recovering hidden extent
without losing the full-IoU floor remains an untested hypothesis. The diagnostic
is validated and waiting as PID36128; no GPU results exist. Independent native
mask replay is implemented; probability-score replay covers bounds and arithmetic
only because probability pixels are not exported. Validation and earlier benchmark
prefixes have already been observed; any retained candidate is exploratory.
[Protocol and evidence](output/shape-probability-final-diagnostic-20261123/README.md).

## Local training input bottleneck measured (23 November 2026, 17:10 UTC)

On the same 512 sampled objects, data loading averaged 36.247592 seconds with
two workers and 22.965310 seconds with four (order 2/4/4/2), a 1.578363x input
throughput ratio during concurrent training. Every sampled input/target tensor
byte matched. Two actual optimizer updates per worker count over 128 objects
also matched all model/buffer, optimizer, scaler, loss and CPU/CUDA RNG states;
both state fingerprints are
3593efc84695777e2ed1d1cd500a4a126cdc25bf3c7eaeb4b324da0eaa6b420a.
These bounded checks establish neither full-trajectory equivalence nor measured
end-to-end speedup. Evidence: output/shape-loader-profile-20261123/result.json
and optimizer-parity.json.

A separately frozen four-worker package passed focused and full CPU checks and
replaced only the unlaunched fallback observer. It waits for the existing run;
its scientific settings, native geometry and admission gates remain unchanged.
No quality improvement follows from this engineering diagnostic.

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

## Shape-only evidence (2026-11-23 13:04 UTC)

- The fixed 193-target VG prefix has positive point estimates on all four metrics,
  but only feature similarity exceeds 3%. It is nonrandom, partial evidence;
  success/failure counts are identical to the control. See the snapshot-quality report.
- All-parameter SAM adaptation improves hidden native IoU but harms full IoU.
  SAM-support restoration recovers only a small part of the loss; it is insufficient.
- cuDNN deterministic=True and False can change a thresholded native shape mask
  for one object on the same Quadro/checkpoint. Diagnostic 45676 shows exact
  replay for 79 objects with baseline flags; differing-pixel count is unmeasured. Preserve the strict guard and match
  the evaluated arithmetic; do not turn a mismatch into a tolerance silently.
- Constrained decoder-only adaptation is queued, unvalidated on GPU after repair.
  The frozen model remains eligible as epoch zero; no new checkpoint may win with
  lower validation full IoU. Native geometry still cannot prove image-quality gains.

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

## Shape-only continuation findings (23 November 2026)

The v1 report confirms naturally incomplete visible SAM masks were used as
training completion targets; only two of 25 audited training objects were fully
visible and within the image. Its controlled benefit did not transfer to the
full benchmark. A stronger architecture alone is not evidence of improvement.
The v2 package instead uses true COCOA amodal labels and excludes all benchmark
COCO IDs plus content matches across every source. Of 1,992 ID-eligible images,
five matched the benchmark near-duplicate screen. After object filtering there
are 11,941 objects: train 9,494, validation 1,264, reserved holdout 1,183. Actual
image counts are 1,554 / 217 / 195; the original plan's counts precede filtering.
Training data contains no holdout pixels. RGB outside visible support is erased
before resizing, and the crop is determined without hidden labels; regression
tests verify both invariants. Colab H100 requires account entitlement/compute:
the authorized account had zero units and allocation was rejected.
[Evidence and limits](output/shape-v2-20261123/README.md).

## Completed annotated full findings (14 November 2026)

The full matched annotated experiment has LPIPS +1.191% [0.749%, 1.641%], paired
CLIP +1.323%, VGG +2.360% and SSIM +1.384%. The frozen unseen 2,017-target subset
has LPIPS +0.801% [0.396%, 1.179%]. These scopes overlap the inconclusive original
204-target primary cohort; they do not replace its endpoint or provide independent
replications. Full comparison includes earlier training/development images.

Control/Surface save 1,394/1,399 outputs, with 12 gained and seven lost. 1,156 of
the Surface arm's 1,166 failures occur before sampling, so its downstream selector
does not act on these cases. Shared-success LPIPS improves 4.044% but all-target
LPIPS is 1.191%. Nine all-invalid fallbacks occur across 2,847 saved candidate
pools. Allocation time is 4.733x control, including reused cohort and timeout.
These are measured metadata/statistical results, not hidden RGB or geometry truth.
See docs/SURFACE_ANNOTATED_FULL_RESULTS_20261114.md and the full review audit.

## Annotated transfer measured findings (12 November 2026)

The frozen 192-image / 204-target cohort is complete. Failure-inclusive LPIPS
improves 0.250% [-0.269%, +0.810%], so the primary result is inconclusive.
Secondary CLIP improves 1.333% [+0.258%, +2.496%] on 87 shared successes.
VGG and SSIM intervals cross zero. All 117 failures occur before surface sampling;
the method neither gains nor loses a successful target. It generates 1,152
candidates in 144 progressive pools and costs 4.565 times the control allocation
wall time. No new training was performed.

Frozen final raw selection versus uniform raw expectation has no clear LPIPS
benefit (-0.0069%, interval -0.0331% to +0.0196%). At fixed final masks and earlier
trajectories, the final RGB correction improves LPIPS 0.3057% while reducing CLIP
0.4252%. Metric-selected examples show substantial mask changes. These are
conditional diagnostics, not hidden-geometry evidence or equal-budget progressive
causal estimates. See `docs/SURFACE_ANNOTATED_RESULTS_20261112.md`.
The full 2,565-target continuation remains separate and unfinished.

## Historical SAM reference drift (12 November 2026)

Identical annotated run configuration, source RGB, selection class/index and
detection seed reproduced a one-pixel mask difference for the electric-guitar
target free_images/pexels-kelly-1179532-28287998_electricguitar. Historical mask
area 346,523; current 346,522; IoU 0.9999971142. A 5x5 erosion changes five alpha
pixels. The exact numerical cause is unisolated. Preserve strict paired-reference
identity; use the documented fresh-reference amendment and a fresh control,
rather than treating historical outputs as the matched runtime baseline.

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
## Shape versus Surface on the slide 18 cohort · 21 November 2026

Shape improved pre-occlusion mask IoU by 0.0172 versus frozen Surface (paired 95% interval 0.0031 to 0.0316). Object LPIPS also improved, while hidden-core MAE and text cosine favored Surface. An equal-weight rank combination gave hidden IoU 0.5823. The original 128 crops and all eight candidate seeds were shared. [Paired comparisons](output/shape-20261121/controlled-comparisons.csv)


On the frozen 204-target natural Shape cohort, the COCOA dog that Shape saved and L40S PMC missed accounted for 0.02145 of the +0.03291 mean IoU gain across 22 polygon matches. In the original image, the visible SAM reference includes the dog and a doughnut-shaped object beside it. Shape retained both. Of 144,902 predicted foreground pixels inside the annotated frame, 69,370 intersect the 71,501-pixel COCOA dog polygon and 75,532 fall outside. Shape IoU with SAM is 0.97210 and with the dog polygon is 0.47180; the SAM mask itself scores 0.46928 against COCOA. A saved completion can receive a substantial score through coverage while retaining a target-selection error. The other change was the VG banana, which PMC saved and Shape missed. The [coverage figure](output/shape-20261121/natural-cohort-coverage.png) shows all changed saved-output statuses, with original-image coordinates and measured scores.


The full 2,565-target benchmark contains 211 targets whose image identity or source bytes occur in the 183-image Shape training set. We compared all 1,910 source files for the frozen 2,017-target subset (1,900 image IDs) with all 477 prior controlled-study images. There were zero shared identities, file hashes, decoded images or aligned near-copy matches across studies. The nearest aspect-matched 64-by-64 thumbnail had RGB MAE 0.10155 against the historical threshold of 0.005. Within the unseen subset, two distinct LAION IDs share decoded pixels, 000012449 and 000002103. Thirteen additional decoded-file pairs belong to targets with the same declared image ID. The primary analysis retains its frozen IDs and bootstrap clusters. Crops and other transformations can escape this screen. [Image overlap audit](output/shape-20261121/unseen-image-audit.json)


The two controlled K=8 selectors chose the same seed on 23 of 128 images. On the 105 disagreements, the equal-rank combined selector used the Shape seed 34 times, the Surface seed 45 times and another seed 26 times. Thirty-six joint choices used the frozen seed-order tie breaker. Independent replay matched every selected index from the saved scores. Within these 105 disagreements, mean mask IoU was 0.79635 for Shape, 0.77540 for Surface and 0.79280 for combined. Hidden-core RGB MAE was 0.16990, 0.16073 and 0.16292 respectively. The four-candidate weighted scorer from the chat produced mean IoU 0.78246 and object LPIPS 0.15632 across all 128. The group of 105 is descriptive. It was identified from the saved choices after evaluation.


On the 204-target cohort, the Shape and combined first passes had identical eight-candidate pools on every one of the 87 sampled targets. The joint ranker changed the chosen slot on 49 and matched on 38. Both methods stopped before sampling on the other 117. Combined applied frozen Surface RGB correction after selection and used that result in later passes. It recovered the banana and saved the dog, where its mask retained the doughnut-shaped object and missed most of the dog. Its dog polygon IoU was 0.19354 versus 0.47180 for Shape. On all 22 COCOA targets, combined averaged 0.24816 versus Shape 0.26312 (paired difference -0.01496, nominal 95% interval -0.04588 to +0.00196). [Matched natural examples](output/shape-20261121/natural-cohort-coverage.png)

On the frozen 204-target cohort, Shape + Surface reduced failure-inclusive LPIPS by 0.013539 versus Shape. Of that reduction, 0.009233 came from the 86 shared outputs and 0.004306 came from the additional VG/2351866_banana output. The 117 shared failures contributed zero. Using just the 86 shared outputs as the denominator gives mean LPIPS 0.282159 for Shape and 0.260258 for the combined method. Both methods used L40S and identical references. [Slide 15 report](output/shape-20261121/REPORT.md)

On the 128 controlled confirmation images, adding the Shape rank to the frozen Surface rank increased mask IoU by 0.0143 (nominal 95% interval 0.0048 to 0.0242) and hidden IoU by 0.0228 (0.0025 to 0.0440). Object LPIPS decreased by 2.71% (0.12% to 5.19%). CLIP text cosine decreased from 0.2937 to 0.2912. The eight raw candidates were identical for both selectors. These paired estimates describe one-step selection under artificial occlusion. [Slide 18](output/shape-20261121/REPORT.md)

In an exploratory check of the frozen 128-image Shape experiment, the archived COCOA polygons uniquely matched 32 source images. COCOA recorded natural occlusion on 23 of the 32 before artificial occlusion. The pre-occlusion SAM target had mean IoU 0.6826 with the independent full-object polygon after projecting the saved crop into the source frame. On the same 32 images, Shape at K=4 scored 0.6732 amodal IoU versus 0.6328 for uniform selection. At K=8, uniform, Shape, Surface and combined scored 0.6357, 0.6740, 0.6626 and 0.6712. The paired K=8 Shape gain was +0.0384 with nominal 95% interval +0.0154 to +0.0631. The experiment still uses artificially occluded inputs. The 32-image polygon comparison, the 128-image SAM comparison and the natural progressive evaluations have different denominators and targets. [Saved scores and provenance](output/shape-20261121/controlled-cocoa.json)

The Shape source's fully visible starting-object condition can be checked for a small subset of the controlled panel. Of 32 linked COCOA objects, nine have zero original annotated occlusion; seven of these touch the source image border. The remaining bowl and cat have identical visible and amodal annotations and fit inside their experimental crops. At K=8, Shape versus uniform amodal IoU was 0.4500 versus 0.5833 for the bowl and 0.6585 versus 0.6395 for the cat. These two cases constrain interpretation of the broader SAM result. [Shape report](output/shape-20261121/REPORT.md)

On the 128 controlled images, ranking candidates by IoU against the learned amodal prior improved hidden-region mask IoU over ranking by visible-mask IoU alone. Both selectors used the same candidates, nonempty-mask rule and seed order. At K=8, 89 selections changed and hidden IoU increased from 0.3593 to 0.5818 (paired gain 0.2225, nominal 95% interval 0.1739 to 0.2727). On the 32 matched independent COCOA outlines, IoU gained 0.0744 (0.0412 to 0.1101). At K=4, hidden IoU gained 0.1544 and the object LPIPS interval included zero. This post hoc contrast isolates the contribution of the hidden-mask prediction during one-step selection on artificially occluded inputs. [Slide 18 evidence](output/shape-20261121/controlled-visible-ablation.json)

The two existing 204-target PMC runs saved outputs for the same 86 targets, with one additional saved dog on Quadro. Among the 86 shared successes, 71 used the same number of PMC iterations. L40S used more iterations for seven targets and fewer for eight. Across all 204 targets, the records contain 135 iterations on Quadro and 145 on L40S. The cross-device allocation times therefore include different amounts of iterative work. For the dog in COCO train image 108113, both runs selected the same box and recorded detector score 0.36. The initial selected masks covered 141,755 and 141,754 pixels, respectively. After the first completion, the L40S segmenter returned an empty mask array. The traces locate the missed completion but leave its cause undetermined. [Quadro scores](output/surface-annotated-20261112-freshrefs/baseline-cohort/per_target.csv) and [L40S scores](output/shape-20261121/progressive-l40s/baseline-cohort/per_target.csv)

On the 204 natural targets, historical Surface and Shape + Surface saved exactly the same 87 target IDs and both failed on the remaining 117. Their cross-device mean LPIPS on those 87 outputs was 0.258011 for Surface and 0.258663 for the combined method. With the fixed penalties over all 204 targets, the LPIPS difference was 0.000278 in favor of Surface. The paired 95% interval for the relative combined gain was -0.300% to +0.201%. Comparing Shape alone to Surface, 86 targets had outputs from both and only Surface saved VG/2351866_banana. The shared outputs accounted for 0.009505 of the 0.013817 mean LPIPS difference, and the banana accounted for 0.004312. [Surface score table](output/surface-annotated-20261112-freshrefs/surface-cohort/per_target.csv) and [combined comparison](output/shape-20261121/combined-cohort-vs-pmc-summary.json)

On the frozen 204-target cohort, historical Quadro Surface and L40S Shape both sampled a first candidate pool on 87 targets. The remaining 117 stopped earlier in both runs. All 87 had identical first-pass crop coordinates and seeds. The first-pass visible mask also matched on 65. Of those 65, 59 had identical scoring-reference hashes. On these 59, all 472 corresponding candidate images differed before selection. The mean absolute RGB difference was 0.584 channel levels on the 0-255 scale, and four candidate masks matched pixel for pixel. These are measurements of two complete runtime configurations. The hardware contribution alone is unidentified. [First-pass evidence](output/shape-20261121/first-pass-cross-device-cohort.json)

With the saved observation, visible mask, occlusion mask and trained checkpoint fixed, a CPU replay reproduced all 436 binary Shape priors from the L40S controlled and cohort runs pixel for pixel. All 692 checked candidate selections and fallback flags also matched when scored on the saved candidate masks and Surface losses. This tests the U-Net prediction and selection stages on those inputs. Diffusion generation and segmentation precede this check. Computation and denominators are recorded under Slurm job 45298 in [session.md](session.md).

The released natural-image evaluator reads the eroded RGBA crop for LPIPS. The 166-target cross-device comparison required three files to match, including the raw SAM mask. On eight additional cohort targets erosion produced exactly the same scoring crop despite a change in the raw mask. One was COCOA dog 108113. Across all 174 targets with matching crops, the relative LPIPS gain of L40S PMC over Quadro PMC was -0.622% (nominal 95% interval -2.084% to +0.211%). The cause of the different completion status for the dog remains unknown. [Paired comparisons](output/shape-20261121/REPORT.md#cross-device-comparison-with-identical-lpips-scoring-crops)

Across 22 polygon-annotated cohort targets, the same seven completions were saved by Shape and Shape + Surface. Their mean amodal IoU difference (combined minus Shape) was -0.01496, with a nominal 95% interval of -0.04588 to +0.00196 from 20 image clusters. The drop for the dog in COCO train image 108113 contributed -0.01265 to that mean difference. Three target scores rose and four fell. All four methods' scores were independently recovered from native completion alpha in original image coordinates. [Per-target checks](output/shape-20261121/cohort-cocoa-native-replay.json)

On the 25 Shape training objects with an independent COCOA outline for the same target ID, the unaugmented 256 by 256 SAM training labels had mean original-frame IoU 0.89649 with the visible outlines and 0.78856 with the amodal outlines. Nineteen objects were already partly occluded before the synthetic mask. Four of the remaining six touched the source image border. Only a car and a cat passed the full visibility and framing screen. All 183 original images, SAM masks, archived labels and mask crops were verified. The held-out polygon evaluation used 32 different objects, with a bowl and a cat passing the same screen. [Training labels and per-target pixel counts](output/shape-20261121/training-mask-audit.json)

Visual Genome inference finished for all 1,294 official targets from 1,234 images in the full Shape run. Shape on L40S saved 724 completions (55.95%). The registered Quadro PMC and Surface controls saved 726 (56.11%) and 729 (56.34%). Shape and PMC both saved 720 targets, with four unique to Shape and six unique to PMC. Shape and Surface both saved 722, with two unique to Shape and seven unique to Surface. This source-wide coverage comparison uses the frozen failure policy and includes the 109 retained VG targets. Full appearance scores and Shape + Surface on VG remain pending. [Per-target VG coverage and record hashes](output/shape-20261121/progressive-l40s/shape-vg-inference-coverage.json)

On the 1,294 Visual Genome targets from 1,234 images, mean Shape SSIM was 0.460034 on L40S. Using the archived Quadro scores, the relative difference was -0.75% versus PMC (nominal 95% interval -1.69% to +0.16%) and -2.44% versus Surface (-3.34% to -1.63%). Both intervals use 5,000 paired image resamples with seed 20261110. The estimates include zero SSIM for 570 failed Shape completions and describe this source before full registration. Device, sampling wrapper and reference differences limit attribution. [Coverage](output/shape-20261121/progressive-l40s/shape-vg-inference-coverage.json) / [SSIM replay](output/shape-20261121/progressive-l40s/shape-vg-ssim.json)

For the four-candidate experiment in the chat proposal, the 32 controlled COCOA images with independent outlines also have known RGB under artificial occlusion. A paired comparison on exactly those images gives mean amodal IoU 0.6328 for uniform and 0.6732 for Shape (gain 0.0405, nominal 95% interval 0.0203 to 0.0640). Hidden-region LPIPS decreases from 0.4329 to 0.4186 (reduction 0.0143, interval -0.0007 to +0.0302). Hidden-core DreamSim decreases from 0.2224 to 0.2040 (reduction 0.0184, interval +0.0062 to +0.0313). The LPIPS interval includes zero. This post hoc subset has 23 already partly occluded objects and two passing the strict fully visible and in-frame screen. The RGB reference covers pixels visible in the original photograph that were subsequently hidden artificially. [COCOA comparison](output/shape-20261121/REPORT.md#controlled-images-with-cocoa-amodal-outlines)

On Visual Genome, lower completion coverage explains part of the Shape SSIM gap with the registered Surface control. On the 722 targets where both saved a completion, mean SSIM was 0.82236 for Shape and 0.83675 for Surface. Those paired differences contributed −0.008031 to the all-target difference of −0.011514. The seven Surface-only and two Shape-only completions accounted for the remaining −0.003483 across all 1,294 targets. Against PMC, the 720 common completions contributed −0.002068 and the ten one-sided completions contributed −0.001401, totaling −0.003469. Each contribution divides the sum of paired score differences by 1,294, with zero SSIM for a failure. The figures describe the existing L40S and Quadro runs with their respective scoring references. [Shape SSIM](output/shape-20261121/progressive-l40s/shape-vg-ssim.json) · [Paired completion status](output/shape-20261121/progressive-l40s/shape-vg-inference-coverage.json) · [Surface scores](output/surface-annotated-20261112-freshrefs/surface/per_target.csv) · [PMC scores](output/surface-annotated-20261112-freshrefs/baseline/per_target.csv)

### Shape on the four most occluded natural COCOA targets

On the plate in COCO train image 060172 (82.67% annotated occlusion), Shape's saved mask overlapped 55,597 plate pixels in a union of 269,810, giving amodal IoU 0.2061. Historical PMC scored 0.9809 and Surface scored 0.1641 on the same annotation. Of Shape's 55,653 predicted foreground pixels in the annotated image frame, 55,597 (99.90%) lay inside the plate. The missing area accounts for most of the error. Of the 222,997 annotated plate pixels outside the visible mask, Shape covered 16,673 (7.48%), PMC covered 220,852 (99.04%) and Surface covered 5,137 (2.30%). Shape ran on L40S and the two controls on Quadro. [COCOA mask](output/shape-20261121/evaluation_only/cocoa/COCO_train2014_000000060172_plate.png) · [Shape RGBA](output/shape-20261121/progressive-l40s/shape/targets/COCOA/COCO_train2014_000000060172_plate/completion.png) · [PMC RGBA](output/surface-annotated-20261112-freshrefs/baseline/targets/COCOA/COCO_train2014_000000060172_plate/completion.png) · [Surface RGBA](output/surface-annotated-20261112-freshrefs/surface/targets/COCOA/COCO_train2014_000000060172_plate/completion.png)

At 05:20 UTC, Shape had also closed the table in train image 013720 and plate in train image 024086 without a saved completion. Each receives zero IoU. The elephant in test image 073268 was outstanding. Given these three completed records, even IoU 1 on the elephant would put the four-target Shape mean at 0.3015, below the registered PMC mean of 0.4455. The Shape mean can range from 0.0515 to 0.3015, which still spans the historical Surface mean of 0.2380. This is a bound on four specific annotated objects pending closure of the full run. [Shape table record](output/shape-20261121/progressive-l40s/shape/targets/COCOA/COCO_train2014_000000013720_table/result.json) · [Shape unsaved plate record](output/shape-20261121/progressive-l40s/shape/targets/COCOA/COCO_train2014_000000024086_plate/result.json) · [Historical target scores](output/shape-20261121/cocoa-historical-controls.json)

### Shape appearance on all Visual Genome targets

The official released scorer evaluated 724 saved Shape predictions on CPU. The other 570 of 1,294 targets received LPIPS 1. Mean LPIPS was 0.598369 for Shape, 0.593162 for historical PMC and 0.583915 for historical Surface. With 5,000 bootstrap draws over the 1,234 image IDs, Shape's LPIPS was 0.878% higher than PMC's (95% interval 0.204% to 1.599%) and 2.475% higher than Surface's (1.829% to 3.214%). On the 722 targets where both Shape and Surface saved a result, their means were 0.281803 and 0.261817. Those common outputs contributed 0.011151 to the overall mean gap of 0.014454. On the 720 targets saved by both Shape and PMC, Shape scored 0.281721 and PMC scored 0.275143. The matched-output comparisons are conditional on both methods succeeding. The Shape predictions used L40S and the registered controls used Quadro, with differences in runtime and reference generation. This source-specific endpoint measures visible-reference appearance. The independent COCOA geometry endpoint is reported separately. [Per-target replay](output/shape-20261121/progressive-l40s/shape-vg-lpips.json)

The appearance difference also persisted when both methods saved the target and used a byte-identical reference crop. Of the 722 VG targets shared by Shape and Surface, 568 met the crop condition. Mean LPIPS was 0.270838 for Shape and 0.251364 for Surface. On 566 of the 720 outputs shared by Shape and PMC, the corresponding means were 0.270695 and 0.262092. These are exploratory subsets selected on joint success. The models still ran with different GPU and software configurations. Crop identities came from the released SSIM audit and archived per-target execution records. [Shape crop hashes](output/shape-20261121/progressive-l40s/shape-vg-ssim.json) ? [Shape LPIPS](output/shape-20261121/progressive-l40s/shape-vg-lpips.json) ? [Surface scores](output/surface-annotated-20261112-freshrefs/surface/per_target.csv) ? [PMC scores](output/surface-annotated-20261112-freshrefs/baseline/per_target.csv)

### First-pass Shape selection on the 82.67%-occluded plate

The original plate has 222,997 independently annotated hidden pixels. In the 879 by 879 diffusion crop, the union of the stored visible and occlusion masks projected back to only 9,729 of those pixels (4.36%). The U-Net prior covered 5,247 (2.35%). Shape ranked candidate 7 first because its mask had IoU 0.64368 with that prior. Its IoU with the COCOA amodal outline was 0.16213, and it covered 6,086 hidden pixels (2.73%). Candidate 1 in the very same pool had prior IoU 0.24054 but the best amodal IoU at 0.60674. It covered 124,002 hidden pixels (55.61%). Candidates are numbered from 1 in the report. This shows a selection error against the independent outline on the recorded first pass. The saved experiment followed candidate 7, so the final result from following candidate 1 is unknown. The selected trajectory ran six passes and finished with amodal IoU 0.2061. [First-pass figure](output/shape-20261121/shape-plate-first-pass.png) ? [Per-pass measurements](output/shape-20261121/shape-plate-trace.json)

### Shape on previously unseen Visual Genome images

The frozen unseen inventory contains 1,094 Visual Genome targets from 1,056 images. Shape saved 581 outputs, historical PMC saved 584, and historical Surface saved 585. On all 1,094 targets, including LPIPS 1 and SSIM 0 for failures, the mean scores were:

| Method | LPIPS ↓ | SSIM ↑ |
|---|---:|---:|
| PMC · Quadro RTX 6000 | 0.613175 | 0.441877 |
| Surface · Quadro RTX 6000 | 0.605121 | 0.448567 |
| Shape · L40S | 0.618978 | 0.437673 |

Shape's LPIPS was 0.95% higher than PMC's (nominal 95% interval 0.29% to 1.66%) and 2.29% higher than Surface's (1.59% to 3.05%). Both gaps persist after excluding images from training and earlier controlled experiments. The subset uses one benchmark source. CPU measurement of saved L40S predictions and the registered Quadro controls retains the hardware, software and reference differences of the original comparison. [Unseen inventory](output/surface-annotated-20261112-design/unseen-inventory.json) · [Image-overlap audit](output/shape-20261121/unseen-image-audit.json) · [Shape LPIPS](output/shape-20261121/progressive-l40s/shape-vg-lpips.json) · [Shape SSIM](output/shape-20261121/progressive-l40s/shape-vg-ssim.json) · [PMC scores](output/surface-annotated-20261112-freshrefs/baseline/per_target.csv) · [Surface scores](output/surface-annotated-20261112-freshrefs/surface/per_target.csv)

### Occlusion and hidden appearance on the two initially fully visible COCOA objects

The four-candidate test in the saved Shape proposal asks about originally complete objects under heavy occlusion. Two of the 32 independently annotated test objects met the original visibility, framing and crop criteria. On the model's input grid, the artificial occluder covered 19,828 of 81,168 annotated bowl pixels (24.43%) and 23,784 of 68,088 annotated cat pixels (34.93%). Both lie in the original paper's 20–50% easy range. The controlled panel therefore provides zero independently confirmed, initially fully visible objects in the paper's 50–80% difficult range. The comparison below is descriptive. Each uniform value is the mean over all four candidates for that object.

| Object | Shape seed | Amodal IoU, uniform → Shape | Hidden spatial LPIPS, uniform → Shape | Hidden-core DreamSim, uniform → Shape |
|---|---:|---:|---:|---:|
| Bowl 020601 | 33 | 0.5767 → 0.5669 | 0.2734 → 0.3628 | 0.1486 → 0.1588 |
| Cat 248755 | 22 | 0.6450 → 0.6585 | 0.4208 → 0.4269 | 0.2844 → 0.2779 |

IoU is higher-is-better. Both appearance distances are lower-is-better. Shape lost on all three measures for the bowl. For the cat, its mask and hidden-core DreamSim improved while hidden spatial LPIPS increased. The area measurements use the independent COCOA amodal outline resized into the input crop. The appearance metrics use the original study's artificially hidden RGB regions. PIL binary intersections and NumPy masking gave the same pixel counts. The candidate indices and score files matched their recorded hashes. [COCOA outlines and candidate IoUs](output/shape-20261121/controlled-cocoa.json) · [Bowl scores](output/shape-20261121/controlled/COCOA-026-b9c73075a3/result.json) · [Cat scores](output/shape-20261121/controlled/COCOA-049-1a7b92033f/result.json) · [Hidden-core DreamSim](output/shape-20261121/hidden-dreamsim.json) · [Input crops and masks](output/surface-confirmation128-20261110/cases.json)

Among all 32 controlled COCOA objects, nine had 50–80% of the annotated object hidden inside the model input crop after adding the artificial occluder. Original-frame mean amodal IoU over the same first four candidates was 0.568201 under uniform selection and 0.619197 for Shape. Shape scored above the uniform expectation on six of nine. Eight were already partly hidden in the original image and the remaining bed touched the image border. This descriptive subset combines natural and artificial occlusion. The two initially complete, fully framed objects are both outside it. [Per-image original-frame candidate scores](output/shape-20261121/controlled-cocoa.json) · [Occluder and crop inputs](output/surface-confirmation128-20261110/cases.json)

### Hidden-region geometry on the frozen natural cohort

Fifteen of the 22 independently outlined COCOA targets had at least one annotated hidden pixel, across 13 images. We removed the COCOA visible mask from both the native prediction and the COCOA amodal mask before computing IoU in the original image frame. Failed completions scored zero. Seven targets had no annotated hidden pixels and were excluded from this supplemental measure. Mean hidden-region IoU was 0.181344 for PMC on L40S, 0.224573 for Shape on L40S, 0.219693 for Shape + Surface on L40S, and 0.210367 for historical Surface on Quadro. Shape gained 0.043229 over PMC (nominal image-paired 95% interval 0.006417 to 0.092775). Its gain of 0.014207 over Surface had an interval from −0.011434 to 0.060190. All six shared PMC and Shape successes together contributed 0.024621 to the mean gain over 15 targets. The extra dog completion contributed the other 0.018608. Shape's mask for that dog also covers a neighboring object. This measure addresses hidden silhouette overlap on a small cohort. The natural appearance scores remain separate. [Per-object intersections, unions and comparisons](output/shape-20261121/cohort-cocoa-hidden-iou.json) · [Native masks and provenance](output/shape-20261121/cohort-cocoa-native-replay.json)

### Full COCOA inventory for hidden-region IoU

Of the 321 matched COCOA objects, 230 from 220 images contain at least one annotated hidden pixel in the original image frame. The other 91 have empty hidden annotations. The frozen unseen split contains 153 eligible objects from 145 images. On all 230, historical Quadro PMC scored 0.259553 mean hidden-region IoU (138 saved) and historical Surface scored 0.266282 (137 saved). Surface minus PMC was +0.006729, nominal paired 95% interval −0.016622 to +0.028729. The corresponding unseen means were 0.263597 and 0.266289, with a difference of +0.002692 (−0.024453 to +0.030204). All failures contribute zero in the eligible set. The independent all-object result covers 321 annotations, and the hidden-region result covers the 230 with a nonempty hidden region. The pending Shape comparisons use the same targets and archived controls. [Native intersections and unions](output/shape-20261121/full-cocoa-hidden-historical.json)


## Full-run human preference protocol (22 November 2026)

The first round uses all shared successful outputs of annotated PD-MC (1,394 saved) and Surface-aware (1,399 saved): 1,387 cases from the full 2,565-target population, with no metric-based case selection. Exclusions are 1,159 failures in both, seven baseline-only successes, and twelve Surface-only successes. Native images share original-coordinate bounds including negative offsets and out-of-frame support. Public files omit case-to-method mappings, target identities, metrics, and coordinator data. A/B positions balance 694/693. Ratings preserve ties, skips, unassessable cases, drafts, and missing optional judgements. Tied-top votes split equally; nonwinning methods in a later three-way round are not inferred to tie. The user is a researcher/rater and must disclose involvement and exposure; this is not independent consensus evidence. Exported JSON feeds a separate report supplement, not a composite metric. Protocol: docs/HUMAN_PREFERENCE_STUDY.md.

Sites transport required bounded Git pushes after HTTP 413; all 6,944 source files were subsequently verified against one final commit by the official helper. A local replica independently matched every Git blob, and the official packager completed on both hosts. Native archive upload has a 512 MiB cap, so this full native-image study uses remote build from the exact pushed commit. All tokens stayed in memory/hidden stdin. The private audience and existing Site identity were preserved.


2026-11-22 15:54 UTC. Human preference reviewer publication completed: https://amodal-preference-dv20261122.rompingsalmon2.chatgpt.site
The existing owner-private Site now serves the compact folder-based reviewer from source b0c8bd34ba624002dbe951eb56f693b3ad29cb54 (saved version 2). Open study folder must select output/human-preference-20261122/dist once per hosted session; all images stay local and are checksum-verified. The offline launcher works without folder selection. All 6,935 images are locally validated, and the updated browser test passed the folder-required state, voting, disclosures, ties, skips, backup export/restore, wrong-round rejection and mobile layout with no console exceptions. This supersedes the earlier publication-in-progress entry. The failed full-image remote-build receipt and original archive/source remain preserved in site-tools; deployment.json is the successful final receipt. No publishing, download, or validation work remains pending. The next input is the user's ratings; Shape and combined remain separate future rounds. No human preference scores have been produced.


## Shape full COCOA outline preverification, 22 November 2026

At 16:16 UTC, all 321 polygon-target records were available in the immutable timeout snapshot from job 44946. Recovered and checksum-verified all records, 181 native RGBA completions and 181 binary masks. Explicit clipped NumPy pixel coordinates and PIL projection agreed for all saved completions. The production geometry routine also matched the independently projected alpha and checked mask/alpha equality. Compared these targets with the existing Quadro controls using the frozen 5,000-draw image bootstrap.

Shape scored 0.454330 mean full-outline IoU, compared with 0.462682 PMC and 0.464422 Surface. Shape minus PMC was -0.008352 [-0.027121, +0.008950], and Shape minus Surface was -0.010092 [-0.028478, +0.006657]. On 230 objects from 220 images with annotated hidden pixels, Shape scored 0.276619. Its differences were +0.017066 [-0.007270, +0.040970] versus PMC and +0.010337 [-0.015240, +0.035974] versus Surface. The unseen comparisons cover 211 full outlines and 153 hidden-region objects, and all four intervals also include zero.

The formerly missing elephant 073268 scored 0.791911 IoU. The four 80-100% occlusion cases now give Shape mean 0.249493, including two failed targets with zero IoU. These are native geometry measurements on completed records. Official 2,565-target scoring, registration and full selector audits are pending. The final audit must reconcile these 321 rows and paired estimates. Evidence is in [shape-cocoa-preverification.json](output/shape-20261121/shape-cocoa-preverification.json), SHA-256 9da6abc07c70b2c8648b15d8f3969ddd9048c507e9e7514481a2c90759490ffe.


## Registered full Shape result and local reconciliation, 22 November 2026

The full Shape run saves 1,393 of 2,565 targets and retains 1,172 failures. Failure-inclusive LPIPS is 0.618641, compared with 0.614408 historical PMC and 0.607088 historical Surface. Shape LPIPS is 0.689% higher than PMC (nominal paired 95% interval 0.118% to 1.259%) and 1.903% higher than Surface (1.395% to 2.449%). On 2,017 unseen targets from 1,900 images, deterioration is 0.935% and 1.750%, respectively, also excluding zero.

The final native-reference audit finds 2,048 targets from 1,918 images with all three reference files identical, and 2,167 targets from 2,020 images with identical scoring crops. On identical full reference sets, Shape LPIPS is 0.753% higher than PMC (0.240% to 1.249%) and 1.589% higher than Surface (1.162% to 2.032%). On identical scoring crops, the respective increases are 0.799% (0.283% to 1.321%) and 1.746% (1.303% to 2.208%). Local bootstrap replay reproduces all four estimates. All 1,136 early native-reference classifications agree. Reference drift therefore does not explain away this observed deterioration. Hardware and sampling-wrapper differences remain confounded; this does not isolate a GPU effect. No duplicate control run is justified by this audit.

The final 321 full-outline scores and 230 eligible hidden-region scores exactly match the earlier independent native replay and all eight full/unseen geometry estimates. Mean Shape IoU remains 0.454330 (full outline) and 0.276619 (hidden region); intervals versus both historical controls include zero. The earlier small-cohort improvement does not establish a full-benchmark benefit. The five severity bins contain 203/54/39/21/4 targets. VG CPU versus registered scores differ by at most 1.7881393432617188e-07 LPIPS and zero SSIM. Shape's actual allocation cost is 109.571 GPU-hours across 44914, 44946 and 45361.

[Local reconciliation and input digests](output/shape-20261121/shape-full-local-reconciliation.json) ? [Native outline replay](output/shape-20261121/shape-full-cocoa-native-replay.json) ? [Metadata replay](output/shape-20261121/shape-full-metadata-replay.json)


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



## Original combined timeout preservation (23 November 2026, 12:21 UTC)

Both confirmed timeout inventories survive continuation 45653 unchanged: 700 records from 44948, 1,085 from 45377, 87 cached scores and all 204 frozen cohort records. Independent replay also matched every archived partial-target directory: seven empty directories and all 22 files for VG/2350254_bus. The new execution retains Shape's full 2,565-target reference map and the 47-file frozen source manifest. Its 1,085-record prefix is not a full efficacy result; inference and official scoring remain incomplete. Terminal combined allocations total 47.358889 GPU-hours before the ongoing 45653 allocation. Preserve this interrupted work in final cost. [Independent recovery and native archive checks](output/shape-20261121/monitoring/20261123T122132Z-combined-continuation.json)


## Original combined maintenance boundary (23 November 2026, 12:48 UTC)

Three timeout snapshots preserve 700, 1,085 and 1,128 completed records, 87 cached scores and all 204 frozen cohort records. Native rehashing verified all 12 archived partial-target directories, including the newly archived 220 files from three unfinished targets and one empty fourth target. Combined has 619 successes and 509 failures so far; neither full efficacy nor native 321-target geometry can be concluded. Only 22/321 COCOA records are complete. Allocated cost is 49.368889 GPU-hours including the retained cohort and all three interrupted jobs. Continuation 45666 and controller 45667 are queued across confirmed maintenance. [Independent terminal-state and preservation evidence](output/shape-20261121/monitoring/20261123T124815Z-combined-continuation.json)

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
