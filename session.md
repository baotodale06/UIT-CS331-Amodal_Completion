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

## Fixed final-epoch diagnostic activated (23 November 2026, 18:06 UTC)

Completed the previously unfinished diagnostic without changing the fixed
policy or the running parent. Added an independent exported-mask verifier,
bounded probability-score arithmetic replay, report, 13 negative/contract/native
replay tests, isolated packaging, and a guarded post-parent runner. All 18 focused
tests passed in the pinned environment without GPU use or skips. Full CPU checks
passed 471 tests (26 optional-dependency skips), all six Bash regressions and
smoke. No GPU diagnostic result or benchmark admission exists yet.

The 16-file archive SHA256 is
c5bf9595678757c0ac8d005e321eb2ffcdb99250ea652c2c8b17aca0f71a045b;
source manifest is 85fe297d9232d46e51e770967bcd1fe6dd48ff8a0953f8d8ffa98218fa2316f7.
Staged at /home/hungvudao/.cache/shape-local-gpu-20261123/probability-final-diagnostic-source.
All 54 parent operational files and 22 staged parent files were verified intact.
Policy SHA256 remains f28370d1bd703b147a83c0ab710cdd43b062c1ea2ae626f1195201c59388fe29.

Activated hidden observer 36128 at 18:05:36 UTC; direct process inspection and
milestone.json confirm it is alive, stderr is empty, and it is waiting for actual
parent process termination. No launch-request/execution exists for the diagnostic.
Parent observer 3532 / Windows WSL 2980 / WSL trainer 502 remain live, with four
loader workers. Five of 20 epochs are complete, epoch6 running, selected0.
Latest full/hidden IoU: 0.8221570680505471 / 0.3403761707419017. Neither floor policy
nor checkpoint selection changed. Activation-training.json is a fresh observation,
distinct from the original three-epoch policy-freeze snapshot.

After all seven parent phases, verified native replay/report and actual process
exit, observer36128 will run the three-object GPU preflight, its independent
replay, the full fixed diagnostic and independent full native-mask replay/report.
Every original mask must match Quadro, every zero-offset final mask must repeat
against a separate pass, and unshifted final aggregate geometry must equal epoch20.
All five offsets are reported; no automatic promotion. Probability pixels are
not exported, so their independent audit covers bounds and aggregate arithmetic.

Kaggle probability version1 and constrained version2 were freshly QUEUED;
Slurm 45673-45675,45677-45685 remain PENDING. UTC reservation end is confirmed
25 November13:00UTC. No jobs or existing collectors were restarted or changed.
Full-benchmark >=3% on all four metrics is still unachieved. See
[diagnostic evidence](output/shape-probability-final-diagnostic-20261123/README.md).

## Local run closed; probability training live (23 November 2026, 17:44 UTC)

Original schema6 training and all seven execution phases completed successfully.
All 20 epochs failed to produce an eligible learned checkpoint; epoch zero is
selected. Independent replay covers all 1,264 masks, selection, native geometry
and image-bootstrap statistics. Selected masks exactly reproduce the original
Quadro predictions. Recorded training took 186.52 minutes. Completed report:
[local result](output/shape-local-20261123/report/REPORT.md). Launcher 328,
trainer 495, native process 11869 and collector 2332 are terminal; do not restart.
Training SHA256: 6c02f0621ee63a331385bdb93e60da44c950fec49e0b73ded7175ed2173fe6f6.
Replay SHA256: 798ae4d76a35badd1040abbfbe5cb2d952c0a21db5a42ae78f2e0cbedffca793.

Observer PID 3532 freshly repeated the predecessor audit, confirmed Kaggle
probability version 1 QUEUED, and started the four-worker execution at 17:26:48
UTC. Initial calibration reproduced all 1,264 original native masks; independent
calibration replay and real optimizer preflight passed. Training started at
17:29:23 UTC and is live under Windows WSL PID 2980 / WSL trainer PID 502.
It has completed 2/20 epochs, selected epoch zero, with epoch 3 running.
Neither learned epoch passed both geometry floors. Calibration handles 28772/424
are terminal. Only an eligible nonzero checkpoint can proceed toward admission.
Actual state: output/shape-probability-workers4-20261123/execution.json,
queue-status.json and launch-milestone.json. The 54-file operational hash remains
9a531b2d0b9ddc938518db4a0ac0ff3cffc694f814eb8fc030bea22a9cfb56fe.

Other observers and queued studies remain intact. Fresh 17:14 Kaggle quota
audit showed 29.918 hours unused, no running quota reservation and no failure
messages. SSH confirms the existing GPU roots are blocked by maintenance,
ending 25 November 13:00 UTC / 20:00 Vietnam. No new benchmark result exists.
The all-four full-benchmark 3% goal remains ACTIVE and unachieved.

## Four-worker probability fallback activated (23 November 2026, 17:10 UTC)

Replacement observer PID 3532 is live and now reports `waiting_predecessor`.
Original observer PID 21512 was deliberately retired before either probability
GPU phase started; its exit-1 notification is expected. Its explicit startup
fence and supersession receipt preserve the original experiment and sources.
WSL launcher 328 and trainer 495 remain live: 18/20 epochs complete, epoch 19
progressing, selected epoch 0. No eligible new checkpoint exists yet.

The separate four-worker package passed 458 CPU tests (26 optional skips), six
Bash regressions and CPU smoke; all 19 focused pinned-environment tests passed
without skips. The 54 operational files and 22 staged source files pass hashes;
all original two-worker operations remain byte-identical. See
[activation evidence](output/shape-probability-workers4-20261123/milestone.json).
Archive SHA256: fe6b8235e4dd4e5908e3452558f4f281d0f18c0fbc392aeb1543f8840a2268f3.
Operational SHA256: 9a531b2d0b9ddc938518db4a0ac0ff3cffc694f814eb8fc030bea22a9cfb56fe.

Only the loader worker count changes from two to four. Sampled tensor bytes and
two actual optimizer updates matched; input throughput improved 1.578x in the
512-object diagnostic, with no measured full-training speedup or quality gain.
The original local run must complete all phases and independent replay before
the replacement checks fresh Kaggle probability version 1. Only QUEUED starts
the local fallback; RUNNING/COMPLETE skips it. Initial calibration still requires
all 1,264 original Quadro masks to match, and epoch 0 cannot be promoted.

Both private Kaggle notebooks remain QUEUED at 17:08:58 UTC. Cluster shape jobs
45673-45685 remain pending, with the calibrated chain and all collectors live.
No benchmark admission or new benchmark target exists. Goal remains ACTIVE and
unachieved; maintenance ends 25 November 13:00 UTC / 20:00 Vietnam.

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

## Shape-only Colab package prepared (23 November 2026)

The active objective remains unmet. The user explicitly requires >=3% relative
improvement on **all four** full-benchmark metrics (CLIP, LPIPS, VGG feature
similarity, SSIM), not only LPIPS. Train only in Google Colab on the highest
available GPU; no combined Shape + Surface work belongs to this continuation.
Latest steering: "Prepare the package first. The account with h100 access is pending."

Delivered and independently validated
[the handoff](output/shape-v2-20261123/README.md), notebook, source ZIP and 297 MB
training-data ZIP. True COCOA amodal supervision: 9,494 training objects / 1,554
images, 1,264 validation objects / 217 images; 1,183 holdout objects / 195 images
are physically excluded from the training ZIP. Excluded 508 benchmark COCO IDs
and five additional benchmark near-duplicate images. All-source byte/decoded/
aligned-thumbnail screening is recorded; transformed duplicates are not ruled out.

The new ResNet34 U-Net, masked input pipeline, soft-IoU scoring, resumable trainer,
and pinned original-control four-metric gate are in separate v2 files. The v1
runtime cannot load the new architecture unchanged; isolated runtime integration,
real GPU checks, training and benchmark evaluation remain pending. Do not claim
improvement from package readiness. Freeze any validation-driven policy changes
before opening the reserved holdout or evaluating the benchmark.

Official Colab CLI 0.8.0 is installed in WSL at
`~/.cache/shape-colab-py314/bin/colab`. Authorization succeeded, but `colab --auth
adc usage` reported 0 compute units; `new -s shape-v2-20261123 --gpu H100` was
rejected for quota/entitlement. No runtime was allocated. Task credentials are
outside the repo; never package them. The correct H100 account needs a new login
through `scripts/colab_shape_auth.py` and a secure authorization-code prompt.

Validation: 333 repository tests (one optional torch skip), five Bash regression
scripts and CPU smoke passed with exit 0; nine focused tests including a real
CPU model forward pass passed without skips. Notebook schema, code-cell syntax,
all 12,529 archived data hashes and packaged-source hashes passed. No training,
Slurm submission, GPU inference or combined-run modification occurred.
[Validation receipt](output/shape-v2-20261123/validation.json).

## Three L40S GPUs verified on Slurm (21 November 2026)

Job **44896** ran on `gus` in partition `gpu`. Slurm accounting reported three
`nvidia_l40s` GPUs, 12 CPUs, 80 GiB RAM, elapsed 00:02:05, state COMPLETED and
exit code 0:0. Submission from the remote project root:

```bash
sbatch --parsable -p gpu --nodes=1 --ntasks=1 --gres=gpu:nvidia_l40s:3 \
  --cpus-per-task=12 --mem=80G --time=00:20:00 --job-name=l40s3-check \
  --output=output/l40s-three-preflight/slurm-%j.log \
  output/l40s-three-preflight/verify-BnIdEZnw.sh
```

The probe used the isolated cuFFT preload and verified the FFT round trip on
all three cards. The annotated PD-MC evaluator ran three concurrent workers on
devices 0, 1 and 4 with fresh outputs for three distinct previously successful
VG images. Each produced a completion PNG with a nonempty mask and exited 0.
The historical Surface wrapper requires four workers. This probe invoked the
shared evaluator directly with `--workers 3`. Shape-aware evaluation and
comparative timing remain pending. [Probe script, Slurm log and worker execution record](output/l40s-three-preflight/44896/).

## Canva Vietnamese refinement saved (16 November 2026)

User approved the reviewed 20-slide draft. Committed transaction `799911289478251733` to existing design `DAHbEx8zTOs`; verified saved title/page count. Edit: https://www.canva.com/d/45_3JDG-_oC01WZ . This supersedes the previous pending Canva transaction. Refined Vietnamese wording, hierarchy, labels, numeric conventions, tables and chart; research claims and Shape-not-implemented status preserved. All 20 Canva slides visually reviewed; no text outside canvas. Editable local PPTX and native PDF: `output/week8-vietnamese-refinement-20261116/`. PowerPoint checks 366 text shapes with zero overflow. See its README and receipts.

## Week 8 Canva upload (16 November 2026)

Uploaded the verified 20-page PDF through the connected Canva account to design
`DAHbEx8zTOs`: https://www.canva.com/d/WY6f-J86-YzFyRN . All 20 imported pages,
16:9 dimensions, Vietnamese content, formulas and numbers were checked. User
browser is at Canva login; the user will authenticate personally.

PDF import compressed spaces. Prepared and visually reviewed 23 text corrections
and repaired the split F_s subtitle in transaction `2028268083182796976`.
The original research characters are unchanged. Draft preview:
https://www.canva.com/d/m-1tLJxNMZhgtoB . An async approval question is pending:
Canva's editing tool explicitly requires preview approval before commit.
Do not commit before the user approves. Upon approval, commit this existing
transaction, verify saved text/page count and update CANVA.md and the receipt.
Do not create a duplicate upload. Details:
output/week8-research-story-20261116/CANVA.md and review/canva-import-validation.json.

## Week 8 research-story presentation delivered (16 November 2026)

Reworked the previous presentation into exactly the user's requested 20 slides.
Start at [the Week 8 package](output/week8-research-story-20261116/README.md).
Deliverables: editable PowerPoint, 20-page PDF, offline viewer, Vietnamese speaker
notes, an 18:20 talk track and rehearsal/Q&A PDF, sources, and a delivery ZIP.
The previous November 14 deck is preserved; its PPTX/PDF hashes were checked.

The user explicitly confirmed Shape-Aware is not implemented. Slides 10–11 leave
method/results to be defined; the combined architecture is proposed, and Shape/Ours
cells in the comparison and ablation remain TBD. Original PMC training-free is
distinguished from the fitted Surface selector. Controlled selection-only and full
progressive selection plus RGB corrections remain separate comparisons.

Severity is a new descriptive reaggregation of existing full-pair metrics on 321
verified COCOA annotation joins; the highest-occlusion bin has only four targets.
No new inference, training, tuning or Slurm job was run. The full LPIPS +1.191%,
4.733x allocation cost and inconclusive primary cohort remain visible.

All 20 slides were rendered in PowerPoint and visually reviewed. Checks found no
text overflow or shapes outside the slide; all slides contain notes. Headline
numbers were checked against saved evidence. Offline navigation, notes, timer and
image loading passed. Rehearsal materials are prepared, not a live human rehearsal.

## Annotated full continuation completed and audited (14 November 2026)

The user's requested continuation is complete. Read
[the full report](output/surface-annotated-full-review-20261114/REPORT.md),
[PDF](output/surface-annotated-full-review-20261114/REPORT.pdf), and
docs/SURFACE_ANNOTATED_FULL_RESULTS_20261114.md. Full 2,565-target LPIPS gain:
+1.191% [0.749%, 1.641%]; paired CLIP +1.323%, VGG +2.360%, SSIM +1.384%.
Frozen unseen: 2,017 targets / 1,900 images, LPIPS +0.801% [0.396%, 1.179%].
The original 204-target primary cohort remains inconclusive. No refitting.

Control saves 1,394 outputs, Surface-PMC 1,399 (12 gained, seven lost); failures
are 1,171 versus 1,166. Of Surface failures, 1,156 precede candidate sampling.
Capped outputs decrease 209 to 199. Shared-success LPIPS +4.044% is conditional,
not a 3% all-target gain. Recorded pools: 2,847 / 22,776 candidates, nine fallbacks.
Allocation cost including reused cohorts and timeout is 24.597 versus 116.406
GPU-hours (4.733x). Retain annotated PD-MC as the operational baseline.

39488 completed; 39490 timed out and the pre-submitted unchanged 39491 completed.
39492 verified/registered both arms; 39497 independent full/unseen audit and 39495
export completed. No study job remains to resume. Only read-only cluster recovery
was performed this turn; no new Slurm submission, training or inference occurred.
The collector recovered all four exports and exited.

All 16,197 metadata member hashes pass. Local provenance, both registry entries,
5,538 historical result-hash uses, all 408 preserved cohort records, all 2,847
stored-feature selector decisions, full/unseen arithmetic and bootstrap intervals
pass. Eleven focused regressions and both source/reference freeze checks pass.
Report validation checks 20,520 CSV values and all four PDF pages; visual review
is recorded. Native full PNGs remain on cluster; neural features/metrics and full
RGB correction were not rerun locally. Original cohort native replay is separate.

New helper commands: scripts/recover_annotated_full_followup.py,
scripts/review_annotated_full.py, scripts/build_annotated_full_report.py,
scripts/validate_annotated_full_report.py, scripts/package_annotated_full_review.py.
They preserve the original full-comparison and remote audit. Follow-up mechanism
experiments need a new frozen design; none were launched. Earlier pending-job
statuses below are historical and superseded by this completion.

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

## 10:20 UTC completed cohort, local delivery checks running

Original work window remains 06:46-11:46 UTC. Jobs 39486/39487 and pool diagnostic
39489 completed. Independent native cluster audits 39498/39499 passed and were
recovered separately. Frozen primary LPIPS: +0.249816% [-0.268570%, +0.810177%],
inconclusive. Secondary CLIP: +1.332553% [+0.257903%, +2.495991%]. Both arms have
87 saved outputs and 117 pre-sampling failures; capped outputs are 5 versus 7.
Surface generated 1,152 candidates / 144 pools with zero all-invalid fallbacks.
Final raw selector LPIPS versus uniform: -0.006931%, inconclusive. Final correction
LPIPS: +0.305666%, CLIP: -0.425185%, conditional on corrected masks/trajectories.

Recovered and independently hashed all 5,577 native members (1.033 GB archive),
1,269 metadata members and pool metadata. First extraction reported transient
OSError 28; unchanged retry succeeded. Windows subsequently showed 50 GB free.
Removed only eight independently hash-verified duplicate transfer parts after
successful native recovery, preserving the archive and receipts; cleanup receipt
is beside the archive. Collector 20156 remains active for full metadata.

Report workflow 81620 is running local WSL cohort replay, then pool replay,
build/validation/package. No duplicate workflow was started. Report interpretation
amendment adds upstream failure counts and native alpha counts for the already
frozen four extreme examples; it is explicitly post-hoc and changes no science
freeze or example selection. Core source/reference freezes still pass.
Read docs/SURFACE_ANNOTATED_RESULTS_20261112.md for measured findings.

Latest full-run state at 10:19 UTC: control 39488 running, 338/2,565 records, no
worker tracebacks. Surface 39490, timeout continuation 39491, finalizer 39492,
full/unseen audit 39497 and export 39495 are dependency-pending. Preserve them.
No new training or refitting exists. Final PDF/gallery visual review and portable
package delivery are still pending; do not call them completed yet.

## 09:50 UTC final cohort targets

Authoritative Slurm monitor: 39487 is RUNNING, 202/204 target records and 143
candidate pools, no worker tracebacks. The guitar target consumed all six allowed
passes; the remaining worker is progressing through its final targets. No
unfinished quality outcomes were examined. Deadline is still 11:46 UTC.

CPU audits 39498/39499 and exports remain dependency-pending. Collector 20156 and
report workflow 81620 are live. Eight synthetic transfer streams measured 883 KB/s
versus 304 KB/s for four; recovery now defaults to eight with whole-archive and
member SHA-256 checks. Three transfer tests still pass. Science freezes remain
unchanged. New scripts/recover_annotated_audits.py preserves completed remote
audits separately from the later local WSL replay; run it only after 39498/39499
complete. No actual cohort export has been recovered yet.

Reporting additions: mathematical figures, deterministic largest two positive
and two negative shared-success LPIPS examples (all 204 remain in the gallery),
portable source paths, recovery receipts and captured inference provenance in the
ZIP. Native rendering/validation/visual review still await completed artifacts.
docs/SURFACE_ANNOTATED_CONTINUATION_20261112.md records all full-run jobs and exact
monitor/recovery instructions. Full runs are valid longer continuations; they are
not completed cohort findings. No new training or refitting has been launched.

## 09:15 UTC recovery and audit continuation

Surface cohort 39487 has 164/204 target records and 111 pools, with no worker
tracebacks. No partial quality outcomes have been inspected. The original
06:46-11:46 UTC work window remains in force; do not restart its clock.

Submitted independent CPU native audits 39498 (cohort, after 39487) and 39499
(pool, after 39489 and 39498). Exact commands/hashes are preserved in
output/surface-annotated-20261112-design/remote-audit-submission.json. The
preflight initially imported user-site packages; PYTHONNOUSERSITE=1 confirms
the existing audit environment is NumPy 1.23.5 / Pillow 9.5.0 / SciPy 1.10.1.
No environment installation or running inference change was needed.

Parallel archive recovery uses four disjoint immutable ranges, validates the
whole remote SHA-256 and each archive member before extraction, and preserves
existing differing evidence. Three new transfer tests and five existing safe
archive tests pass. Original study/method/reference amendment freezes pass.
Hidden collector 20156 and report workflow 81620 remain live. The latter waits
for verified native cohort and pool metadata, then performs local pinned WSL
replay, report construction/validation and packaging. Remote CPU replay and
compact review images provide an independent evidence path if native recovery
extends beyond the deadline.

## 08:24 UTC continuation audit

Surface cohort job 39487 remains live: latest checked count is 46/204 target
records and 27 candidate pools, with no worker tracebacks. No partial quality
metrics were inspected. Fresh baseline cohort 39486 and GPU smoke 39496 are
complete; collector process 20156 is live. Deadline remains 11:46 UTC. Slurm
reports approximately 44 GB peak resident memory within the 64 GB allocation;
local disk has 53.6 GB free for cohort artifact recovery.

Added docs/SURFACE_ANNOTATED_METHODS_20261112.md with the exact historical
183-image ridge objective, annotation-assisted detection, candidate seeds,
observation anchoring, bounded affine/consensus corrections, released metric
operations, failure-inclusive estimands, image-cluster bootstrap and saved-pool
interpretation. Added a method diagram and appendix to the prepared report
builder. Report validation checks the appendix hash. These are reporting-only
changes; original experiment/method/amendment source freezes still pass.

Pending: complete cohort and pool diagnostic; recover native evidence; execute
independent audit in WSL surface-replay-py310; build/validate/visually inspect
report and all-target gallery; package; document exact full-run continuation.
No new training or fitting is being performed for this transfer test.

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

# Active session: Surface-PMC on annotated vocabulary (12 November 2026)

User work window: 06:46-11:46 UTC. Frozen design: docs/SURFACE_ANNOTATED_20261112.md.
Primary: 192 image-disjoint source images / 204 targets, annotated K=1 control versus
unchanged trained corrected K=8 Surface-PMC. No refitting. Both arms use annotated
scene queries and identical canonical annotated reference bytes; candidate masks
retain target-only segmentation. Exclude all 477 prior controlled-study images;
one near-copy was screened out before generation. All failures remain included.

Local source/protocol freeze passed; 54 source/design files deployed with backup and
SHA-256 receipt. Both reference copies passed all 10,261 file hashes. Jobs
39473/39474 run the control/surface cohorts, 39475/39476 their full continuations,
39477 the unchanged surface run after a timeout, and 39478 full verification and
registration. Diagnostic 39479 runs after the surface cohort, before its full
continuation. Exact commands/dependencies are in the design submission receipts.
Preserve all previous runs and keep AGENTS.md local.

07:13 UTC: control cohort has 89/204 target records with no worker tracebacks.
GPU CUDA-extension, FFT and pipeline-import checks passed. All 268 final CPU
tests, five shell regressions and synthetic smoke passed without skips. Additional
cohort/diagnostic focused checks passed. No paired quality result exists yet.

The cohort will be independently snapshotted and compared before unchanged full
2,565-target continuation. Full runtime is expected to exceed this work window;
leave valid jobs running and report their IDs/status. Do not reinterpret partial
results as full results or tune against the frozen cohort.

---

# Current session handoff

Updated 2026-11-11. Research runs complete; scientific briefing revision delivered.

## Scientific revision after the user's feedback (11 November)

The user found the manager deck insufficiently scientific and asked where the
formulas were. The corrected [research presentation](output/research-briefing-2026-11-11/presentation.pdf)
retains the complete project history and integrates 35 scientific slides into
the main narrative: **75 slides, 57 rendered equations**, and approximately
19,300 words of detailed explanation. [Offline HTML](output/research-briefing-2026-11-11/presentation.html),
[notes](output/research-briefing-2026-11-11/briefing.md),
[equation index](output/research-briefing-2026-11-11/equations.md), and
[exact ranker parameters](output/research-briefing-2026-11-11/ranker-parameters.csv)
are included in the portable package.

Coverage includes formal notation, PMC context mixing, surface and local losses,
validity/normalization, conditional-risk and identifiability arguments, exact
gain/regret decompositions, all 22 fitted features and coefficients, ridge target
and derivation, affine/consensus corrections, cohort lineage, metric protocols,
paired bootstrap, practical thresholds, real gains/losses, compute and the saved-pool
oracle bound. Every equation has explanatory text and evidence links. Original
proposal, implementation, conditional theory and measured evidence are distinguished.

Validation covers rendered mathematics, all slide bounds, navigation/Notes,
all PDF pages/equation labels, links, parameter equality, scoped numeric claims,
source hashes and ZIP payloads. PDF formula pages and contact sheets were visually
reviewed. Both original manager briefings and 215 inventoried source/test/configuration/
helper/old-brief files are preserved. This is a reporting correction: no new
experiment, remote operation or neural-metric rerun; existing conclusions stand.

## Latest reporting task: updated manager briefing (11 November)

The user requested the same format as the 5 November manager brief. Delivered
[40-slide PDF](output/manager-briefing-2026-11-11/presentation.pdf),
[offline presentation](output/manager-briefing-2026-11-11/presentation.html),
and [detailed briefing / speaker notes](output/manager-briefing-2026-11-11/briefing.md).
The notes contain approximately 10,300 words, displayed tables and evidence links.
The package includes the unchanged eight-run historical CSV, a 28-row scoped
results appendix, source hashes, and a portable ZIP.

The story covers engineering, dataset/protocols, historical recovery, accepted
baseline, both appearance audits, pending human ratings, initial surface study,
fitted follow-up, independent confirmation, full progressive results and final
diagnostics. It preserves the controlled confirmation's positive scope and the
unmet full 3% target. All research jobs remain complete; no new experiment,
neural-metric computation or remote operation was performed for this briefing.

Delivery checks cover every slide's bounds, all embedded images, navigation,
Notes mode, PDF page count/text, evidence links and archive hashes. The original
manager brief and 181 inventoried maintained-source/test/configuration/helper
and old-brief files remain unchanged. Read the package's validation.json and
delivery-receipt.json for the final checks and ZIP identity.

## Latest: final-RGB diagnostics completed and audited (11 November, 12:01 UTC)

Cluster access is restored. Jobs **39404** (7m28s) and **39407** (31s) completed
successfully. All 2,565 ablation records and 958 raw derivations were recovered
and independently audited. Raw-final versus matched control gains: CLIP **1.423%**,
LPIPS **0.596%**, VGG **2.121%**, SSIM **1.087%**. Against corrected output, raw-final
improves CLIP 0.293% but worsens LPIPS 0.240%, VGG 0.650%, SSIM 0.683%. Original
score replay drift is at most 1.79e-7. The full 3% objective remains unmet.

Submitted and completed frozen headroom diagnostic **39411** (23m36s): all eight
saved final candidates, 958 pools / 7,664 scored RGB images, with corrected final
masks, geometry, preceding trajectories and all failures fixed. No diffusion or
fitting. Individual reference-informed raw-pool ceilings versus control: CLIP
**2.817%**, LPIPS **0.740%**, VGG **3.257%**, SSIM **1.407%**. These use DIFFERENT
oracle choices and are not a combined attainable output or deployable selector.
Three ceilings below 3% rule out final raw RGB reranking alone on these saved
outputs. New generated pools, candidate masks and earlier trajectories are outside
these bounds. Do not launch an expensive raw full run based solely on controlled
gains or assume final correction caused the full shortfall.

Read [the combined report](output/surface-lastpass-review-20261111/REPORT.md),
[headroom audit](output/surface-lastpass-review-20261111/headroom-audit.json), or
[portable package](output/surface-lastpass-review-20261111/surface-followup-diagnostics.zip).
The ZIP is 569,403 bytes, all nine payload hashes/CRC pass, SHA256
6590b4eab32ae1fef70f844eb7235b4c50522015d6a8539ddcbefae25760454a.
The ablation audit verifies 2,577 recovered members and 7,047 original files;
headroom audit verifies 2,582 recovered members, 2,565 parent links, 7,664 metric
tuples, 3,832 oracle choices, every mean and paired image-cluster interval.
Native PNGs remain on the cluster; local neural metrics were not rerun.

Headroom freeze: output/surface-candidate-headroom-20261111-design/freeze.json,
SHA256 dc0f22bfdf1ff72e31f53400a520ffc303c3a55a353a6ce0a1cd81b732ac7cd7.
Exact command is in submission.json. Preflight verifies nine sources, all 2,565
parent records and native selected-RGB serialization. Both GPU integration checks
pass. All selected image hashes and metric scores reproduce exactly, with zero
score drift. Seven new focused audit/headroom regressions pass. Source freezes
and previous scientific implementations were preserved.

Both collectors finished, including headroom session82848; local audits8453/10560
also finished. No jobs need resuming and no inference jobs remain active. Do not
resubmit completed work. The original five-hour deadline was exceeded. The
independent 128-image raw-selector confirmation remains positive within artificial
occlusion; the full progressive goal remains unmet. Further work needs a separately
frozen generation/trajectory intervention or another explicitly isolated mechanism,
while preserving these negative results and all original denominators.

## Latest: full result verified; final-pass diagnostic queued (11 November, 11:00 UTC)

Full surface continuation **39387** completed successfully at 10:17:13 UTC;
finalizer **37814** completed at 10:24:23 UTC. Both runs have all 2,565 targets,
passed full/provenance verification and were registered. The old evidence
collector finished recovery; do not restart it.

Full paired gains: CLIP **1.123%** [0.768,1.480], LPIPS **0.835%**
[0.520,1.173], VGG feature similarity **2.789%** [1.911,3.764], SSIM **1.782%**
[0.934,2.694]. The full 3% objective is NOT met. CLIP uses 946 shared successes;
other metrics retain all failure penalties. No natural hidden RGB MAE exists.
Saved outputs: control 951 / surface 958; shared 946, gained 12, lost 5. The independent
128-image raw-selection confirmation remains a separate positive result.

Read output/surface-progressive-review-20261111/REPORT.md or REPORT.pdf; the
portable surface-progressive-review.zip includes the audit and all paired records.
Local audit independently reproduces means, intervals, failures and denominators,
verifies 14,123 recovered members and 4,921 pre-resume record hashes (2,565 control
plus 2,356 surface). Native PNGs remain on the cluster; neural metrics were not
rerun locally. The old timed-out execution snapshot remains unfinished, while its
successful continuation and pre-resume records are verified separately.

The sampler ran on 966 targets; 1,599 never reached it. Shared-success LPIPS gains
4.165%, versus 0.835% over all targets. Do not substitute that diagnostic subset
for the frozen primary denominator or change detector vocabulary to claim a
surface gain.

New frozen diagnostic: output/surface-lastpass-ablation-20261111-design/freeze.json,
SHA256 4866d3f2a1a26e207b45a101ef8e3c0316cc18af828188b1edf017229a7210c8.
It uses each existing final raw candidate selected by the frozen ranker, retains
the corrected run's final mask, frame, geometry, failure status and preceding
trajectory, and removes final RGB corrections/restoration. Baseline, corrected
surface and raw-final images are rescored together. This is exploratory, NOT an
independent holdout or an end-to-end raw progressive method. No new diffusion,
ranker fitting or hyperparameter search is performed.

Four-Quadro/four-worker measurement job **39404** is queued (two-hour limit).
CPU finalizer **39407** waits on afterok:39404 and produces analysis/REPORT.md,
analysis/summary.json and a hashed evidence archive. Exact commands and uploaded
source hashes are in the design submission.json and finalizer-submission.json.
The last start estimate was11:37 UTC /18:37 Vietnam; a test-only gpu-partition
probe was later and created no submitted job. Preserve these valid jobs.

New read-only collector is alive in unified execution session **62314**:
`.venv/Scripts/python.exe -u scripts/collect_surface_lastpass_evidence.py --measurement-job 39404 --finalizer-job 39407`.
It writes output/surface-lastpass-ablation-20261111-recovery/status.json and
recovers completed numerical evidence with independently obtained hashes. Do not
start a duplicate. It stops on integrity or job failure; network errors retry.
The previous network interruption was resolved when the user reconnected.

Four focused derivation regressions pass on Windows and pinned WSL Python3.10.
Remote preflight verifies frozen hashes and one real candidate's crop, selected
identity, alpha, and JPEG/RGBA serialization. The queued job performs the existing
GPU integration checks before scoring. No frozen scientific implementation was
edited. Corrected accidental spaces in four confirmation handoff paths.
Original five-hour deadline exceeded; no full metric success is claimed.

Next: check 39404/39407, recover and inspect raw-final versus corrected and matched
control comparisons, including original-score replay drift. A final-pass result
cannot establish what a wholly raw progressive trajectory would do. Do not start
an expensive new full run based solely on the already-known controlled gains.

## Current result and resumed full run (2026-11-11)

Cluster access works again after the user reconnected. Confirmation job **37843**
completed successfully in 1h20m33s. Its independently frozen 128-image test meets
the four predeclared controlled endpoints versus expected uniform selection from
the same eight candidates: MAE +7.644% [4.287,11.291], object LPIPS +11.328%
[8.021,14.828], DreamSim +16.766% [11.808,21.583], CLIP cosine +4.068%
[3.504,4.653]. Intervals are individual nominal 95% paired source-stratified
bootstrap intervals, not simultaneous coverage. Zero failures in 1,024 candidates.
SSIM improves only 1.668%; do not claim every recorded metric clears 3%.

Local audit passes 2,631 recovered archive member hashes, 1,024 candidate identities,
128 rank decisions, exact selected RGB/masks, selection-before-scoring timing,
all summary means/intervals, and independent selected-image hidden MAE for all
128 cases. Native generation-node replay passed; local audit did not rerun neural
metrics. Read output/surface-confirmation128-20261110/report/REPORT.md and REPORT.pdf.
All 128 gallery images load; filters show 64 VG, 64 COCOA, 86 gains and 42 losses,
with no horizontal overflow. PDF layout was visually inspected. This is one-step
artificial occlusion, not a completed full progressive benchmark improvement.

Full job **37812** reached its 24-hour limit after 2,356/2,565 target records.
Continuation **39387** was submitted at 07:17:49 UTC on 11 November, with four
Quadro GPUs, four workers and 12 hours, using the same freeze/output directory.
The one incomplete target was archived before retry; completed results were kept.
Exact command, archive hash and dependency update are in
output/surface-followup-20261110-design/resume-20261111-submission.json.
Finalizer **37814** now correctly waits on afterok:39387. At 07:42:53 UTC, control
has all 2,565 records and surface has 2,383; the job is RUNNING. Its corrected
policy differs from the independently confirmed raw selection policy.

The read-only collector now watches **39387/37814/37843**, using
`.venv/Scripts/python.exe -u scripts/collect_surface_job_evidence.py --progressive-job 39387`.
Existing Windows process pair 40212/37892 is one collector (venv launcher/child),
not duplicate collectors. Do not launch another while it is alive. Read
output/surface-job-recovery-20261110/status.json for current progress. Five collector
regressions pass, including replacement-job monitoring. Earlier session 20571 and
the queued/unmeasured confirmation statements below are historical.

Remaining work: recover finalizer outputs, review full denominators/failures and
paired comparisons, and report the full result separately. Preserve the frozen
method and previous primary/secondary labels; do not tune on confirmation data.
The original five-hour deadline was missed. Leave valid inference running.

Portable delivery: output/surface-confirmation128-20261110/START_HERE.md links
surface-confirmation-delivery.zip (5,341,427 bytes; 443 members; all 442 payload
hashes and ZIP CRC pass). SHA-256:
e988278dbe6f8bc45ed7336d230c2deffbc1a47a1b135497b40e49e70a70b924.
The earlier 49 MB reading-package.zip is superseded; it also contains redundant
native selected PNGs. Final delivery omits those images but includes every gallery
panel, all metric records/decisions, frozen source and audit receipts.

Final health check at 07:49:16 UTC: 2,394/2,565 surface records; two workers have
finished and both remaining logs are actively updating, with no recent traceback.
Both remote frozen-policy hashes remain exact. Finalizer dependency is still
afterok:39387. Receipt: design/resume-20261111-health.json (under the original
surface-followup-20261110-design directory). The five collector regressions pass
on Windows Python 3.14 and pinned WSL Python 3.10. Do not interpret the unfinished
full comparison as either a positive or negative quality result.

## Active five-hour follow-up (2026-11-10)

05:19 UTC: goal marked **blocked awaiting external Slurm results**, after the
same missing-result dependency persisted across more than three goal turns and
all independent review/recovery preparation was completed. This does not stop
any experiment. Last live collector output:37812 RUNNING,1,076/2,565 control
records;37814/37843 PENDING. Collector20571 is live, polling and ready to recover
finished evidence. The metric objective remains unverified; resume assessment
when confirmation and full-comparison artifacts exist or a job requires repair.

05:17 UTC CURRENT: all independent implementation/review work is complete;
remaining quality verification requires externally produced Slurm artifacts.
37812 remains RUNNING with 1,068 control records; 37814 and 37843 are PENDING.
There are zero confirmation candidates and neither final result artifact exists.
The same external result dependency has persisted across at least three live
goal checks. No method change is justified by incomplete test outputs.

Added and started a read-only evidence collector in execution session **20571**:
`scripts/collect_surface_job_evidence.py`. It polls all three jobs every 60 seconds,
recovers the finished confirmation archive and full progressive reports/JSON
metadata, validates independent remote checksums, and preserves differing local
evidence. It never mutates remote files or jobs. Four focused safeguards pass on
Windows and pinned WSL Python 3.10; immutable confirmation preflight passes.
Read output/surface-job-recovery-20261110/README.md and status.json. Poll session
20571 rather than launching another collector. Old monitor77472 still exists.
The correct full result filename is progressive-comparison/comparison.json,
not summary.json. After artifacts exist, review the recovered reports and
receipts before deciding whether the original metric objective is achieved.

05:07 UTC CURRENT: capacity review found no faster validated execution route.
Local RTX 3060 has 6 GiB VRAM; the full runtime does not fit its validated
configuration. A read-only `sbatch --test-only --partition=gpu --gres=gpu:4`
probe returned 21:48:49 UTC, later than pending 37843's 09:45:49 UTC estimate.
The test-only probe created no job. Preserve all existing allocations. Current
37812 control has 963/2,565 records; all four worker logs updated within ten
seconds and have no traceback in their last 200 lines. Surface has not started;
37814 waits afterok and 37843 has zero candidate records. No method, source,
cohort, or scientific measurement changed. Live receipts and jobs.json are
updated; see confirmation design/status-20261110T050720Z.json. Monitor 77472
still runs for 37812/37814; query 37843 separately. The queued confirmation is
estimated for 16:45:49 Vietnam time, subject to scheduler changes. Remaining
work depends on these valid jobs finishing, then artifact recovery and reporting.

04:17 UTC CURRENT: independent selection-only confirmation is frozen and
submitted as Slurm **37843**, debug-gpu, four GPUs/four workers, four-hour limit.
Read docs/SURFACE_CONFIRMATION_20261110.md. New experiment is
output/surface-confirmation128-20261110; design is the same name plus -design.
All 128 cases (64 VG, 64 COCOA) exclude all 349 previously used images. Eligibility
was 861; decoded and aligned near-copy audits pass. Freeze SHA is
711f49db26e19d0655c64b59682d9ae50c0841d38071096ad9dc7f83d733d4c4.
The parent ranker/seeds are unchanged; correction, consensus, and visible RGB
restoration are disabled. This independently confirms the previously secondary
raw component; it does not replace the earlier corrected primary. No outcomes
from these 128 have been generated or measured. Remote preflight passed hashes.
The source/input archive SHA is c07874df475cece5ec04acf9ed7189ca2cafd4090f9f178c02f78e19dbd93038.
New scripts prepare_surface_confirmation.py, finalize_surface_confirmation.py,
and run_surface_confirmation_slurm.sh are frozen with their dependencies.
248 CPU tests: 242 passed, six optional-dependency skips; shell/smoke pass. The
two new regressions also pass in the pinned replay environment. No parent-frozen
scientific source was edited. Job 37843 runs GPU checks, all 1,024 candidates,
features, observation-only decisions, metrics, analysis, exact same-node replay,
and creates analysis/CONFIRMATION.md plus design/confirmation-evidence.tar.gz.
Download the archive and independent .sha256 after completion; preserve raw
data and report whether every point estimate and/or lower bound exceeds 3%.
Submission receipt and exact command are design/submission.json. Actual squeue
estimates 09:45:49 UTC; cluster display is EST (04:45:49). Estimates can change.
Do not submit duplicate work. Full progressive 37812 remains running (823
control records at 04:50 UTC); 37814 waits afterok. All four worker logs update
without a recent traceback. This continuation is a verified wait. Current
control uses about10 GB per allocated24 GB GPU; reported batch peak host RSS
is46 GB against64 GB allocated. Receipt: design/runtime-capacity-20261110T0432Z.json
under the original followup design directory. This is current control capacity,
not a certification of full surface-stage peak memory. No scientific code,
policy, allocations, or measurements changed. Monitor session77472 still
works for those two jobs only; query 37843 separately. The original five-hour
deadline is already missed. Keep the goal active until measurements exist.

03:59 UTC historical: read-only SSH monitor is running in unified exec session77472.
Poll it with write_stdin(session_id=77472, chars='', yield_time_ms<=60000); it
queries authoritative sacct states and counts both run directories every60seconds.
It stops on a terminal primary failure or terminal finalizer, without changing jobs.
Its latest live output:37812 RUNNING,37814 PENDING, control302 targets, surface0.
Current receipt: design/progressive-status-latest.json. This goal turn was a
verified wait; the control advanced without requiring code or experiment changes.
All four worker logs had updated within9-20seconds at03:52; none of their last
200 lines contained a traceback. The recent slower minute was not a stalled job.
This continuation is a verified wait; no code, policy or scored results changed.
If monitoring disconnects, re-query those same Slurm job IDs; a dead monitor does
not prove the GPU job stopped. Do not start duplicate inference or reset outputs.
No scientific work remains independent of the pending full comparison; keep the
goal active while these valid jobs advance. The delivered controlled result and
frozen parameters remain unchanged.

03:36 UTC historical: verified live job37812 advanced to151/2,565 control targets;
37814 is still pending. Storage has28TB available. Independent inspection of
the final progressive smoke verifies19 artifact hashes, eight distinct derived
seeds, the target seed and diagnostic linkage, selection index1 under the frozen
ranker (maximum loss discrepancy1.11e-16), and exact preservation of81,241 observed
pixels in selected.png. Receipt: design/progressive-smoke-artifact-audit.json.
This audit is one smoke pass, not full-run validation or another quality result.
No new method changes or experiments were made; the full comparison is pending.

03:32 UTC historical: job37812 is live; all four control workers are generating.
Matched control has127/2,565 recorded targets (37 saved,90 failures); full surface
starts after control verification.37814 remains pending on37812. This is progress
and a verified wait, not a stalled process or a completed full comparison.
Improved the pending comparator to show separate overall means/denominators,
per-source saved/failure counts, shared-success CLIP, and immutable analysis-source
copies. A synthetic changed-coverage case exposed a zero-baseline bootstrap crash:
full-run analysis now reports the relative interval unavailable, counts undefined
draws, and retains the absolute interval; it never drops those draws. Empty shared
CLIP is similarly unavailable. No fresh controlled metrics or method changed.
Nine focused tests pass locally and under pinned cluster Python3.10. Four uploaded
file hashes match; all17 frozen scientific sources remain byte-identical. Receipt:
design/progressive-report-deployment.json. Remote prior reporting sources backed
up in design/progressive-report-source-before-20261110T0331Z. The delivered reading
ZIP is immutable; these full-comparison reporting changes are separate.

03:26 UTC historical: controlled deliverable is complete and locally verified.
Read output/surface-followup-20261110/START_HERE.md. Three-page report PDF,
all166 searchable panels, numerical records, choices, frozen source and receipts
are packaged in surface-followup-reading-package.zip (8,839,075bytes;555 members;
SHAe57f9a6779584853201154746923558b8ba1dece5fbcd6390c18bdb3161e1e05).
All ZIP member hashes pass. Fresh evidence65,366,265bytes matches remote SHA
3bddef2f178b29db9979fa1869be258a99ff91b1cc2f0472fffc742c658b949d and is extracted.
Report validator recomputes primary/secondary means and checks332 selected
artifacts. Browser confirms166 loaded panels,120 MAE gains/46 losses, correct
source/search filters and no horizontal overflow; PDF is visually reviewed.
The control-comparison table now explicitly labels gains of the primary over
each named control. Frozen scientific files and all166 decisions are unchanged.
Job37812 passed its final eight-target/four-worker progressive smoke, including
a saved surface completion, and provenance verification. At03:25:48UTC its full
matched control has82 target records; full method has not started yet. Valid job
37812 remains running on four Quadros;37814 waits afterok to verify/compare/register.
Current remote snapshot is design/progressive-status-delivery.json; smoke receipt
matches independent remote SHA. No new studies should be tuned on fresh outcomes.
The original five-hour deadline was missed. Primary CLIP misses3%; secondary
selection-only clears all four point estimates, with MAE/CLIP intervals below3%.
Full progressive results remain outstanding. Do not mark the ultimate metrics
objective achieved from controlled gains or terminate valid long-running jobs.

03:13 UTC historical: fresh166 scoring finished. Corrected primary gains are
MAE9.086%, LPIPS12.560%, DreamSim18.553%, CLIP2.422% (CLIP target missed).
The pre-specified selection-only component trained_surface_raw reaches
MAE5.845%, LPIPS6.754%, DreamSim12.764%, CLIP3.075%; all point estimates exceed3%,
but raw MAE/CLIP intervals extend below3%. Keep this SECONDARY; no post-test
refreeze or substitution of the primary. Summary/csv are local and archive SHA
verified in fresh-summary-download-receipt.json. Zero candidate failures.
Job37785 failed strict primitive-score replay after successful GPU measurement.
Cross-CPU Lab histogram boundary changes persist even with pinned packages;
CPU37798 on floyd also failed. Original generation node gus matches exactly.
Native CPU-only job37799 is RUNNING there (~14min, healthy four-core compute).
Dependent37812 has four Quadros for final smoke then full matched control/method;
37813 packages all controlled evidence after37799; 37814 finalizes full outputs.
Old37795 was cancelled/superseded. None of the17 frozen scientific files changed.
Wrapper run_surface_final_slurm.sh gained SURFACE_PROGRESSIVE_ONLY=1, requiring
a complete matching replay before skipping already-finished controlled work.
New wrappers isolate user-site packages. Failed runner snapshot is preserved.
Extra candidate transfer63363 FINISHED, SHA91a8ea4661a00f08bc504ffd3f2fd8f7db7d3791d5a96337318eb7ac8c643f3e.
Extracted to design/extra-pools-166/full; receipt saved. Both four/eight native
candidate pools are now local. Evidence archive from37813 is not ready yet.
Once replay succeeds, download design/fresh-controlled-evidence.tar.gz and
independent .sha256, verify then extract at repo root (this archive has full
project-relative paths). Rebuild report with --reuse-panels using cookbook,
run validate_surface_followup_report.py, inspect PDF/gallery, update findings,
and run package_surface_followup.py. START_HERE.md and report builder already
include primary/secondary distinction, exact values and elapsed deadline.
Full GPU job remains valid long work; do not mark goal complete before required
results exist or promise that the5h target/time was met. Original deadline passed.

02:51 UTC: final job37785 is healthy. Observation-only inference finished all166
at02:48:59UTC, before fresh scoring. Quadro CUDA extension, real FFT, full imports
and four observable-feature audits pass (zero feature difference). Complete CPU
suite passes241 tests plus shell regressions/smoke. Dependent CPU job37795 is
queued afterok:37785 for scripts/finalize_surface_progressive_slurm.sh. Extra
candidate SCP session63363 remains active; do not hash/unpack its partial archive.
Need retrieve completed fresh metrics/inference/analysis/runtime evidence, build
report and all166-case gallery, record measured results and any missed targets.

2026-11-10 02:43 UTC continuation: original five-hour deadline HAS PASSED, do not
claim on-time completion. User explicitly kept goal active. Job37568 finished
all349 eight-candidate pools and fresh observable features at20:43:57UTC Nov9.
No fresh quality metrics were opened before final freeze. Eight-pool source-CV
selected ridge100 all-appearance+CLIP(-16), affine1, consensus0.5/sigma4. CV gains
MAE11.415/LPIPS14.378/DreamSim19.042/CLIP3.150%; unconfirmed development estimates.
Final freeze: output/surface-followup-20261110-design/final-method-freeze.json,
SHAa0a626fcfddec095a1d7b677b496309fccb79ff6b4e446f5127b69368f3358e1.
All17 frozen source files are deployed and hash-checked remotely. Fresh truth
archive uploaded with matching SHA199c8c6a697404b91a7821bd5871d8bfa9b1ec9a3256180984403f38f1b7fdfe.
Final Slurm job37785 submitted debug-gpu/fourGPUs/24h (Gus busy, Quadros available).
It selects all166 observation-only outputs, computes all raw/refined/calibrated
metrics, analyzes with cached paired bootstrap, independently replays scores and
MAE, runs frozen8-pool progressive smoke, then matched full baseline and surface.
Full baseline path is now pdmc-matched-control-20261110 (not L40-specific).
Main log design/final-37785.log. Fresh outputs metrics-eight/analysis-eight/
inference-eight under output/surface-followup-20261110. Preserve frozen method;
no tuning after fresh metrics. Report scope accurately, never claim controlled
results prove full progressive improvement.
Extra166 candidate bulk download running exec63363: fresh-extra-results.tar.gz,
expected SHA91a8ea4661a00f08bc504ffd3f2fd8f7db7d3791d5a96337318eb7ac8c643f3e.
Dev87 JSON archive downloaded/extracted and SHAverified (4f20c250e732d3ad843de359a43fc01f8dd6459716941f27f110cad6e7a25295).
Four eight-pool search jobs finished; best linear beat all nonlinear alternatives.
Analysis caching change passes exact source-stratified CI regression; no model
math changed after freeze. Need final artifact verification/report/gallery and
monitor legitimate long full runs; goal not yet achieved.


19:59 UTC update: all 183 old images have eight generated candidates. Development
87 metrics finishing; ensemble87 correction measurement ~52/87 cases. Extra166
fresh generation starts next. Four-pool nonlinear comparison finished, best still
ridge (MAE +2.674%, LPIPS +9.999%, DreamSim +13.621%, CLIP +2.462% in source-weighted
leave-source-out dev CV). Final eight-pool grid will include the same modest tree
family; no fresh outcomes viewed. New portable tree scorer in surface_refinement.py;
CPU-only sklearn1.2.2 installed in WSL pinned environment, no GPU dependency changes.
Nonlinear ordering/export test passes. Full pre-freeze suite passes238, shell
regressions and synthetic smoke; new validator and deployment-hash tests also pass.
WARNING final deployment needs uploading new surface_refinement.py, surface_policy.py,
measure_surface_followup.py, select_surface_pool.py, develop_surface_multimetric.py,
freeze_surface_followup.py and final runner before final freeze/evaluation. Existing
remote running development jobs use the earlier source (same corrections/ridge math).
Final freeze should be made in WSL; generator now stores POSIX source-hash keys.
Fresh metric and observation-only scripts now verify every frozen source before
proceeding, so stale runtime cannot silently consume a nonlinear frozen model.
Full comparison runner prepared, not submitted. Both matched-runtime control and
surface development smokes complete and provenance-verified. The large ensemble96
archive transfer was intentionally stopped (contained images); JSON-only archive
is downloaded/extracted with matching independent SHA948e8d50c67ba3bfeb47beb0521da70240a98782969aa5c9f54087aa2600becc.


19:50 UTC update: 37568 still generating extra seeds on the 87 old images (~240/348
records); 96 extra pools, features, metrics and joint correction pools complete.
Fresh truth remains unmeasured. Final selector NOT frozen. Fresh-four archive is
fully downloaded/extracted and hash-verified; prior SCP finished.
Progressive smoke step 37568.7 failed DINO/InstaOrder utils collision. A scoped
module-cache isolation fixes this; six runtime tests pass. Step 37568.8 succeeds,
all four workers, 1/8 saved completion, scored and provenance verification passed.
Output: pdmc-surface-progressive-smoke-v2-20261110. A matched L40 baseline smoke
runs in exec 37076. The runner now supports baseline invocations (without surface
cache CLI flag). Failed first smoke is preserved. New final workflow script is
prepared but NOT submitted: run_surface_final_slurm.sh will select observation-only,
measure 166 fresh cases, analyze/replay, final-policy smoke, then matched L40 full
baseline and full progressive surface run (each 2,565). Longer runs may continue.
Local ensemble-eight-96 metadata transfer uses exec 13446; verify completion and
SHA 5d5466e8cc8164ef6cb5b76a3c5d4881947f4f62ecbe399191cef16147e5b8e2 before use.


19:24 UTC update: job 37568 is running extra four seeds (first cohort 96 nearly
finished). Both four-pool ensemble metric cohorts finished in 37568.2; metadata
archive hash verified locally. Correction strength=1,sigma=4 reduces dev selected
MAE .165210 -> .157371, with small CLIP loss. Final eight-pool method NOT frozen.
New primary development utility matches known fresh source counts 64/64/38/0;
unweighted sensitivity remains. No fresh truth metric has been opened.

Opt-in progressive adapter now in surface_runtime.py and evaluate_open_world.py,
with run_surface_progressive_slurm.sh. It retains original measured support
across iterations, explicitly seeds candidates, restores original visible pixels
at native resolution before final segmentation, and retains upstream stopping.
Four focused tests pass; full WSL check passes 233 tests, all shell regressions,
and synthetic smoke. Additional source-weight arithmetic test passes. Runtime
feature-provider equivalence audit runs in 37568 via exec session 60380; logs in design/runtime-feature-audit-logs. No GPU progressive
smoke or full surface run has been launched yet. Source/evaluator uploaded.

Fresh four-pool bulk SCP started 19:05 still running (Windows scp PID 83196);
local fresh-four-results.tar.gz is partial until transfer completes. Do not unpack
or claim verified until remote independent SHA256 matches.


User requests 3–5% relative MAE and paper-metric improvement, continuing the
surface-consistency direction. Window: 2026-11-09 18:07–23:07 UTC. Goal is active.
See `docs/SURFACE_FOLLOWUP_20261110.md`. New cohort `output/surface-followup-20261110`
contains 166 images (64 VG, 64 COCOA, 38 free-images), disjoint by identity and
bytes from all 183 earlier cases. LAION eligibility is exhausted. All old 183
cases are explicitly development now, without rewriting their old test results.

19:03 UTC continuation: all 183 development images now have paper-style black-
background LPIPS/DreamSim/CLIP measurements and observation-only deep features.
The balanced multi-metric ridge grid is in `development-multimetric-v2`; four
candidates have only 3.15% maximum development CLIP headroom, and forcing CLIP
hurts MAE. Additional seeds 55,66,77,88 for all 349 images are queued in Slurm
**37568** after **37566**, with three cohorts 96/87/166. Script
`scripts/run_surface_extra_slurm.sh` generates each cohort, extracts eight-pool
features, and measures OLD development cohorts only. Fresh truth is still local
and unopened by any metric job. Never tune on the fresh 166 outcomes.

Development metadata archives are locally verified against remote SHA-256.
`provisional-development-eight.json` is remote and deliberately provisional.
Eight-pool outputs go to design `extra-pools-{96,87,166}/full`; eight-pool
development metrics/features go under design `eight/`. Primary protocol now uses
eight candidates with equal-budget validity/consensus/semantic/deep controls.
Final controlled parameters are NOT frozen yet.

Full cached postprocessing is separately frozen in `full-calibration-freeze.json`
(affine strength 1, max shift 32, exact observed-pixel restoration, unchanged
alpha). Remote `output/pdmc-surface-calibrated-20261110` has all 951 outputs and
1,614 original failures. Paired scoring in allocation 37566 completed. Finalizer
exec session **26847** verifies/aggregates all 2,565 targets; read log
`full-calibration-finalize.log`. The finalizer uses source-image cluster bootstrap.
Do not change this frozen calibration after seeing full metrics. Four-source
visual audit is saved locally as `full-calibration-visual-audit.jpg`.

An extra consensus correction is being developed before fresh-test exposure:
`amodal_benchmark/surface_ensemble.py` adds bounded smooth median-candidate RGB
residuals only on mask-agreement support, keeping chosen masks and visible pixels.
CPU grid runs locally in exec **35609**, writing `ensemble-development-four.json`
on completion and progress to its `.log`. It is roughly halfway done. Its five
core tests (four ensemble, one clustered bootstrap) pass; two deployable-policy
tests also pass. Optional SciPy 1.10.1 is now installed only in the isolated WSL
`surface-replay-py310` environment. All six ensemble/policy tests pass there.

New `surface_policy.py` and `scripts/select_surface_pool.py` can run the final
method from observation manifests without evaluation data. Metric evaluator
supports `--extra-run-dir`, `--features-dir`, variable seeds, optional ensemble
correction, calibration-only ablations and explicit valid-pool controls.
Feature audit passed four GPU cases: hidden-observation randomization invariance,
zero identical-visible distances, and official DreamSim forward equivalence.
Latest full CPU log: `logs/surface-followup-check-extended-20261110.log`: 226 tests,
five optional SciPy skips (subsequently checked separately), all Bash regressions
and synthetic smoke. PowerShell incorrectly reports native stderr as wrapper exit
1; inspect completed logs/native exit codes, do not misreport test failures.

New module `amodal_benchmark/surface_refinement.py` implements bounded visible
residual calibration and a small regularized within-pool ranker. Development
leave-source-out search currently selects ridge=10 and LPIPS objective, giving
exploratory cross-validated MAE reduction 4.995% and masked LPIPS 5.748%; these
are not fresh-test results. Correction grid is running locally in exec session
19286; outputs go to `output/surface-followup-20261110-design/development`.

Quadro job 37564 was pending until after the deadline and was cancelled by us.
L40S smoke 37565 passed CUDA extension forward/backward, real FFT, full imports,
and all eight generated/masked candidates. A separate cuFFT 10.9.0.58 library
at `.amodal-cache/surface-cufft-cu118` is loaded with LD_PRELOAD only for this
study. The original pinned library remains unchanged. Cross-runtime RGB is
not byte-identical; MAE differences are recorded in `smoke-audit.json`.
Fresh 664-candidate generation is job **37566**, four GPUs/four workers on gus.
Log directory: `output/surface-followup-20261110-design/job-logs`.

An isolated `surface-metrics-env` inherits the pinned runtime and adds DreamSim.
Setup initially found missing wcwidth, then peft 0.2's custom-model config error;
script is updated to wcwidth 0.2.13 / peft 0.5.0, pending rerun. Weights are cached.
Local smoke download matches independent remote SHA-256; four seed-11 images
were visually inspected. Do not overwrite accepted baseline or historical studies.
Freeze the method before evaluating fresh truth; then run controlled metrics and
the separately identified full-baseline surface-postprocessing comparison.

## Surface-PMC task completed (2026-11-08)

The requested theory-to-evidence work is complete within the 07:42-12:42 UTC
window. Start with [the handoff](output/surface-consistency-20261108/START_HERE.md),
[13-page report](output/surface-consistency-20261108/report/REPORT.pdf), or
[durable results note](docs/SURFACE_CONSISTENCY_RESULTS.md).

All 732 candidates for 183 images are available locally and on the cluster.
Development has 24 images; pilot test 72; image-disjoint replication 87.
Primary full-surface pooled gain is +0.004310 [-0.000512, +0.009074], inconclusive.
The fixed local variant has exploratory pooled gain +0.007187
[+0.002183, +0.012296], with supporting LPIPS, but no established superiority to
full or visible-preservation selection. The 51/159 fixed-support truth
counterexamples, filtering/ranking decomposition and qualitative losses explain
the limited compatibility assumption. Independent human preference is absent.

Both cohorts pass exact prepared-pixel/scorer/metric/policy replay in isolated
WSL Python 3.10, NumPy 1.23.5, Pillow 9.5.0. Full CPU checks passed 212 tests,
five Bash regressions and smoke. The final report validator and browser checks
pass; all 159 cases and 318 panels are present. Frozen analysis hashes match.
Full/local decisions are unchanged in the separate Windows numerical-portability
audit; one boundary-plus-color pilot decision changes.

Generation and LPIPS jobs 36891/36894/36895/36897/36898 completed. Validation job
36958 failed immediately because scripts/prepare_surface_experiment.py was not
deployed remotely; this is recorded, not hidden. Successful local exact replay
supplies the validation. Cluster diagnostics completed separately and match the
local pinned diagnostic summary. No requested training/inference job is pending.
Original cluster archives and SHA-256 receipts are preserved. Do not duplicate
completed transfers or resume old session IDs in the historical notes below.

Scientific source and both development/test decisions remain frozen. The report
builder, validator, portability audit and delivery packager are reusable; commands
are in cookbooks/surface-consistency.md. Future scorer changes require new
validation/test data and a separate output directory. The original accepted
baseline and unrelated user jobs were preserved.

## Historical Surface-PMC execution notes (superseded by completion above)

Continue theory to explainable measured results within the user's five-hour
window. Goal is active. Protocol: `configs/research/surface-consistency-v1.json` and
`docs/SURFACE_CONSISTENCY_EXPERIMENT.md`. Artifacts:
`output/surface-consistency-20261108/`. Ninety-six prepared cases, 24 development
and 72 test, 24 per source, one target per image identity. Evaluation truth stays
local and separate; the generator receives only `inputs/`, `protocol.json` and
`generation_manifest.json`. Seeds 11,22,33,44; MC 35/50; one-step shared pools.

Implemented `amodal_benchmark/surface.py`, preparation and GPU candidate scripts,
and `scripts/run_surface_slurm.sh`; 11 focused tests passed. Smoke job **36891**
requests four GPUs on debug-gpu, four workers, first four cases and first two
seeds. It runs real CUDA/FFT and pipeline import checks first. Remote/local paths
are both `output/surface-consistency-20261108`; remote logs in `job-logs/`.
Source is unchanged for the accepted baseline. Unrelated user jobs exist and
must be preserved. Next actions are smoke inspection, full shared-pool generation,
development-only normalization freeze, held-out analysis and honest report.

08:03 UTC update: smoke 36891 completed (8/8 valid generated outputs; runtime
checks passed). Full 384-candidate job **36894** runs; LPIPS **36895** waits
after it; replication **36897** waits after metrics. Replication output root is
`output/surface-consistency-replication87-20261108`, 24/24/24/15 new cases by
VG/COCOA/free-images/LAION, all test and disjoint by image ID and bytes. The
96-case attempt exhausted LAION eligibility and is retained incomplete at
`output/surface-consistency-replication-20261108`; it was never generated.
Use `analyze_surface_experiment.py --stage freeze` on the pilot only, then
`--stage test` on the pilot and `--stage test --freeze-from PILOT/analysis/development_freeze.json`
on replication. Test analysis must wait for the freeze. Existing census scoring
is running locally in exec session 70278; its log is `logs/surface-census-20261108.log`.
Full WSL check passed 203 tests and all Bash/smoke checks. Source snapshots and
exact-hash recovered original preparation source are retained under pilot/source/.

08:37 UTC continuation: development was frozen on the cluster at 08:22:29 UTC;
source hashes match locally and the selected exploratory development rule is
`texture`. Pilot test analysis completed at 08:27:47 UTC: 72 cases, 288 generated
outputs, full-surface gain +0.002972 (95% CI -0.004722 to +0.011277), local-seam
+0.011096 (+0.003282 to +0.019391). These are pilot findings only. Original full
pilot archive transfer is still active (Windows SCP PID 68588). Replication
36897 is running, followed by metric job 36898. Exec session **82966** waits for
both, analyzes replication with the frozen pilot lock, packages artifacts and
starts SCP to the replication folder. Do not duplicate these operations.

Added post-hoc explanations without changing frozen science: exact filtering /
ranking / oracle-regret decomposition, within-image pair ordering, and a
fixed-support truth-versus-selected compatibility diagnostic. Declaration is
`posthoc_explanation_declaration.json` (08:30:53 UTC, after pilot errors and before
replication errors). New derivations are in `docs/SURFACE_CONSISTENCY_THEORY.md`.
The independent validator now recomputes every hidden metric, policy and summary
from candidate artifacts, and rejects corrupt or nonfinite saved results.
Full CPU checks are running in `logs/surface-explanation-check-wsl-20261108.log`.
Historical census associations are complete; full-surface AUC is 0.522 on 1,102
assessable valid cases against prior assistant labels, and is descriptive only.

## One-human visual review prepared (2026-11-08)

The user requested a handoff for the report's independent human rater. Preparation is complete at [output/human-review-20261108/README.md](output/human-review-20261108/README.md). The default is a random 160-case audit (eight batches; about 3–6 hours); a full 1,390-case census alternative (70 batches; about 23–46 hours) is also packaged. An optional workload question was asked; no answer was received before preparing both options. Choose one scope before collecting responses.

Both reviewer ZIPs contain an offline HTML form, guide, blank CSV, neutral preview panels and byte-identical native source/visible/completion/raw images. Previous assistant labels, numeric metrics and private identity mappings are excluded. The form requires six ratings plus evidence/confounds, handles U/NA, records independence disclosures, saves drafts and exports/imports JSON. Reviewer data is not uploaded anywhere.

Source helpers: `scripts/prepare_human_review.py`, `scripts/validate_human_review.py`, `scripts/summarize_human_review.py`; instructions: [docs/HUMAN_VISUAL_REVIEW.md](docs/HUMAN_VISUAL_REVIEW.md). The importer writes a new human annotation set, per-criterion counts and human–assistant agreement without overwriting the assistant census. It rejects incomplete assignments unless explicitly asked for an interim summary.

Validation: 15 focused tests passed, including synthetic end-to-end import. Both ZIPs match their exact public manifests; all 800/6,950 images decode; 640/5,560 native evidence hashes match. Browser tests covered offline opening, blank defaults, validation, autosave/reload, import/export and confound handling. Synthetic browser data was cleared. No real human responses, external messages, inference, cluster operation or changed experiment configuration occurred.

Audit ZIP SHA-256: `519ab277cb4a40dbbb8a3742fba65a0b9782219a5456e61de4c3f7e504704cf2` (226,870,269 bytes). Census ZIP SHA-256: `e78b58b30ff903613bb1f1786385438807a9c7a4ed0bde7394fead3e174e34cc` (1,920,708,510 bytes). Actual independent human evaluation remains pending; no human-consensus claim is supported.

## Expanded-vocabulary appearance census completed (2026-11-08)

The user requested the Person 2 deliverable on expanded vocabulary, with all outputs if possible and a three-hour budget. The resulting [13-page PDF](output/appearance-expanded-20261107/REPORT.pdf), [report](output/appearance-expanded-20261107/REPORT.md), [taxonomy](output/appearance-expanded-20261107/TAXONOMY.md) and [filterable census viewer](output/appearance-expanded-20261107/review/index.html) use the recovered `pdmc-open-world-sam-annotated` run. The earlier baseline sample below remains separate.

- All 2,565 records were analyzed; all 1,390 saved outputs received explicit single-assistant visual labels. The 1,175 no-output failures are accounted for separately. The intermediate 552-case sample is archived and superseded by the final census with unit inclusion probabilities.
- At least one observed appearance defect: 652/1,390 (46.9%). Positive/assessable counts: texture 578/1,304; color 398/1,345; pattern 311/601; semantic drift 6/1,350; hidden-region inconsistency 559/1,113; co-occurrence 11/1,199. Maximum severity: 347 mild, 243 moderate, 62 severe. Labels overlap and U/NA do not count as passes.
- All eleven confirmed co-occurrence cases are in raw RGB only; extras are excluded from returned masks. Full-source checks removed six false positives involving existing context, and one ambiguous bus/car-like depiction became unknown. Twenty-four explicit adjudications preserve the initial census labels and evidence. Sixteen output target-switch tags are tracked separately from semantic drift.
- True COCOA occlusion is available for 321 targets, including 181 saved outputs. Requested 20-80% bins contain 114 targets and 71 saved outputs. Availability by bin: 24/30, 10/24, 12/24, 9/15, 13/16, 3/5. No monotonic or causal performance claim is justified. Independent mask decoding reproduces every bin.
- Generated expansion remains a separate diagnostic: Spearman rho=0.297 on 181 paired outputs, median discrepancy 12.21 percentage points. Category tables cover 284 saved-output analysis categories. Hidden RGB ground truth and independent human adjudication are absent.
- Fifteen focused appearance/census tests pass. Artifact/report validation reproduces all metric bins and label counts, passes 9,300 source-artifact hash checks, verifies all 1,390 viewer panels and the 13-page PDF. Browser failure filters return exactly the recorded positive counts. A strict confound-exclusion sensitivity analysis is included.

Final labels: `output/appearance-expanded-20261107/review/annotations.csv`, SHA-256 `ebf4f6b036d8cf592fe115e2e4811da891d29e1fe56a37a3da0ac7090b718062`. New helpers compile explicit batch/adjudication records, build the expanded report, and independently validate its artifacts. Existing generic report rendering now accepts census counts and an explicit footer. Historical execution provenance was not fabricated. No inference, remote operation, scientific configuration change or experiment registration occurred. The requested analysis is complete; independent human review is a follow-up recommendation.

The [portable delivery ZIP](output/appearance-expanded-20261107/appearance-expanded-delivery.zip) contains 1,562 files, including every census panel, and is 169,950,750 bytes. ZIP CRC verification passed; SHA-256 is `ec61b2b0558504b0740f895b1fb00a1b566caf2b023b0b43daa1dcaf2c521797`. The package omits models, original datasets, external annotation JSONs, environments and duplicate contact sheets. Its manifest and receipt are retained beside the archive. Final PDF SHA-256: `989ce2552337741e6c4492d8b6edb7cd4e86b749014d6166f62dc6b9ffe51184`.

## Appearance-failure analysis completed (2026-11-07)

Completed the user's Person 2 deliverable using the accepted baseline, without rerunning inference. Start with [the 12-page PDF](output/appearance-failures-20261107/REPORT.pdf), [full report](output/appearance-failures-20261107/REPORT.md), [taxonomy](output/appearance-failures-20261107/TAXONOMY.md), or [offline audit viewer](output/appearance-failures-20261107/review/index.html).

- All 2,565 targets retain their original recorded scores/status; 951 saved outputs have aligned mask/color diagnostics. The baseline passes full/provenance verification.
- External COCOA annotations uniquely match 321 targets by image identity and exact region name. Independent mask decoding reproduces the supplied occlusion ratios. The requested [20,80)% bins contain 114 targets and 44 saved outputs; 2,244 benchmark targets have no matched ratio.
- A deterministic sample of 120 saved outputs was reviewed by one assistant across eight expansion-proxy strata. Final labels are authoritative in `review/annotations.csv`; initial labels and same-reviewer revisions are retained. This is not independent human consensus.
- Raw positive/assessable counts: texture 54/112; color 51/113; pattern 25/45; semantic drift 4/114; hidden-region inconsistency 59/100; co-occurrence 11/105. Counts overlap. Severity, unknown/NA counts and exploratory stratum-weighted estimates are included.
- Expansion is not true occlusion: Spearman rho=0.133 on 142 matched saved outputs. No monotonic performance trend or hidden-appearance ground-truth claim is supported.
- Existing validation: 167 Python tests, five Bash regressions and CPU smoke passed in isolated WSL CPU tooling. Final report validation reproduces all bin means/denominators, sample identities and raw/weighted taxonomy, verifies local links/images and checks the 12-page PDF. Browser filters and all 120 panels work.

New offline tools are `amodal_benchmark/appearance.py`, `scripts/analyze_appearance_failures.py`, `scripts/audit_appearance_evidence.py`, `scripts/build_appearance_report.py`, and `scripts/validate_appearance_report.py`, with focused regression coverage. The report builder deliberately rejects a different baseline/review identity because its narrative describes this frozen audit. Runtime/model code, accepted artifacts, historical outputs and the registry remain unchanged. The shareable ZIP excludes the validation environment and external annotation archive. No task work remains; independent human review and controlled surface/context experiments are recommendations.

## Full baseline accepted (2026-11-06)

The requested full baseline is complete. Slurm job **31548** completed with exit code 0 in **4h 02m 38s** on four Quadro RTX 6000 GPUs. GPU inventory/FFT, CUDA extension build/forward/backward, full pipeline imports, all source-image checks, the four-worker scored smoke, full evaluation, provenance-required verification and registration all passed. The unchanged scientific contract is `configs/baseline.json` (SAM references, semantic matcher v2, released vocabulary, seed 0).

**2,565 targets: 951 saved completions and 1,614 failures (62.92%).** Saved outputs comprise 796 non-cap termination states and 155 iteration-limited outputs; neither category certifies visual completeness. There are zero missing images and zero reference failures. Overall CLIP is **28.393098 (n=951)**; LPIPS **0.735009**, feature similarity **0.242636** and SSIM **0.303397** each use **n=2,565**, including the configured failure penalties.

Recovered **16,814 files (2,960,571,039 bytes)**. Every file matches a fresh independently computed remote SHA-256, and the recovered run passed local `verify --require-full --require-provenance` against the existing dataset. All 36 captured project source files match the maintained source. See the [accepted report](output/baseline-establishment-20261105/REPORT.md) for per-source results, failure reasons, provenance, recovery evidence and interpretation limits.

- Remote immutable run: `/home/daoqm/Vu_Hung/cv_nangcao/output/pdmc-baseline-provenance-20261105`.
- Recovered immutable run: `output/baseline-31548/output/pdmc-baseline-provenance-20261105`.
- Local copy of the cluster ledger: `output/baseline-31548/experiments.csv`; it is a separate snapshot, not an additional experiment or a replacement for a local ledger.
- Registered entry: `ec823094e31f226957ef0442f0480cf14ae2db91c79550a9701a9e0aef99b07c`.
- Execution: `20261106T053037.969259Z-fdce84ff`; workflow: `20261106T052433Z-c6cfb8b1`.
- `output/baseline-establishment-20261105/status.json`, `watcher-status.json` and `final-acceptance.json` record completion. The watcher and its slow bulk transfer were stopped after successful cluster completion; exact-byte recovery replaced the transfer. No baseline job or watcher needs resuming. Do not submit the finished baseline again.

Historical semantic-v2 comparison: identical scientific/metric configurations, target/input identities and recorded seeds, but six gains and one loss among saved completions; all 945 shared completion PNGs and 32 initial reference masks differ. The historical environment was not preserved, so this is not a causal improvement or exact historical reproduction claim.

The baseline task is finished. The expanded annotated vocabulary was already evaluated on all 2,565 targets: 1,390 saved completions and 1,175 failures, with independently verified recovered artifacts in `output/artifact-recovery-2026-11-05/downloads/pdmc-open-world-sam-annotated`. Reuse those historical results; no routine rerun is needed or planned. Their missing historical execution/source/environment snapshots prevent provenance-required registration and an isolated comparison with the newly validated baseline. A fresh run would only address a separate explicit need for current-runtime provenance. Surface-PMC remains a retained, unimplemented proposal. Earlier job-state entries below are historical and are superseded by this acceptance record.

## Active task: full baseline (2026-11-05)

The user requested a complete baseline with the current provenance workflow and validated GPU runtime, with a three-hour working window beginning about 07:05 UTC. Longer valid inference may continue. This is pretrained PD-MC evaluation, not training.

Prepared [the exact launch and acceptance procedure](output/baseline-establishment-20261105/LAUNCH.md), `scripts/run_baseline_slurm.sh`, `scripts/run_baseline_workflow.py`, `configs/baseline-gpu-smoke.json` and six regression tests. The unchanged full contract is `configs/baseline.json`; new full output is `output/pdmc-baseline-provenance-20261105`, with a separate `-smoke` directory. GPU validation runs both required integration checks and exercises all four devices. Smoke must generate at least one verified completion with all four metrics, then full scoring, provenance-required verification and registration proceed sequentially. There is no session timeout that kills a valid job.

Validation: all 155 Python tests pass under WSL Python 3.10 without skips, all five Bash regressions and synthetic CPU smoke pass. Source-image preflight passes all 2,565 targets with zero missing images. Reviewed 88-file source-only upload excludes AGENTS.md/data/results/notes/ledger; hashes are in `prepared-source-manifest.json`. Windows planning was fixed to retain Linux executable paths and rechecked, including the focused WSL tests.

Read-only SSH confirms other user jobs belong to `playground-series-s6e11`; none were modified. The remote amodal project lacks the new provenance modules. The compute-node cache `/dev/shm/daoqm/amodal-cache` cannot be checked on the login node; direct SSH to gebru is denied by cluster policy. Do not work around that restriction. Validate within the normal submitted allocation. Local GPU is a 6 GB RTX 3060 and cannot support the documented four-worker baseline.

The user subsequently explicitly authorized agent-operated transfers/submissions and requested AGENTS.md updated; this is now persistent project permission and must not be requested again. All 88 source files are uploaded and match local SHA-256. The previous remote source is backed up at `.deployment-backups/baseline-20261105T094541Z.tar.gz` (hash recorded in the launch package). AGENTS.md remains local.

At 10:13 UTC, the persistent runtime is validated on Quadro RTX 6000, including the added real FFT round trip (job 30435, successful). Job 30425 had repaired the cache and passed the original CUDA/import/pinned-package tests. Scored smoke 30428 ran on L40S, wrote seven expected selection-failure records, then failed during LaMa with CUFFT_INTERNAL_ERROR. Isolated FFT diagnostic 30440 reproduced the same error on L40S without loading models. Preserve this failed attempt; its execution/source/environment/artifacts and logs are downloaded under `output/baseline-establishment-20261105/cluster-evidence/`. No scientific settings or pinned package versions were changed to work around the hardware incompatibility.

Added FFT checks to `tests/validate_cuda_extension.sh` and the baseline launcher GPU inventory, reran the six workflow regressions, uploaded both files and verified their hashes. Full baseline 30426 is back in the Quadro-only `debug-gpu` partition, with a valid dependency on new Quadro scored smoke 30436; it no longer depends on failed 30428. Smoke 30436 waits for one full GPU, and the full baseline waits for it plus four full GPUs. The full job retains four workers, 16 CPUs, 64G, 24 hours, its own eight-target/four-worker smoke, and provenance verification/registration gates. Leave these valid queued jobs in place. No full baseline is accepted yet. Current machine-readable state and all receipts are in the launch package.

Prior job 30424 failed immediately because the `/dev/shm/daoqm/amodal-cache/conda-env/bin/python` executable was absent. Pending job 30423 was cancelled and replaced to use the persistent cache; no running inference was stopped. A rejected CRLF stdin submission created no job; subsequent sbatch scripts are sent as LF bytes. Receipts, scripts and status are under `output/baseline-establishment-20261105/`. Monitor `logs/pdmc-runtime-30425.out`, `logs/pdmc-runtime-smoke-30428.out` and `logs/pdmc-baseline-30426.out`. Preserve failed attempts, update dependencies if repair is needed, and do not edit scientific settings or import historical results. Earlier sections describe previous tasks and their deployment restrictions are superseded by the current AGENTS.md.

## Experiment artifacts recovered and verified (2026-11-05)

The user's one-hour recovery task completed in 57.1 minutes. Recovered 70,012 files (11.90 GB) into `output/artifact-recovery-2026-11-05/downloads/`; every file matches an independent remote SHA-256. See [the recovery report](output/artifact-recovery-2026-11-05/REPORT.md).

Recovered semantic v1/v2, annotated, paper-v2 and shared historical LISA references. Six full historical runs and both 20-target runs pass CPU artifact verification. The three older LISA reports use separate `legacy-verification/` views with documented metadata additions; original reports, target records, scores and references are unchanged. These views use hard-linked targets and must remain read-only.

An additional 275 previously absent target records, including 82 successful completions and their references, were recovered through the documented older SAM-to-bgfix reuse chain. This isolated 295-target archive is unscored; no original learned-metric caches were synthesized.

The user explicitly authorized downloading the listed cluster folders for this task. Remote reads used shared run locks; no remote files were written and no jobs were submitted. Slow bulk streaming was replaced by local hash deduplication and exact PNG reconstruction from recorded masks/RGB using the pinned CPU environment. All 23,724 original experiment files and the maintained-source manifests remain unchanged.

New paired artifact evidence: all 946 semantic-v2 successes remain saved in annotated, with 444 additional successes. Exactly 941 shared completion PNGs are byte-identical; five completion changes match the five changed metric tuples. Reference artifacts change for 622 targets. This strengthens the historical coverage finding without establishing a pure generator improvement.

Historical execution/source/environment provenance was absent from the recovered run directories and was not fabricated. No GPU inference, learned-metric recomputation or experiment registration was performed. The next research gate remains a new provenance-backed baseline. Original report-only folders remain unchanged; use the recovered paths and reference directory documented in the report. Earlier statements that artifacts were missing describe the pre-recovery state.

## Manager review and presentation

Completed the requested team review, building on the eight-report audit and fresh CPU checks. Deliverables: [16-slide PDF](output/manager-briefing-2026-11-05/presentation.pdf), [offline presentation](output/manager-briefing-2026-11-05/presentation.html), and [speaker notes / complete results / evidence index](output/manager-briefing-2026-11-05/briefing.md).

The paired historical comparison is the central finding: annotated vocabulary recovered 444 outputs relative to semantic v2, lost none, and retained identical reported metrics for 941 of 946 shared outputs. Aggregate improvement primarily reflects coverage recovery under failure penalties. Surface-PMC remains unimplemented and unvalidated. Semantic v1/v2 and annotated detailed artifacts are absent locally, so report arithmetic is verified but full image-level audit remains pending. SAM-bgfix passed the fresh artifact verifier, with historical execution provenance absent. No local experiment ledger exists.

The proposed next gates are artifact recovery, a provenance-backed baseline under the current workflow, and a shared-candidate boundary-scoring experiment. These are recommendations, not completed experiments or newly authorized remote work. All 16 slide layouts fit, all four embedded example images load, local briefing links resolve, and the exported PDF has 16 pages. The 59-file maintained source/config/test/helper manifest remained unchanged. No GPU run, remote operation or registration was performed.

Earlier handoffs follow; their task descriptions are historical.

## Current task and result

The user asked to read `ChatGPT - Assess Reproduction Feasibility.html`, understand specifically Surface-Consistent PMC, and retain the knowledge for future questions. Completed as a documentation task.

- Read [docs/SURFACE_CONSISTENT_PMC.md](docs/SURFACE_CONSISTENT_PMC.md) for the research question, equations, example boundary scorer, integration, evaluations, limitations, optional extensions and unresolved choices.
- Read [the focused source extract](docs/research/SURFACE_CONSISTENT_PMC_SOURCE.md) for original Vietnamese wording and pseudocode. The original HTML remains unchanged; SHA-256 is `a398b6d068bd7abb2e7aee9453c6796f2c486d9b5143624b2c0112f624eeed3f`.
- Local `AGENTS.md` instructs future work to read the reference when answering Surface-PMC questions. Durable recall depends on access to the project files.
- Surface-PMC is an unvalidated proposal: sample K PMC completions and minimize boundary/color/texture inconsistency relative to visible object appearance. No code or configuration implements it as a result of this task, and no improvement was measured.
- Preserve the distinction between pseudo-occlusion hidden-ground-truth evaluation and this project's SAM/LISA visible-reference protocols. Further implementation requires explicit coordinate handling, original-visible anchoring, valid-candidate fallback and a fair sampling-budget control.

The baseline/annotated evaluation plan is unchanged. No remote operations or GPU runs were performed. Earlier repository-organization work and its validation are retained below.

---

# Previous session handoff: repository organization

Updated 2026-11-05.

## Current task and result

Review and organize the codebase, cleaning unnecessary files with extreme caution. Conservative cleanup is complete. Scientific source, entry points, configurations, tests, original papers/release ZIP, datasets, inputs, outputs and the existing environment are preserved.

Documentation starts at [docs/README.md](docs/README.md), with a [codebase map](docs/CODEBASE.md) and [test guide](tests/README.md). The root README links these guides. The paper summary moved to `docs/PAPER_SUMMARY.md` with only its relative link corrected. Both completed plans moved from `docs/superpowers/plans/` to `docs/archive/plans/` with identical bytes. The historical root `run_amodal.log` moved to `logs/run_amodal.log` with identical bytes.

## Cleanup and rollback

Removed from the workspace: three source/test `__pycache__` directories (76 bytecode files), generated `amodal_benchmark.egg-info` (six files), and the empty former `docs/superpowers/` tree. The generated files total 939,305 bytes. Automatic approval review blocked permanent deletion with the generic reason "blocked by policy"; all these items were instead moved into a rollback directory. Nothing was permanently deleted.

Rollback location: `C:\Users\daovu\AppData\Local\Temp\cv-nangcao-cleanup-20261105-0yh43rr4`.

- `source/` and `source-manifest.json` preserve the 84 original source/documentation files and their hashes.
- `run_amodal.log` preserves the original root log; `moves.json` records all four file relocations.
- `generated/`, `generated-manifest.json` and `removed-from-workspace/` preserve the generated files, sizes and hashes. The latter also contains the empty old documentation tree.
- `protected-inventory.json` records paths, sizes and modification times for all 29,318 protected files. It is an inventory, not a backup of the dataset or environment.

To undo a move, first check that its original destination is absent, then move the file back using `moves.json`. Restore the paper summary from `source/paper_summary.md` to retain its original relative link. Restore edited README/research notes from `source/` only after checking for later edits. Generated caches need no restoration; the normal Python/install workflow recreates them. Keep the rollback directory until the organization is accepted.

## Validation

- Before cleanup, Windows Python 3.14 ran all 149 tests successfully with nine expected Linux-only skips: `logs/cleanup-baseline-windows-2026-11-05.log`.
- After cleanup, WSL Python 3.10.22 ran all 149 tests without skips, all five Bash regressions and synthetic CPU smoke: `logs/cleanup-check-wsl-2026-11-05.log`.
- Installed Windows `amodal` CLI and CPU smoke passed after root egg-info/cache removal. The editable environment's own distribution metadata remains installed.
- Source/result transfer helpers passed `-WhatIf`; no host was contacted. All 123 local links checked in the root README, documentation tree and test guide resolved.
- SHA-256 checks confirm all 63 original code/configuration/test files are unchanged. Protected paths, sizes and modification times match the pre-cleanup inventory; this did not re-hash every historical image or result.

No GPU code or runtime was changed. No GPU validation, remote transfer, Slurm submission or experiment registration was performed. The research next steps below remain applicable.

---

# Previous session handoff

Updated 2026-11-04. Preserved from the preceding rebuild.

## Task

Adapt the codebase to `SETUP_NEW_COMPETITION.md` within the user's one-hour budget, retaining only the workflow needed for this amodal research project.

## Current state

Implementation, documentation and local validation are complete. No remote transfer, Slurm submission, GPU inference or experiment registration has been performed.

## Constraints

- Preserve dataset bytes, scientific protocols, pinned GPU dependencies, prior outputs and resume compatibility.
- The user performs SSH/SCP/Slurm operations. Keep `AGENTS.md` local.
- Full cluster runs request four GPUs before the script name and four workers after it.
- New scientific settings require a new output directory. Historical CLI-only runs remain supported.

## Checks and evidence

- Linux/WSL Python 3.10.22, NumPy 1.23.5 and Pillow 9.5.0: 149 tests passed, all five Bash regression scripts passed, and synthetic CPU smoke passed (`logs/check-py310-final.log`). This is CPU compatibility coverage using the runtime's Python/NumPy/Pillow versions, not a GPU execution.
- Windows Python 3.14.6: 149 tests ran successfully, with nine expected Linux-only skips (`logs/tests-windows-final.log`).
- WSL Python 3.14 also passed its 147-test suite before the final two protocol regression cases were added (`logs/check-wsl-final.log`).
- The installed `amodal` command and smoke work from outside the repository in the Python 3.10 environment.
- Source-image preflight passed for all 2,565 targets; configured two-target SAM preflight passed.
- Transfer helpers were exercised with `-WhatIf`; neither contacted the cluster.
- The printed baseline plan requests `--gres=gpu:4` and `--benchmark-workers 4` in the correct positions; its CPU allocation is also forwarded to the runner's thread budget.
- The final reference-protocol preflight regression passed (`logs/protocol-preflight-final.log`).
- After the final planner CPU-budget fix, all 25 focused workflow tests passed again (`logs/workflow-py310-final.log`).
- Earlier logs retain intermediate fixture/environment failures. Use the final logs above for the delivered state; the shell fixture's temporary runtime was corrected before the final full checks.

## Environment locations

The local editable CPU installation is `.venv` on Windows. WSL verification used `/home/hungvudao/.cache/cv-nangcao-cpu` (Python 3.14) and a separate `/home/hungvudao/.cache/cv-nangcao-py310-rebuild` (Python 3.10). No GPU environment was upgraded.

## Next action

Follow the cluster section of `README.md`: preview and upload source, run GPU extension/import checks inside an allocation, then a separate two-target smoke and full baseline. Reuse the existing remote dataset/cache. Verify and register the full result before comparing the annotated-vocabulary experiment.


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

## Approved failed-transfer cleanup (2026-11-16)

User approved removal of the failed transfers from group B of docs/OUTPUT_CLEANUP_REVIEW_20261116.md. Removed exactly 10 reviewed files (396,648,697 bytes; 378.27 MiB), including the truncated baseline transfer, abandoned partial transfers and redundant transfer chunks. Checked absolute paths and regular-file status, retained SHA-256 hashes, and verified all deleted paths absent and required recovered evidence present. Completed archives, actual results, logs, recovery receipts, validation environments, and parent directories were retained. No cluster access occurred. Exact receipt: docs/OUTPUT_CLEANUP_RECEIPT_20261116.json. Groups A and C remain unapproved proposals.
## Shape-Guided PMC on four L40S GPUs · 21 November 2026

The controlled study and figure are in [the report](output/shape-20261121/REPORT.md). The exact chat section is “2. Người 1 — Shape-Guided PMC” in the root HTML. Week 8 slides 11, 14, 15, 16, 17 and 18 identify the missing method, joint selector, full metrics, occlusion analysis, qualitative examples and ablation. Jobs 44902/44903/44926 completed training, controlled inference and native replay. Job 44925 was a one-second no-op and produced no evidence.

The rejected historical-reference attempt was 44905. Its dependent jobs did not run. New Slurm cohort jobs are 44913 (PMC), 44914 (Shape), 44915 (Surface) and 44917 (combined). Full jobs are 44919, 44920, 44921 and 44923 in that order. CPU comparison jobs are 44916/44918 and 44922/44924. Jobs 44929 and 44930 build the cohort and full reports. Exact submissions are in the remote `output/shape-20261121/progressive-l40s/submissions.tsv`. Each GPU submission uses `-p gpu --gres=gpu:nvidia_l40s:4` and four workers. Inspect sacct and native artifacts before interpreting full results.

The user then directed the study to reuse all completed controls. Slurm confirmed zero allocated seconds for the canceled Surface cohort (44915), full PMC (44919) and full Surface (44921). The completed 204-target PMC snapshot (44913, 1,264 seconds on four L40S GPUs, 86 saved) supplies the cohort comparison. The full controls come from the registered Quadro runs of 14 November. CPU job 44942 compared the 204 pairs of native reference masks and found 38 changed masks, 212 changed pixels and zero changed crop geometries.

The only current GPU submissions are Shape cohort 44914, combined cohort 44917, Shape full 44946 and combined full 44948. The cohort results are reused in their respective full continuations. Full Shape generates the remaining SAM references and full combined receives verified copies from it. CPU jobs 44943/44944 audit the cohorts, 44945 writes the cohort report, 44947/44949 audit the full arms and 44950 writes the full report. The prior queue receipts are archived in `progressive-l40s/submissions-before-historical-reuse.tsv`. Full comparisons report the two GPU types, the exact-reference subset and failure-inclusive metrics. Keep the strict same-device Shape/combined reference checks. This amendment predates any natural-image Shape score.

The first completed natural passes for VG/285758_cat and VG/61549_elephant passed independent replay of eight candidate seeds, native prior-overlap scores, selected index and RGB bytes. A separate projection using the completion PNG alpha channel reproduced the original-frame polygon IoU for all six saved L40S PMC outputs among the cohort's 22 COCOA joins. All six crops extended beyond the original frame. The 16 remaining joins had failed completion and received zero. These checks validate the measurement and selector implementation. Cohort quality estimates require the full frozen inventory.

CPU job 44951 completed the supplemental hidden-core DreamSim measurement on the existing 128 test objects. Local and remote SHA-256 for `hidden-dreamsim.json` is `856f65bdd13c2c34591bf8d32a72e0927608cfd4c7bf260442781d875cde9f8d`. The [report](output/shape-20261121/REPORT.md) distinguishes this masked endpoint from the original object DreamSim score and includes text for slides 19–20. The 2,565-target severity report will use the same five bins for LPIPS and externally annotated amodal IoU.


2026-11-21 · Historical geometry check for Shape transfer. CPU jobs 44959 (13 seconds) and 44960 (10 seconds) ran via srun -p cpu -c2 --mem=3G --time=00:10:00 in ~/Vu_Hung/cv_nangcao using .amodal-cache/surface-metrics-env/bin/python. The first projected historical PMC and Surface completion alpha masks into the 321 COCOA frames and saved output/shape-20261121/cocoa-historical-controls.json (SHA-256 bc450f0f0de6d64b1053ba304985474413f702c3d71abf366c1d31f8d7bcfe3a). The second verified all 642 scores with scripts/compare_shape_progressive.py. Both historical aggregate verification receipts match the November 14 archive byte-for-byte. The cluster and local report normalize to SHA-256 b306c0d4baaabe77254c84ebc19f518d42eacfe11ca7b947c3ae7bf0e33e0af3 before cohort completion.

2026-11-21 · Slurm 44914 and 44943 completed. Retrieved the verified 204-target Shape report, pairs and summary via python scripts/collect_shape_results.py. SHA-256 d86e7e0eb46bda94468e49edd8eb8ce19782b4a6db09b0e3b7abb42878808f36 for the pairs and a60aa6465ac38a67d778b88eb0524597307832ff8a8d0793c75714c1df372a31 for the summary. The two verification receipts agree on target, input and reference identities. Local bootstrap replay passed with 5,000 draws clustered by image and stratified by source. Slurm confirmed combined cohort 44917 running. For its first bicycle and white-house pools, CPU replay reconstructed the selected corrected RGB byte for byte; each shared all eight raw first-pass candidates with its corresponding Shape pool. Cohort comparisons for combined and all full results remain pending.

The two 204-target coverage transitions are rendered by scripts/render_shape_examples.py --natural-cohort. The output natural-cohort-coverage.png has SHA-256 bb8dd358c2f14fdd64cb3910c77de2a17c1f94d76bd38b13807965176ad24fb1. Native alpha/COCOA/SAM intersections explain the dog example in discoveries.md and the Week 8 report. The figure reuses the completed PMC and Shape snapshots.


Ran the local CPU audit with `python scripts/audit_shape_unseen.py` using the four original case manifests, the archived unseen-inventory hash, official target annotations and native image bytes. The audit receipt has SHA-256 1a95ec12e41bd88614b9611065a7db863e02467e220c9d49ddd4e0be31ef2283. Uploaded that receipt and the updated report builder to the existing remote study. Rebuilt with `.amodal-cache/surface-metrics-env/bin/python scripts/build_shape_report.py --accounting`. Remote and collected local REPORT.md had SHA-256 47aa23040779c99a58c8f0bf471e3705e56b14eb9867d2e1c9c01124b27a6edf at this stage. The frozen cohort and full inventories remain in use. Slurm reported 44917 running at 01:10:11 with 162/204 native target records. Both full continuations were pending dependencies.


2026-11-21 UTC. Verified and collected the cohort coverage breakdown in REPORT.md (SHA-256 21b537cdc13289fea87339d634010f0b0a1d638ac462b56bffdb40cb6d55ff08). The WSL CPU environment passed scripts/check.sh including the new coverage-denominator regression. Slurm last confirmed 44917 RUNNING at 01:38:42 with 198/204 result records, 145 saved candidate pools and continued diffusion progress from worker 01. Jobs 44944, 44945, 44946 and 44948 were pending dependencies. SSH subsequently lost DNS resolution for slurm.csse.rose-hulman.edu; local GlobalProtect adapter was disabled. Collector PID 18524 remained live and logged its retry at 15:17:48 UTC. Await restored cluster access for current state, audit and full continuation evidence.


2026-11-21 UTC. Local replay verified the saved controlled-review archive hash, the frozen 128-image inventory, all per-case result bytes and the Shape, Surface and joint indices against the archived ranking inputs. Added the K=4 weighted row and the 105-disagreement breakdown to the local Week 8 report and builder. Rendering the controlled sections in memory reproduced the edited report. Cluster DNS was still unavailable on the next SSH attempt. On reconnection, check the current remote builder (last known SHA-256 c4ead5af4e13475235d682602c7a6adbf0128242039c3c59e9e352cd6310e6cd), deploy the updated scripts/build_shape_report.py (421f4cb5b322e92ed7fbaf6656679fa9827cf8f0fbb04be87f9ca902180c4a86), rebuild with --accounting and collect the authoritative report along with any newly finished audits. GPU jobs and scientific configurations were untouched.


Cluster access restored. sacct confirms 44917 COMPLETED at 01:49:59, 44944 COMPLETED at 00:03:36 and 44945 COMPLETED at 00:00:01. Used python scripts/collect_shape_results.py and verified the combined pairs (SHA-256 a77bb86a950f3120631482b4eb8ac10fbfe67b1908478612e3447c400aa55633), summary (1bdaa53dc3381104754d97d787066d56d5b05750a1b0ce01a32811c5f85b2c09) and native/provenance receipt (63329e763f33d5c4379db02633f237f87aba6bf94161e581e85f129975912128). The combined CPU audit replayed 150 eight-seed pools with one fallback. Uploaded updated report/figure builders and local CI replay receipt; CPU job 45051 ran srun -p cpu -c2 --mem=3G --time=00:10:00 -J shape-cohort-figure .amodal-cache/surface-metrics-env/bin/python scripts/render_shape_examples.py --natural-cohort. Rebuilt with scripts/build_shape_report.py --accounting. The collected four-column figure and report matched remote SHA-256 7226ac3fa3d23bde96af9255852d2af349c40b236ca02c039194ecb59e5df14f and b7f8b3fe1cc481398cb779c9bd297e887c19801361f6b994721c8cc3a0542f80. Full Shape job 44946 was RUNNING at 00:44:21 with 276 of 2,565 result records including the cohort. Full combined 44948 and CPU audits 44947/44949/44950 were pending dependencies.

All 204 native Shape cohort result files in the running full directory matched the snapshot execution hashes at 00:45:10. Both run/reference configurations agreed. The two-case figure shows the combined detector retained mostly the doughnut-shaped object at the dog target. Updated the Vietnamese caption accordingly; retrieved final cohort report SHA-256 170c0304937ad0e5d28f1fd388ff691effb1e8907038aed692e17fcb7682b74e. At 15:36:49 UTC, the collector found 278 Shape records including the preserved cohort.

For Week 8 slide 16, the 321 archived COCOA targets gave historical PMC/Surface saved counts of 107/109, 34/33, 23/23, 17/16 and 2/2 across increasing occlusion bins (group sizes 203, 54, 39, 21 and 4). In the last group, a plate accounted for 0.2042 of the 0.2075 mean amodal-IoU decrease. The full report builder now exports per-method saved counts with the failure-inclusive means. The local replay reproduced all ten historical bin IoUs within 1e-12, and all nine Shape tests passed. CPU job 45052 rebuilt the report in one second using `sbatch -p cpu --cpus-per-task=2 --mem=3G --time=15 -J shape-slide16-report -o output/shape-20261121/progressive-l40s/slide16-report-%j.log --wrap '.amodal-cache/surface-metrics-env/bin/python scripts/build_shape_report.py --accounting'`. The recovered report SHA-256 is ea94ffccb4753eaffd70462d4a98aeb37d4729a4cb4f9e51c4b07fbc0b8f569e. A direct read of the four archived configurations confirmed identical non-method configuration, package inventories, and metric settings. The two archived Surface policies also matched.

The full report will print unseen LPIPS for each L40S arm versus both historical controls and the direct same-device comparison. The latter verifies the frozen inventory against the source-image audit and checks each method mean against its full audit. A temporary end-to-end rendering with the archived Quadro rows in the method columns reproduced the published historical unseen gain +0.801% [0.396%, 1.179%], all four comparator labels and all five severity-bin coverage pairs. These were reporting fixtures only. All nine Shape regression tests passed. The deployed report builder SHA-256 is f9e9a489ce99585f71f0bedd1b5770d5e73b5d4e02eb4d6570b0fdcc28320afd; existing Slurm audits 44947 and 44950 will use it.

Ran `bash scripts/check.sh` on the current worktree in the WSL CPU environment. Unittest discovered 286 tests (280 passed, six optional SciPy/scikit-learn cases skipped), all five Bash regressions and the CPU smoke passed. Re-ran those exact six cases with the existing WSL surface-replay-py310 environment; all six passed. All 20 relative file links in the collected report resolved. The completed checks cover the current code and reporting changes. Full L40S outcome verification is pending.

Week 8 source-level LPIPS table. Local target-multiplicity resampling reproduced all 12 new cohort intervals to report precision. The same calculation reproduced the four archived full Surface-versus-PMC source intervals. Ten focused Shape tests passed. CPU job 45055 ran `sbatch -p cpu --cpus-per-task=2 --mem=3G --time=00:10:00 -J shape-source-report -o output/shape-20261121/progressive-l40s/source-report-%j.log --wrap '.amodal-cache/surface-metrics-env/bin/python scripts/build_shape_report.py --accounting'` in two seconds. The collected and remote report hashes agree at 5935ec23b115cfbf8518f3455eeaf9cdbf1b7ff2dc4f24fd1dc59a58f648544b. The table was the only report change.

For scheduling only, the four full shards contain 598, 582, 599 and 582 fresh targets after excluding the 204 saved records. The historical Surface run used 677, 630, 724 and 672 eight-candidate pools on those targets. An exploratory timing projection fitted each L40S cohort's successive per-worker record times to a per-target intercept, first pool and later pools, using the archived Surface pool counts as a proxy. The slowest shard projected to about 24.2 h for Shape and 26.4 h for combined before final scoring. Newly completed Shape prefixes took 73-75 min each versus 55-69 min projected. Actual pass counts, runtimes, queueing and scoring determine finish time. Both allocations have a 24 h limit. Job 44946 was still RUNNING with 325 records at 01:18:31; 44948 was dependency-pending.

The local collector now uses the latest submission for each role and stage when detecting a terminal failure, while retaining all attempts in its Slurm accounting. After a confirmed timeout and a recorded continuation, an earlier timeout or canceled dependency can no longer stop the relaunched collector. The existing live collector should finish its current watch normally or signal the first genuine failure. A read-only probe of the active study matched all 12 collected files; all 11 focused Shape tests passed, including a simulated Slurm timeout and replacement.

Read-only continuation check on 325 atomically written Shape records. CPU job 45060 ran `srun -p cpu -c 1 --mem=2G --time=00:10:00 -J shape-resume-check .amodal-cache/surface-metrics-env/bin/python -` in 10 seconds. Every record matched the verified historical PMC target ID, source, label, file paths, original image ID, input hash and seed. The CPU check rehashed all 121 newly processed source files and 975 native reference files, verified each saved completion through the standard resume reader and reproduced the frozen hashes of all 204 cohort records. Run, reference and metric configurations were byte-identical to the Shape cohort. SHA-256 of the sorted JSON mapping from these 325 target IDs to record digests was 2c7ad006b99bc460e0fc6f0677c7bed304bebbc489d20cde21d430c1d1d812f0. This was a partial integrity check; the full score, verification and registration still await completion. Slurm then reported 44946 active at 01:22:09 with 326 records, four active worker logs, and 44948 dependency-pending.

2026-11-21 UTC. CPU job 45069 rebuilt the slide 15 coverage table with the direct Shape + Surface versus Shape comparison. Submitted from the remote project with `sbatch --parsable -p cpu --cpus-per-task=2 --mem=3G --time=00:10:00 -J shape-coverage -o output/shape-20261121/progressive-l40s/coverage-report-%j.log --wrap '.amodal-cache/surface-metrics-env/bin/python scripts/build_shape_report.py --accounting'`. It completed in one second. Local and remote REPORT.md SHA-256 matched at 675e82a0cf9bd0749ad8252f48c89c78e8e78459f348709719d408765805ca7d. An independent join of the two digest-verified cohort pair files reproduced all contributions. The 11 focused Shape tests passed. The collector observed 332/2,565 Shape records at 16:21 UTC. Slurm showed Shape 44946 running and combined 44948 pending its dependency.

The full comparison now checks the direct Shape/combined source hashes, target seeds and scoring-reference equality flags. It recomputes first-pass counts from the combined paired records and reports one-sided sampling or differing candidate pools if they occur. All 204 cohort rows agreed. Of 87 identical first-pass pools, 49 selections differed and 38 agreed. The other 117 targets never reached sampling. All 11 focused tests passed in WSL. A local in-memory rendering using live cohort Slurm accounting reproduced the existing REPORT.md SHA-256 675e82a0cf9bd0749ad8252f48c89c78e8e78459f348709719d408765805ca7d byte for byte. Deployed only scripts/build_shape_report.py (SHA-256 6b32c5eda21b98ec301350d421f571c7fe9f743c63cba67d1edf196e1fda7e1c) for the already scheduled audits. Slurm showed 44946 RUNNING at 01:35:22 with 338 records and 44948 dependency-pending.

Read-only score check while 44946 was running. Compared each registered Quadro control's full per_target.csv against its frozen cohort CSV for all 204 target IDs, after validating all four CSV hashes against their execution receipts. All columns matched for PMC and Surface. Independently matched all 204 rows per L40S arm against each method's native result status, target seed and metric cache or failure penalties. Shape retained all 344 measured values from its 86 saved cohort outputs, and combined retained all 348 from its 87 saved outputs. Full-arm score tables and registration still require the final evaluations.

The slide 18 report now includes the controlled contrast of adding Shape to Surface. Local resampling with image multiplicities independently reproduced all four endpoints and their nominal absolute and relative intervals from the original 128 case records and ordered manifest. The audit hash identified the published numerical summary. The report change is one paragraph. Eleven WSL Shape tests passed. CPU job 45078 completed in one second after submission with `sbatch --parsable -p cpu --cpus-per-task=2 --mem=3G --time=00:10:00 -J shape-ablation-report -o output/shape-20261121/progressive-l40s/ablation-report-%j.log --wrap '.amodal-cache/surface-metrics-env/bin/python scripts/build_shape_report.py --accounting'`. The collected report matched the cluster SHA-256 6730111e1091eccf660e85d0cec411862b8f2bbcd3849d294e1d6c847a0ce0e0. The collector observed 350/2,565 Shape records at 16:36 UTC.

The scheduled full audits now check every frozen cohort metric against the scored per-target CSV saved in that arm's cohort execution. They verify both CSV hashes against the execution receipts, join by target ID, preserve missing CLIP values and accept equivalent numeric formatting. This complements the native result hash and resume-inventory checks. The regression caught a changed cohort LPIPS even when the full table had its own valid receipt and the native result hash was identical. All 11 focused WSL tests passed. On the cluster, the check reproduced all 204 scored rows and all 204 native result hashes in each registered historical control. The existing 204-target Shape and combined contrasts against historical Surface replayed identically, including their paired intervals. Deployed scripts/compare_shape_progressive.py (SHA-256 e0e0a9f1f0139c42774073b4532c297a74cabb0ccbda7fe316c6f8071d3bac36) for audits 44947/44949. Shape job 44946 remained running at 01:51:13 with 355 records. Combined 44948 was dependency-pending.

2026-11-21 UTC. Retired the duplicate Windows collector after checking both command lines and the launcher/worker parent relationship. Stopping worker 17344 also closed its launcher 18524. Rechecked WMI and found only the replacement Python watcher 23944. Its log at output/shape-20261121/collector-scores.log advanced from 362 to 369 Shape records at 16:49:36 UTC, including the frozen 204. Slurm showed job 44946 running on four L40S GPUs and 44948 pending its dependency. No new cluster jobs were submitted.

Prepared the final Vietnamese text for slides 19–20 in scripts/build_shape_report.py. It takes the all-target LPIPS, unseen LPIPS, COCOA polygon IoU and Slurm cost from the full paired analyses; the report emits this text only after both full arm audits are present. A regression checks the sign and endpoint order when expressing LPIPS gain as percentage change and rejects a cohort denominator. The WSL scripts/check.sh checks passed, as did all 13 focused Shape tests. An in-memory build on the actual cohort evidence and Slurm elapsed seconds reproduced the existing report byte for byte (6730111e1091eccf660e85d0cec411862b8f2bbcd3849d294e1d6c847a0ce0e0). Deployed the report builder with scp to the existing remote scripts/ directory; local and remote SHA-256 fc19df4b4fbe9fd9833bd0a6446eb1c99e3c0bc57666c13e55e5a74b38043656. CPU report job 44950 will run after the two full audits.

2026-11-21 UTC. CPU job 45079 scored the fixed candidate selections on the 32 controlled images that also have unique COCOA polygons. Submitted from the remote project via `sbatch --parsable -p cpu -c 2 --mem=3G --time=00:10:00 -J shape-cocoa-panel -o output/shape-20261121/progressive-l40s/controlled-cocoa-%j.log --wrap '.amodal-cache/surface-metrics-env/bin/python scripts/audit_shape_controlled_cocoa.py'`. Slurm reported COMPLETED in three seconds. The recovered receipt output/shape-20261121/controlled-cocoa.json matched remote SHA-256 c16e1d4752fc4ccb99b01a84dc8a730aafc39045846aee231a82e3f551849f89. Verified all 32 original SAM reference digests and reproduced the saved SAM crops locally. Streamed the 256 archived mask PNGs, checked their digests and reproduced every amodal IoU using clipped source rectangles, all 160 selected values and all four paired intervals. No GPU allocations or new generation were used for this analysis.

CPU job 45080 rebuilt the report in one second via `sbatch --parsable -p cpu -c 2 --mem=3G --time=00:10:00 -J shape-cocoa-report -o output/shape-20261121/progressive-l40s/controlled-cocoa-report-%j.log --wrap '.amodal-cache/surface-metrics-env/bin/python scripts/build_shape_report.py --accounting'`. The cluster, independent local in-memory rendering and collected REPORT.md agree at SHA-256 518d983833de0ab10f2cddeb8062ac2612f62b5c963cbfecd49d95f2b7e5e467. Added the secondary polygon table and Vietnamese limitation. WSL scripts/check.sh passed all 291 Python tests, Bash regressions and CPU smoke. Collector 23944 recovered the report and observed 374/2,565 Shape records and 204 combined records at 16:58:46 UTC. Slurm last reported 44946 RUNNING and 44948 dependency-pending.

2026-11-21 UTC. The full Shape audit (44947) can now publish the slide 16 severity tables as soon as the Shape scores are verified. Its report uses all 321 COCOA joins and both registered historical controls. It names the pending combined method. The final report (44950) fills its column and exports the figure and CSV after both full audits. In a temporary historical fixture, the first-arm report ran through its normal entry point. The five coverage pairs and ten historical polygon means matched the archived controls, and the four-method export produced a valid 1824×912 PNG. The WSL replay environment lacks Matplotlib, so the export check used native Windows Python; the cluster reporting environment separately loaded Matplotlib 3.10.8 with Agg. WSL scripts/check.sh passed 292 Python tests, the Bash regressions and CPU smoke. The current cohort-only evidence still rendered the collected report byte for byte. Transferred only scripts/build_shape_report.py via scp to the remote scripts directory; local and remote SHA-256 608d56ea98b9bb7a8c3d3b5929fc2e34334978a250ad1ad1cd2dc79f42ca7231. The direct cluster check found 405 Shape and 204 combined records. Job 44946 was running and 44948 dependency-pending. No jobs were submitted.

2026-11-21 UTC. Replayed the 204 historical Quadro and L40S PMC score pairs against their archived score-table and reference-audit hashes. The only difference in coverage was COCOA train image 108113 (dog). Its LPIPS changed from 0.208432 to the 1.0 failure penalty. This contributed 0.00388023 of the 0.00399675 all-target mean difference (97.1%). The 86 shared successes contributed 0.00011651. The dog's references differed by one pixel and its crop geometry matched. The two runs do not isolate the cause of the missed detection. Added the case to the existing report with links to both score tables and the reference audit.

Transferred scripts/build_shape_report.py with scp (SHA-256 bff6c84b20dbbbcbf648102adba0b024398d932b8929b23f116e32664a819657). CPU job 45101 completed in one second via `sbatch --parsable -p cpu -c 2 --mem=3G --time=00:10:00 -J shape-cohort-report -o output/shape-20261121/progressive-l40s/device-cohort-report-%j.log --wrap='.amodal-cache/surface-metrics-env/bin/python scripts/build_shape_report.py --accounting'`. The collected report SHA-256 d9ed280e100bb75050a046c38531bc07262fa6349e61ce162c9d497460466796 matched an independent local rendering; all earlier report paragraphs matched byte for byte. WSL scripts/check.sh passed in the surface-replay-py310 environment. Slurm showed Shape 44946 running, with 433 of 2,565 records on disk, and combined 44948 waiting with its 204 cohort records already on disk. No new GPU allocation was submitted.

2026-11-21 UTC. Rechecked section 2.3 A1 of the saved Shape proposal. The completed Surface confirmation panel uses pre-occlusion SAM masks. To examine the source's fully visible, untruncated-object criterion, checked the 32 independently matched COCOA polygons. Nine have zero annotated occlusion and identical visible/amodal rasterizations. Seven of the nine touch the image border. Two have a clear border and are completely contained in the input crop. The report gives the bowl and cat scores at K=4 and K=8 separately, including the frozen Surface and combined selectors. The 128-image primary analysis, 32-image polygon comparison and natural cohorts retain their original inventories.

An independent NumPy pixel-edge and crop-containment calculation checked all 32 mask hashes and recovered both eligible IDs. The selected and uniform polygon IoUs match the archived candidate scores. Added a regression for all four image edges, crop bounds and a hidden pixel. WSL scripts/check.sh passed in the surface-replay-py310 environment. Deployed scripts/build_shape_report.py (195fce82ac5d7cf1c532e1707d459a93007b5a51a3fd127eb9cf84718218acf7) via scp. CPU report job 45102 completed in one second via `sbatch --parsable -p cpu -c 2 --mem=3G --time=00:10:00 -J shape-visibility-report -o output/shape-20261121/progressive-l40s/visibility-report-%j.log --wrap='.amodal-cache/surface-metrics-env/bin/python scripts/build_shape_report.py --accounting'`. The cluster report, collected report and local reconstruction agree at aa061df4b249033b937f923f0011239c04f4e2c3edd44cc89ff47b77d34684b7. Shape full job 44946 was still running; combined 44948 was dependency-pending.

2026-11-21 UTC. Scheduling check only, full Shape job 44946. At 3.57 hours Slurm reported RUNNING and 504 records existed (204 preserved and 300 new). The completed prefixes of the four official worker partitions contained 61, 81, 75 and 83 new targets. The fresh records had 401 eight-candidate Shape pools; historical Surface had 387 for the same IDs. A nonnegative least-squares fit to elapsed times between successive fresh results on each worker estimated 6.1 seconds per target, 123.6 additional seconds for a first pool and 119.4 for each later pool. Applying those rates to the remaining historical per-target pool counts projected the last shard at about 25.6 hours from allocation start, before scoring. All 300 observed fresh targets were VG, so timing on later sources and future pool counts remain uncertain. The job limit is 24 hours. Continue monitoring 44946, confirm its terminal state before any continuation, and preserve the frozen outputs and accounting. The forecast did not change any experiment or submission.

2026-11-21 UTC. Completed the source proposal's K=4 oracle and first-seed comparison on the 32 controlled images with independent COCOA outlines. Mean original-frame amodal IoU was 0.6333 for seed 11, 0.6328 for uniform selection, 0.6732 for Shape and 0.6968 for the per-image polygon oracle. The oracle uses evaluation labels. On the two originally fully visible, untruncated objects, Shape and the oracle scored 0.5669 and 0.6414 for the bowl and 0.6585 and 0.6587 for the cat. Independently streamed the 128 original mask PNGs from the confirmation run, verified each digest, clipped each resized crop to the source frame and reproduced all candidate IoUs and selected Shape scores. WSL `scripts/check.sh` passed. Deployed `scripts/build_shape_report.py` (SHA-256 1208779b38bc8e68a90d62570eb7d380a737d1903fea6e1f7ae0152ee273c9ce). CPU job 45124 ran `sbatch --parsable -p cpu -c 2 --mem=3G --time=00:10:00 -J shape-oracle-report -o output/shape-20261121/progressive-l40s/outline-oracle-%j.log --wrap '.amodal-cache/surface-metrics-env/bin/python scripts/build_shape_report.py --accounting'` and completed in two seconds. Cluster and local reports and an independent local rendering matched SHA-256 b27ceffeb8d00db805a4d4e3b4753a8fc92d1b8521e4e6704ddfc59917f336b5. At 04:55:27 elapsed, full Shape job 44946 was running with 640 native records and combined job 44948 was dependency-pending. No GPU jobs were submitted.

2026-11-21 UTC. Full-run cohort reuse verification now checks every original per-target CSV column, including reference status and failure stage, against the verified cohort snapshot. The four metric values still accept equivalent numeric formatting. The regression changes reference status while retaining the original result digest and metric values and giving the edited CSV a valid execution receipt. The checker rejected that change. Local and remote replay of the archived Quadro controls confirmed all 204 records and all 18 columns of all 204 score rows in both PMC and Surface. The scheduled Shape and combined CPU audits use scripts/compare_shape_progressive.py, deployed via scp with SHA-256 744605870bf7373faa837d1139f8488a8adabdfe09c9f2d3899d0110b205e5ca. All 16 focused Shape tests and 293 Python tests passed in WSL, followed by the Bash regressions and CPU smoke. At 05:00:28 elapsed Slurm showed 44946 RUNNING on four L40S GPUs. Shape had 652 processed records, all four worker logs had just advanced, and combined job 44948 was pending with its 204 cohort records. No jobs were submitted.

The gpu partition lists one node with L40S devices, gus, with six physical L40S cards (sinfo -N -h -p gpu). Shape job 44946 holds four. Two study jobs requiring four L40S cards apiece cannot overlap on this partition. Shape supplies the verified references for combined job 44948. Removing that dependency would not allow an earlier four-GPU allocation while Shape runs.

2026-11-21 UTC. The local collector now retrieves the immutable reference-copy receipts for the two L40S cohorts and the full combined run, including any recorded full combined attempt that copied references before a timeout. Retired local watcher 23944 after checking its command line, and restarted the updated watcher as PID 26388 with its existing logs opened in append mode. The first collection recovered both cohort receipts using their remote SHA-256 digests. An independent local join against the official annotations, ordered cohort executions and digest-verified paired scores found identical file inventories for Shape and combined: all 612 RGB/crop/mask PNGs, the 204 reference records and the configuration. Each transfer copied 817 files from the verified baseline cohort. Full transfer evidence will come from the actual Shape-to-combined copy. All 16 focused tests and all 293 Python tests passed, followed by the Bash regressions and CPU smoke. Slurm reported Shape 44946 RUNNING at 05:20:03 with 687 target records and recent writes from all four workers. Combined 44948 was dependency-pending. No cluster jobs were submitted.


2026-11-21 UTC. Verified that the full Shape output belongs to Slurm 44946, with the official 2,565 targets in order and a matching environment digest. Archived PMC and Surface, the running full Shape configuration, and the combined cohort agree on the common inference settings, metric configuration and recorded package versions. The hardware and reference differences remain part of the published cross-device comparison.

Finished and tested the timeout inventory before any continuation. After sacct confirms TIMEOUT, it locks the matching output, preserves every completed result hash, archives unfinished target directories by atomic rename and records their file inventories. A retry of the snapshot can finish an interrupted archive. The final full audit compares the preserved result hashes and checks the archived partial bytes. The WSL scripts/check.sh suite passed, along with the 17 focused Shape tests. Deployed scripts/snapshot_shape_timeout.py (562a25f04aecc1dac97797af7abed5f159e2289c243349022ab856f306553713) and scripts/compare_shape_progressive.py (c1ea73ee128af434525365ac16f5e4934b43beb98c042a8843f26717c516e57d) with scp. Their remote hashes matched. The helper skipped both active job IDs in a cluster check. From the remote project submitted, for (arm, parent) in (shape, 44946), (combined, 44948), `sbatch --parsable -p cpu -c 1 --mem=1G --time=00:25:00 --dependency=afterany:PARENT --kill-on-invalid-dep=yes -J snapshot-ARM -o output/shape-20261121/progressive-l40s/snapshot-ARM-%j.log --wrap '.amodal-cache/surface-metrics-env/bin/python scripts/snapshot_shape_timeout.py ARM PARENT'`. The two CPU jobs are 45134 and 45135. Neither requests a GPU. Inspect the receipt and terminal state before submitting any needed continuation.

Checked the cohort severity finding against masks and source records. The COCOA bitmasks independently reproduce all 22 occluded fractions. The plate in COCO train image 024086 was the only polygon match at or above 60% (82.3579%). The three L40S runs have byte-identical native result records, SHA-256 25e0aed06a05716e3e090a2ddf806d802ee3a3eae0692dd50fd07a321e61a638. All three scored rows report zero iterations and query_mask_too_small. Deployed scripts/build_shape_report.py (d96bcbcfcc98323d711832a59803b1252c34d4b98084f0a4f476f034c20df1c3). The CPU command `sbatch --parsable -p cpu -c 2 --mem=3G --time=00:10:00 -J shape-severity-report -o output/shape-20261121/progressive-l40s/cohort-severity-%j.log --wrap '.amodal-cache/surface-metrics-env/bin/python scripts/build_shape_report.py --accounting'` ran as job 45136 and completed in one second. The cluster report, retrieved local report and independent rendering with the recorded 1264, 6593 and 6599 cohort elapsed seconds matched SHA-256 e62a866da91381c5daa5e18c1ce60f6195a41286c54ea262ab617f931744ef46. At 05:35:15 elapsed Shape had 718 processed targets, and combined was dependency-pending. No GPU jobs were submitted.

2026-11-21 UTC. Added automatic continuation for a confirmed full Shape or combined timeout. The CPU controller verifies Slurm state, reruns the idempotent timeout inventory, submits the recorded four-L40S command in the same output directory, and keeps the already queued child job IDs. Its continuation depends on the controller successfully publishing the new job IDs and updating the child dependencies. A second timeout without additional completed targets or cached scores stops for diagnosis. Non-timeout failures remain diagnostic. Completed jobs release their dependants immediately. The controls use the registered historical runs.

Rehearsed both paths on the cluster CPU partition in `.amodal-cache/shape-slurm-check-qbs3yv8s`. Held job 45137 used a one-minute limit and timed out at 00:01:03. The child 45139 remained pending on `afterok:45137(failed)?afterok:45138(unfulfilled)`. Controller 45138 submitted continuation 45140, which depended on 45138 completing, then changed 45139 to depend on 45140. All three completed and the child checked the continuation's marker. Success-path test 45145 used `--dependency=afterok:45140?afterok:45142` and completed while standby controller 45142 was still pending. All five checks used one CPU and zero GPUs. WSL scripts/check.sh and all 18 focused Shape tests passed. Transferred scripts/resume_shape_timeout.py and scripts/snapshot_shape_timeout.py via scp and verified remote SHA-256 05d75a9059922bafdb720fbe237bbe11c83cf378bed6d2937cd552d4055653fd and b455719c968b03c841f3e70c3e7a6314df1fac7fdb6da443a13725c79af8e5df.

From the remote project, CPU 45142 was submitted with `sbatch --parsable -p cpu -c 1 --mem=1G --time=00:25:00 --dependency=afterok:45134 --kill-on-invalid-dep=yes -J resume-shape -o output/shape-20261121/progressive-l40s/resume-shape-%j.log --wrap '.amodal-cache/surface-metrics-env/bin/python scripts/resume_shape_timeout.py shape 44946 44947 44948'`. CPU 45143 used the same resource flags, `--dependency=afterok:45135 -J resume-combined -o output/shape-20261121/progressive-l40s/resume-combined-%j.log --wrap '.amodal-cache/surface-metrics-env/bin/python scripts/resume_shape_timeout.py combined 44948 44949'`. Recorded both in progressive-l40s/submissions.tsv. Ran `scontrol update JobId=44947 'Dependency=afterok:44946?afterok:45142'`, the same update for 44948, and `scontrol update JobId=44949 'Dependency=afterok:44948?afterok:45143'`. Verified each new dependency through scontrol. The original successful GPU job remains sufficient on its own in each case.

The collector now retrieves timeout receipts and waits for a pending or running recovery controller after a confirmed timeout. Replaced watcher PID 26388 with PID 19700 after checking its process identity. The new watcher retrieved the revised ledger (local and remote SHA-256 ea0f047a8c497844ecd3a11461134b4862daf40940cf86687ccabc4776870a02), observed both CPU controllers pending, and retained the verified cohort report. At 05:47:11 Slurm showed 44946 running on four L40S devices, with 758 target records on disk. Combined 44948 was pending with its original 204. No study GPU jobs were added in this update.

2026-11-21 UTC. From the remote project, CPU job 45150 ran `sbatch --parsable -p cpu -c 2 --mem=3G --time=00:15:00 -J visible-prior-ablation -o output/shape-20261121/progressive-l40s/visible-prior-%j.log --wrap '.amodal-cache/surface-metrics-env/bin/python scripts/compare_shape_visible_prior.py'`. Slurm recorded 13 seconds. The local and cluster controlled-visible-ablation.json SHA-256 was 116dafeb1438077694c92b4bfac5fa75f44a34f2374633b83d2fab4c30ab7050. Its source script and COCOA audit hashes matched 517e4b5bf81ab270ccd23c87eea2081b14c9792548c25c6d89ab20b252c88564 and c16e1d4752fc4ccb99b01a84dc8a730aafc39045846aee231a82e3f551849f89.

Rendered the report locally with the actual cohort accounting for 44913, 44914 and 44917, checked all eight interval cells against the evidence, and confirmed a single slide 18 insertion relative to the prior published report. Transferred scripts/build_shape_report.py by scp, verified its SHA-256 9b1350d1ee57d4263e1b2af516864cbbb2955605b541730ba085c404f54481ad, and installed it over the verified previous version. CPU job 45155 ran `sbatch --parsable -p cpu -c 2 --mem=3G --time=00:10:00 -J shape-visible-report -o output/shape-20261121/progressive-l40s/visible-report-%j.log --wrap '.amodal-cache/surface-metrics-env/bin/python scripts/build_shape_report.py --accounting'` and completed in one second. The cluster report, its retrieved copy and the independent WSL rendering matched SHA-256 5cf155e87b43ec24df192d9c8ea8d90626088748ed6fdffe728e110d4b5a3660. In the existing WSL CPU replay environment, scripts/check.sh passed 295 Python tests, the Bash regressions and CPU smoke. At 20:49:05 UTC direct cluster enumeration found 782 Shape and 204 combined native records. Slurm showed 44946 running and 44948 waiting on its recorded dependency. Neither full registry existed. Only CPU allocations were used for this report update.

2026-11-21 UTC. Checked GPU-hour accounting against sacct for the five historical full/cohort allocation IDs 39486, 39488, 39487, 39490 and 39491. Their elapsed seconds reproduced 24.597 reserved Quadro GPU-hours for PMC and 116.406 for Surface, including the 86,419-second timeout. L40S jobs 44913, 44914 and 44917 reproduced 1.404, 7.326 and 7.332 reserved GPU-hours for the three completed cohorts. The separate failed historical-reference attempt 44905 used 0.120 L40S GPU-hours, consistent with the protocol used in the Surface report. The canceled repeat controls had zero allocation.

CPU job 45157 ran a read-only comparison from the remote project via `srun -p cpu -c 1 --mem=512M --time=00:05:00 -J pmc-trace-check python3 -`. Checked the two frozen execution digests in pmc-cohort-cross-device.json, all 408 native result hashes, target order, input digests and seeds. Both runs initially selected a target on 130 cases and neither selected one on the other 74. Every selected bounding box matched. Among the 130 boxes, recorded detector score matched on 113 and selected mask area on 92. The one coverage change was COCOA/COCO_train2014_000000108113_dog. Its result SHA-256 was 3353de90d646e6c6e52d1de9cd87a5ad01d50de266b41239c8cb6d4a433dd5c1 on Quadro and b56a9b26f14b62943d3394731192b2af861956082c46ec14bb59b860b3b92e20 on L40S. On this dog both traces record the right boundary and one iteration, with empty_mask_array recorded at completion segmentation on L40S. A separate local join of the two digest-verified CSVs reproduced the one coverage transition, 15 different iteration counts on the 86 shared successes, and 135 versus 145 total recorded iterations. The GPU work remains the two authorized full Shape arms.

2026-11-21 UTC. Added the published Quadro Surface cohort to the three L40S rows for slide 15. The builder verifies its score-table digest against the paired audit, uses all 204 failure-inclusive rows, and reads the cost of original job 39487. Restored the existing published cohort-compute.json receipt to its corresponding report directory on the cluster. Its SHA-256 (23ae8d9473077359a19658ffc890005a3e432033ae5eb175a0e30485d327183b) matched the archived package manifest. Slurm independently confirmed four GPUs for 6,574 seconds. A local render using the actual three L40S allocations reproduced the verified historical summary row and showed that the report gained one row and its cross-device note. The new regression verifies the failure penalties and rejects altered historical scores. WSL scripts/check.sh passed 296 Python tests, all Bash regressions and CPU smoke.

Transferred scripts/build_shape_report.py with scp and verified SHA-256 f4af491c5914c8155451d99d6551e46e788531a6d68b1da67d3bbd990ccaf7c1. CPU job 45158 ran `sbatch --parsable -p cpu -c 2 --mem=3G --time=00:10:00 -J shape-cohort-table -o output/shape-20261121/progressive-l40s/cohort-table-%j.log --wrap '.amodal-cache/surface-metrics-env/bin/python scripts/build_shape_report.py --accounting'` and completed in one second. The cluster report, retrieved report and local rendering matched SHA-256 b541847710a31d3cb64e0abf630f62d827f7f428f924842f73d31af5c0f5e25e. At 20:58 UTC full Shape 44946 had 796 processed targets and combined 44948 retained 204 while waiting for Shape. No GPU job was submitted.

2026-11-21 UTC. Read-only CPU job 45159 ran from the cluster project with `srun -p cpu -c 1 --mem=2G --time=00:10:00 -J shape-full-crop-check .amodal-cache/surface-metrics-env/bin/python -`. At a snapshot of 798 completed full Shape records, all 204 original Shape cohort result hashes matched. The 817 existing reference files in the combined destination matched their frozen receipt and the corresponding current Shape source files byte for byte. The remaining references will come from the scheduled copy after Shape finishes.

For a progressive geometry check, selected the lexicographically first completed non-cohort target at each observed pool count from one through six. In increasing count order these were VG/1159292_microwave, VG/1159707_redvan, VG/1159295_bus, VG/1159300_redvan, VG/1593053_surfingboard and VG/107927_greycar. All 21 crops extended outside the source image. Independently projected the original visible mask from source coordinates into each saved crop. All 21 matched the saved visible masks, kept every visible pixel in the prior, and added predicted support only inside the recorded occlusion region. Replayed all 168 native candidate mask scores and verified the selected slot, fallback flag and raw selected RGB bytes for every pool. This checks the crop geometry and selection of these six completed targets, without evaluating a partial quality score.

2026-11-21 UTC. Extended the scheduled full report's coverage decomposition to each audited historical control, including Surface. Each comparator uses its own completion statuses and LPIPS. Replayed the Surface comparison on all 204 cohort targets using both digest-verified paired files, the archived Surface CSV and the frozen target order. The contributions matched both recorded cross-device mean differences to within 1e-12. Shape and Surface shared 86 successes, with one additional Surface output. Combined and Surface shared all 87 successes. The normal report entry point with actual cohort accounting reproduced published REPORT.md SHA-256 b541847710a31d3cb64e0abf630f62d827f7f428f924842f73d31af5c0f5e25e. The WSL scripts/check.sh suite passed 297 tests, Bash regressions and the CPU smoke. Deployed scripts/build_shape_report.py by scp after checking the previous version; the installed SHA-256 is cb1a462df6a8c5062ed34154515908c4d063781627d10be708d405612f31564a. The existing audit/report jobs will render the full coverage rows after the runs complete. At 21:13 UTC Shape had 809 completed records and combined was waiting with 204. No new Slurm job was submitted.


2026-11-21 UTC. Retrieved the native L40S PMC dog record for COCO train image 108113 from the verified cohort snapshot. The local Quadro and L40S records match both cohort execution digests. Both selected the box [14, 68, 640, 420] with score 0.36. The L40S trace records an empty mask array during first-pass completion segmentation. Updated the report to describe that observed stage and link both traces. The reference audit confirms a one-pixel difference, and independent recomputation from the 204 paired scores attributes 97.1% of the PMC mean LPIPS gap to this dog. The records do not determine a hardware cause.

Deployed scripts/build_shape_report.py via scp (SHA-256 a9e19b390ba37270e77a56da05b03b59602e3758aa71edc26e3166549488bcbf). CPU job 45162 used `sbatch --parsable -p cpu -c 2 --mem=3G --time=00:10:00 -J shape-cohort-trace -o output/shape-20261121/progressive-l40s/cohort-trace-%j.log --wrap '.amodal-cache/surface-metrics-env/bin/python scripts/build_shape_report.py --accounting'` and completed in one second. Local and remote REPORT.md matched SHA-256 f860b552509ec64279f66b5af43c053327e5ff78e8f6a7627fa382542ae8612a. Only the dog paragraph changed. At 21:19:59 UTC Shape 44946 was running with 816 processed targets, combined 44948 was pending with 204, and the full audits and report were pending. The collector was live as PID 19700.

2026-11-21 UTC. Checked the heavy-occlusion question in section 2 of the saved proposal using the 32 frozen controlled COCOA cases. Matched and verified the cases.json hash, every annotated raster and each actual artificial occluder. In the 512-by-512 model crop, the numerator is all annotated object pixels except those both originally visible and outside the artificial occluder. Padded and out-of-crop pixels do not add to the denominator. An independent pixel count reconstructed every occluder from the recorded direction and threshold. The five bins contain 3, 14, 11, 4 and 0 images. Elephant 220612, microwave 174113, bed 244581 and cat 238453 fall in the 60% to under 80% bin. The actual controlled crop table reports all bins using the frozen four-candidate Shape and uniform original-frame IoU. The group-size-weighted means reproduce the original means to the table's displayed precision.

Published the report builder with scp after verifying its installed predecessor. New script SHA-256 75d53e40501028bcfcd7a7ae20acb0eda210523cf7a21d19bef26b7c03503a70. From the remote project, `sbatch --parsable -p cpu -c 2 --mem=3G --time=00:10:00 -J shape-crop-severity -o output/shape-20261121/progressive-l40s/crop-severity-%j.log --wrap '.amodal-cache/surface-metrics-env/bin/python scripts/build_shape_report.py --accounting'` returned job 45163. It completed in two seconds. The independent WSL render with actual cohort accounting, downloaded report and cluster report matched SHA-256 2a4a5bf9af30e242edff29499fede948f5408a3d240f3023151f5f722329da04. Removing the new subsection reproduced the preceding report text. All 298 WSL Python tests, Bash regressions and CPU smoke passed with scripts/check.sh. At 21:27:47 UTC Slurm showed Shape 44946 running with 840 records and combined 44948 pending with 204. The full registrations and audits remained outstanding.

2026-11-21 UTC. CPU job 45166 ran a read-only pairing check from the remote project using `srun -p cpu -c 1 --mem=2G --time=00:05:00 -J shape-target-pairing .amodal-cache/surface-metrics-env/bin/python -` and completed in eight seconds. At the time of enumeration the full Shape run had 843 committed target records. For each, checked the official target ID, source, label, original image ID, paths, input digest against image_checksums.json, seed derived from the target ID and base seed, and the native record digest for each registered historical control against its execution manifest. All 843 inputs and seeds matched both Quadro controls. The 204 reused Shape records matched its frozen cohort execution byte for byte. All 639 fresh records came from VG. Their current reference metadata agreed with the three artifact hashes in the corresponding Shape result. Both historical controls agreed with one another on the same three hashes for every target. For 482 of the 639 fresh VG targets those hashes also matched Shape, with at least one different hash on 157. Among the original 204 the counts were 166 and 38. This check compared recorded hashes and did not re-read all PNG bytes or analyze quality. At 21:34:43 UTC, 44946 had 844 records with all four worker logs active. Combined 44948 was dependency-pending with its 204 cohort records.

2026-11-21 UTC. Joined the archived Surface polygon audit to each verified L40S cohort by target ID and original image identity. Independently projected the seven saved historical Surface-cohort completion alpha masks into the original COCOA image frames, verified their digests and the 22 full/cohort native record digests, and retained zeros for the 15 failed completions. All three methods saved the same seven objects. A separate implementation sampled the 20 image IDs in clusters for 5,000 paired bootstrap draws, seed 20261110. The means and interval endpoints agreed within 1e-12. Relative to historical Surface (mean 0.26760592), Shape had mean 0.26312102 and absolute difference -0.00448490 [-0.01666890, +0.00747284]. Combined had mean 0.24816440 and absolute difference -0.01944152 [-0.05642522, +0.00045792]. Comparisons are cross-device. The 321-target full polygon comparison will use the audited surface-all/amodal_iou estimate for each arm.

Transferred scripts/build_shape_report.py via scp and submitted from the remote project with `sbatch --parsable -p cpu -c 2 --mem=3G --time=00:10:00 -J shape-polygon-table -o output/shape-20261121/progressive-l40s/polygon-table-%j.log --wrap '.amodal-cache/surface-metrics-env/bin/python scripts/build_shape_report.py --accounting'`. CPU job 45175 stopped in one second on a missing scripts-package import when invoked as a direct script. Moved the shared estimator import to program startup with the project root on sys.path. Uploaded the correction (SHA-256 5049cfe38a930f38f3dd09cbd1085b63cc579f8160805f4a857e49248f729b8a) and repeated the CPU command as job 45176, which completed in two seconds. Both jobs used zero GPUs. The downloaded report, independent WSL render with recorded allocation times, and remote report matched SHA-256 3d3c1525f14bca4e9ab25829c99ed995daa648016d58545ecda89543ad988bb6. Only two geometry paragraphs changed in the current cohort report. WSL scripts/check.sh passed 299 Python tests, Bash regressions and CPU smoke after the import correction. At 21:41:06 UTC Shape 44946 was running with 851 records and combined 44948 was waiting with 204.

2026-11-21 21:45 UTC. Read-only SSH checks on the cluster confirmed the full Shape execution and both registered historical controls have the same 2,565 ordered target IDs. Their base evaluation configurations, metric configurations and recorded Python package inventories match, including the completed combined cohort on the configuration checks. The two new methods reference the same verified checkpoint and training receipt, and the combined cohort's frozen Surface policy matches historical Surface. These checks do not establish numerical equivalence across GPUs. Compared each arm's current native files with its own cohort snapshot. All 204 Shape result hashes and 86 cached success metric files matched, as did all 204 combined result hashes and 87 cached success metric files. At 21:45:22 UTC, Shape job 44946 had 861 processed records and combined job 44948 was waiting with its 204 cohort records. Windows collector PID 19700 was running the expected watch command. The full score tables and registrations are pending.

2026-11-21 UTC. Compared the Week 8 slide 17 evidence with the native umbrella (VG/2316897_sunumbrella) and bus (VG/2352721_bus) candidate pools. Read-only CPU job 45202 ran from the cluster project with `srun -p cpu -c 1 --mem=512M --time=00:03:00 -J slide17-check .amodal-cache/surface-metrics-env/bin/python -` and completed in one second. Verified all 12 displayed image tiles against the resized source pixels, recomputed the eight silhouette IoUs from native masks, and matched the eight LPIPS captions to the archived scores. Shape chose seed 44 for the umbrella and seed 11 for the bus. The local and remote comparison figure matched SHA-256 5773e65b9f0ca48339dc50bdde6d09531828bab5e182a83ff7873cab2000f7af. All 39 local link references in this version of the Week 8 report resolved, including all three figures. Slide 16 still awaits the full L40S results.

2026-11-21 UTC. Verified the cross-device sensitivity analysis on the 166 cohort targets from 159 images with byte-identical reference artifacts. Joined the checksum-verified CSVs for L40S PMC, Shape and combined and Quadro PMC and Surface to the reference audit by target ID. A separate implementation weighted the per-image sums by bootstrap draw multiplicities within source for 5,000 draws with seed 20261110. It reproduced all three LPIPS means, relative gains and interval endpoints to within 1e-12. PMC on L40S versus PMC on Quadro gave +0.033% [-0.173%, +0.281%]. Shape and combined versus historical Surface gave -1.887% [-3.873%, -0.616%] and +0.066% [-0.109%, +0.267%], respectively. Positive gains mean lower LPIPS on L40S.

Added these existing paired intervals, subset means and image counts to the cohort and scheduled full report. The independent WSL render changed three paragraphs in the current report. WSL scripts/check.sh passed 300 Python tests, all Bash regressions and the CPU smoke check. Transferred scripts/build_shape_report.py by scp, verified the deployed predecessor and staged digest, and installed SHA-256 04f35c5185a54602dd7cd871fd2b2ac1c4fa67f7d95a03ef354ba1685d125a90. From the cluster project, `sbatch --parsable -p cpu -c 2 --mem=3G --time=00:10:00 -J shape-ref-intervals -o output/shape-20261121/progressive-l40s/ref-intervals-%j.log --wrap '.amodal-cache/surface-metrics-env/bin/python scripts/build_shape_report.py --accounting'` submitted 45219. It completed in two seconds on CPU. The cluster report, independent WSL render and retrieved report match SHA-256 ce07bc61e5282c0fffe57a4a87c58e403e6b6437b1b4adbd9ea377ee6fc20e4a. At 21:53:35 UTC, Shape job 44946 had 879 processed records and combined job 44948 was waiting with 204. Both full registrations are pending.

2026-11-21 UTC. Compared the archived source and environment for the two completed 204-target PMC controls using read-only SSH. Job 39486 ran on gebru in debug-gpu and 44913 on gus in gpu. Verified each execution against pmc-cohort-cross-device.json, both source and environment manifests against the executions, and every file in each source snapshot against its manifest. The Quadro snapshot contained 45 files and the L40S snapshot 47. Of the 45 shared paths, 44 were byte-identical, including the inference adapter. The sole edited shared file was evaluate_open_world.py. Its diff adds the optional Shape argument and guards Shape setup, worker forwarding and installation behind that argument. The two additional files implement Shape. Both PMC runs had an empty Shape checkpoint and Surface freeze argument. The base run configurations and metric configuration bytes matched, as did the upstream main file hash, upstream commit, four checkpoint hashes, diffusion revision, Python version and all 111 recorded package entries. Source manifest digests were f080bbc4093c340a44e79071b2ff4184b93d9017b683f53c8327d7061ce1dc87 (Quadro) and 2d9ae393bdc7f14977147d3a364f3491540d7558a92556a079e1487aad42f302 (L40S). This comparison establishes the recorded software configuration and the scope of the evaluator changes. It leaves the cause of the divergent dog segmentation undetermined.

A second read-only check parsed both archived evaluator entry points. Removing only the one new CLI declaration and simplifying the five Shape guards using the recorded absent checkpoint produced identical syntax trees (SHA-256 14bce81f2b3b483fa3b9ea2371f6e4b3db24b7228a8b5a1072eff6c6822647c2). This check covers the evaluator entry point. At 22:03:39 UTC, Slurm showed Shape 44946 running with 891 target records and combined 44948 pending with 204. Neither full registry existed.

2026-11-21 22:12 UTC. Slurm sacct reported CPU jobs 45246 (shape-sampling-counts) and 45247 (shape-pass-report) completed in two seconds each. The retrieved Week 8 report and the cluster copy matched the expected SHA-256 bce622678d9f5a0603a3106b51c24b4586bc412db65d5219370d73c8ccc968f1. The report records 158 Shape passes and 150 combined passes. Slurm showed Shape 44946 running and combined 44948 dependency-pending. Native result counts were 900 and 204. The collector was active under PID 19700. The published PMC and Surface full controls remain the comparison rows per the user's hardware decision.

2026-11-21 22:16 UTC. Checked slide 16 in the Week 8 Vietnamese PowerPoint. Its embedded chart (ppt/media/image22.png, 2063 by 963 pixels) matches the archived severity-vietnamese.png in every RGB pixel. Both full Quadro per_target.csv files matched their execution digests and ordered 2,565 IDs. All 321 COCOA target IDs, LPIPS scores and statuses matched severity-joined.csv for both controls. The original slide uses the recorded annotation ratio and the Shape report computes the ratio from polygon masks. Their largest absolute difference was 3.66e-10 or less, preserving every target's original bin and all ten published mean LPIPS values exactly. At this check Slurm showed Shape 44946 running (909 native result files) and combined 44948 waiting (204 retained). No allocation was submitted.

2026-11-21 UTC. Recomputed the controlled Shape and combined differences versus Surface on the 32 distinct COCOA images with independent amodal outlines. The local controlled-cocoa.json matched the audited cluster digest c16e1d4752fc4ccb99b01a84dc8a730aafc39045846aee231a82e3f551849f89. Joined all 32 saved decision files by case and target ID, checked their recorded hashes and reapplied the three frozen selections to their eight original-frame polygon IoUs. A separate count-weighted implementation of the 5,000 paired image draws with seed 20261110 gave Shape minus Surface +0.011397406 [-0.010142867, +0.036257719] and combined minus Surface +0.008553272 [-0.009120163, +0.028420177]. These are post hoc controlled comparisons on the same candidate pools. The WSL rendering using the three recorded cohort allocations added one paragraph. WSL scripts/check.sh passed the Python suite, Bash regressions and CPU smoke.

Transferred the report builder by scp. Verified the cluster predecessor and installed SHA-256 4b333523aea0403b4a29a8f13c56ca997ce04cc1a36e48b755aeb104d2705b20. From the cluster project, sbatch --parsable -p cpu -c 2 --mem=3G --time=00:10:00 -J shape-outline-report -o output/shape-20261121/progressive-l40s/outline-report-%j.log --wrap '.amodal-cache/surface-metrics-env/bin/python scripts/build_shape_report.py --accounting' submitted 45259, completed in two seconds with exit 0. The retrieved report, independent WSL render and cluster report matched 7f9eb7825de5ab8205691b866cf8ea556f05d4aa98667444dfb28ed5338655a6. At 22:22:24 UTC, Shape 44946 was running with 928 processed targets and combined 44948 was waiting with 204.

2026-11-21 UTC. Compared each cached score in the two live L40S arms with its own frozen cohort by target ID and SHA-256. All 86 Shape and 87 combined metric files matched. Neither arm had scored a further success. The evaluator writes each score atomically and reads cached scores when the completion and reference digests match.

The timeout snapshot previously counted score files. It now records the SHA-256 of each existing metrics.json alongside the completed result inventory. The final CPU audit checks those files against their pre-timeout hashes and allows additional scores from the continuation. Older receipts retain their original meaning. The recovery regression exercised a changed score with the same target and file count, newly scored targets and an older count-only receipt. The WSL check log recorded 300 Python tests passing, the Bash regressions and CPU smoke. The outer log filter returned nonzero after the suite; the test log confirmed completion. Transferred only scripts/snapshot_shape_timeout.py by scp and atomically installed it after verifying the prior and staged digests. The cluster digest is 22f72d006b276ec0c183edace9122b972f2c40ba4f196795fcb0686303c42c63. No GPU job was submitted or restarted. Slurm showed Shape 44946 running with 954 target records at 22:32:05 UTC and combined 44948 pending with 204.

2026-11-21 22:59:57 UTC. At the first check above 1,000 native Shape results, a read-only comparison ran on the cluster from the project root using srun -p cpu -c 1 --mem=512M --time=00:08:00 -J shape-input-check python3 -. Slurm job 45275 completed in six seconds with zero GPUs. Verified the execution digests of both registered full controls against cocoa-historical-controls.json and checked their ordered 2,565 IDs against the active Shape execution. For every one of the 1,006 Shape records present at enumeration, the two archived control records matched their published hashes and all three had identical target IDs, input hashes and seeds. All 204 Shape cohort record hashes matched their original execution. The recorded scoring-reference hashes matched both historical controls on 781 targets, including 166 in the cohort. The two historical controls agreed on all 1,006. This running inventory does not establish a full-benchmark reference match rate or a method effect. Shape 44946 remained running with only its 86 previously cached success scores. Combined 44948 was waiting with 204 retained targets. Neither full registry existed.

2026-11-21 23:25 UTC. CPU job 45286 published the cohort compute comparison with the report builder's --accounting option. sacct recorded four L40S GPUs for 6,593 seconds on Shape job 44914 and 6,599 on combined job 44917. The combined/Shape allocation ratio is 1.001 after rounding, over 150 and 158 sampling passes respectively. The cluster report and local WSL render both hashed to 904e13b91a6924665dd936282a723380ada6754b4a5eb1bc321471cd187f4724.

From the cluster project, srun -p cpu -c 1 --mem=1G --time=00:12:00 -J shape-first-pass-check .amodal-cache/surface-metrics-env/bin/python - ran job 45288. A second invocation with -J shape-first-pass-receipt ran job 45289. Both read the completed Surface and Shape cohort snapshots. They verified each result against its execution digest, each first decision against the native result trace, and the decoded images against the PNG digests. The NumPy array comparison and the Pillow histogram replay agreed on the sum of absolute RGB differences (216,738,018 across 371,195,904 channel values) for the 472 candidate pairs from 59 targets with matching references, crop coordinates, visible masks and seeds. The saved receipt binds every target to its first-pass decisions and both cohort executions. Downloaded SHA-256 171a185f7ac57580fc03483a33f7301f2abe626c39ab860fe3dde4c23c53eec6. The local identity check matched all 204 IDs and all reference classifications to the earlier independent mask audit.

Staged scripts/build_shape_report.py with scp and installed the verified SHA-256 3748aacb002d8cdc8b1d01ca7a71cb8f63a39f36788b928a2bf1c19a9466532b. Submitted the report-only CPU publication with sbatch --parsable -p cpu -c 2 --mem=3G --time=00:10:00 -J shape-first-pass-report -o output/shape-20261121/progressive-l40s/first-pass-report-%j.log --wrap '.amodal-cache/surface-metrics-env/bin/python scripts/build_shape_report.py --accounting'. Job 45290 completed in two seconds. The local independent render, retrieved report and cluster report matched SHA-256 93b9504d6a0a43a33fc924882abb4b5c724d5ab80b663b947fe0ad01f27ae9ff. All 40 local relative report links resolved. WSL scripts/check.sh passed 300 Python tests, Bash regressions and the CPU smoke. At 23:24:56 UTC, Shape 44946 was running with 1,061 of 2,565 records and 86 cached scores. Combined 44948 was waiting with 204 records and 87 cached scores. The Windows collector was live under PID 19700. Both full audits and the final report are pending.

2026-11-21 23:34 UTC. The final combined CPU audit now replays the full Shape-to-combined reference copy against the native files in both run directories. It checks the configuration, each official target's full RGBA image, crop, mask and JSON record, and the receipt's paths, source target count and digest map. It records the receipt digest and reported new-file count for the final allocation and each timed-out allocation with a verified execution snapshot. Missing or changed reference files fail the audit. The result will appear in combined-full-summary.json under reference_transfers.

Staged scripts/compare_shape_progressive.py via scp. From the remote project, srun -p cpu -c 1 --mem=1G --time=00:10:00 -J shape-check-copy .amodal-cache/surface-metrics-env/bin/python - ran the staged code as job 45291. Both existing 204-target receipts and all 817 native source and destination files per arm passed, including the reference records and configuration. The two SHA-256 digests were e3316db155bf6b78e02d2feca067c8aea05dde4cfa1ba1a6a4e3d20676609028 for Shape and 2a89fad1f6111ad47915cb9ab91529fd387307f5e17fb73d41b4fddec4687ce2 for combined. The cluster completed the CPU job in eight seconds. The test fixture exercised an original and a resumed receipt, an incomplete receipt, a changed destination and a missing source record. All 24 focused tests passed on the installed revision. WSL scripts/check.sh passed 301 Python tests, Bash regressions and CPU smoke. Installed the staged CPU audit with a digest-checked atomic rename, replacing c1ea73ee128af434525365ac16f5e4934b43beb98c042a8843f26717c516e57d with 0d2d87544e1891e5ed2e4c5707bc57d9bc0c2a9ec9ab454483893f813d34dbba. At 23:33:34 UTC Shape job 44946 was running with 1,074 committed target records. Combined job 44948 had 204 records while waiting on Shape. The full reference copy, GPU scores, registrations and final audits remain outstanding.

2026-11-21 23:58 UTC. To check the natural-image example for Week 8 slide 19, ran `srun -p cpu -c 1 --mem=1G --time=00:05:00 -J shape-bed-check .amodal-cache/surface-metrics-env/bin/python -` from the remote project. Job 45293 verified the two cohort result and completion digests for COCOA train image 009199, bed, and checked matching inputs, seeds and scoring references. The recorded LPIPS scores agreed across the execution-verified CSV, per-target metric cache and paired audit. Independently pasting each saved alpha mask onto the original image frame reproduced COCOA amodal IoU. Shape scored 0.461658 LPIPS and 0.760512 IoU. Shape + Surface scored 0.454907 LPIPS and 0.694033 IoU. This is an illustrative case from the completed cohort.

Deployed only scripts/build_shape_report.py, SHA-256 056f09d85b7db7771e6b6b8855a86e78cc00446efe9a2c1f7c5ee2dbc1df2a0d. The report explains the eroded visible SAM reference on white, the failure penalties and the bed example in Vietnamese. Job 45294 used `srun -p cpu -c 2 --mem=3G --time=00:10:00 -J shape-metric-report .amodal-cache/surface-metrics-env/bin/python -` to run the builder with --accounting. Its report, the SCP download and the independent WSL render matched SHA-256 5d9e12bd028411028c31e08711cf17a0d2d68305f31ef7512169a424131502b0. All 43 local links resolved. The additions left the existing tables intact. WSL scripts/check.sh passed the Python suite, shell regressions and CPU smoke. The two Slurm jobs used only CPUs. At 23:58:33 UTC Shape 44946 was running with 1,132 of 2,565 records. Combined 44948 was waiting with its 204 cohort records.

2026-11-22 00:03 UTC. CPU job 45295 used the four live worker partitions and completed record timestamps to estimate the remaining Shape inference time. From the remote project it ran `srun -p cpu -c 1 --mem=1G --time=00:08:00 -J shape-remaining-time .amodal-cache/surface-metrics-env/bin/python -`. It confirmed 204 frozen records and 935 new records, all VG. The workers had completed 201/598, 258/582, 247/599 and 229/582 new targets. These 935 targets took 1,087 eight-candidate Shape passes and the historical Surface run took 1,062 passes on the same targets. At the observed rates (5.38 s per target, plus 116.35 s for a first pass and 116.36 s per later pass), using historical pass counts for the remaining targets projected the slowest worker to finish inference 23.86 h after execution start, around 21:43 Vietnam time. The initial allocation ends at 21:51:11 Vietnam time on November 22 according to Slurm. The projection excludes scoring, verification and registration. The other sources have yet to contribute fresh Shape timing observations. For scale, scoring the remaining 1,312 outputs in historical Surface job 39491 ended 400.8 s after the last native inference record. Current Shape timing can differ. Recovery remains conditional on Slurm confirming a timeout. No GPU jobs or dependencies were changed.

2026-11-22 00:10 UTC. Recovered the three immutable 204-target cohort metadata snapshots into output/shape-20261121/progressive-l40s. A read-only SSH transfer ran `tar -czf - --hard-dereference --null -T -` in the remote study directory using an explicit file inventory. The 866,117-byte stream contained 1,039 files from baseline-cohort, shape-cohort and combined-cohort. Used scripts.collect_surface_job_evidence.recover_archive to check every file against its independently fetched remote hash and preflight existing local files. The canonical manifest digest was b90dc2d76e1a16efcc70a2b9913d28c835f4ff7f8a371b0fe9be82c1acfb727d.

Ran amodal_benchmark.provenance.verify_provenance on the staged and installed copies. The execution, source files, environments, 612 original target records and the scored CSVs all passed. The native snapshots identify L40S jobs 44913, 44914 and 44917. Checked all 204 ordered IDs per method, the seed and status on every row, and all four CSV metrics against 259 native success caches or the frozen penalties for 353 failures. Every success cache was bound to its saved completion and scoring-reference crop digests. The native PNGs remain on the cluster. The transfer used no Slurm allocation. At 00:10:20 UTC, Shape job 44946 had 1,158 target records and combined job 44948 was pending with 204.

2026-11-22 00:22 UTC. Read slides 8-20 in the Vietnamese Week 8 PowerPoint and checked their placeholders against the current report. Added Vietnamese copy for slide 11's geometry input, progressive selection and evaluation, and slide 14's equal-weight rank combination, tie breaking and fallback. These passages were checked against amodal_benchmark/shape.py and shape_runtime.py. The report builder deployed at SHA-256 bd88f71e00ca5a7c2d5569a722846dff5fcc28472b2a44a61ffb4e16ea36a975. Slurm job 45297 ran the builder with --accounting using srun -p cpu -c 2 --mem=3G --time=00:10:00 -J shape-slides-method, completing in one second. The cluster report, recovered copy and independent WSL render using the recorded Slurm allocations matched SHA-256 593923b27afe9ad7336f0b0af915b62d660657c8921f69a579d4623146d19d3e. All 43 local links resolved and all published score tables matched the preceding revision. At 00:22:10 UTC Shape job 44946 was running with 1,192/2,565 records and combined 44948 was dependency-pending with its 204 cohort records. Both full score tables remain outstanding.

2026-11-22 00:37 UTC. From the remote project root, CPU job 45298 ran `srun -p cpu -c 4 --mem=4G --time=00:12:00 -J shape-prior-replay env PYTHONNOUSERSITE=1 .amodal-cache/conda-env/bin/python -u -`. Slurm recorded 72 seconds, four CPUs, zero GPUs and exit 0. PyTorch 1.13.1 loaded prior.pt on CPU after checking the training receipt, checkpoint, predictor implementation and confirmation inventory digests. For the controlled cases, read the three hash-checked input PNGs and the eight archived candidate masks. For both cohort snapshots, checked every target result against the completed execution and every decision against the recorded trace, and verified the input, prior and candidate mask digests before replay. Recomputed the U-Net prior from the observed RGB, visible and occlusion masks on every pass. Pixel differences versus the archived L40S priors were 0/33,554,432 for the 128 controlled cases, 0/41,418,752 for 158 Shape cohort passes and 0/39,321,600 for 150 combined cohort passes. The three controlled choices per image (Shape K=4, Shape K=8, combined K=8) and the 308 cohort choices, including fallback flags, all matched. The check reused candidate masks and the stored Surface losses. At 00:37:06 UTC Shape 44946 was running with 1,212/2,565 target records and combined 44948 was waiting with 204.

2026-11-22 01:03 UTC. Joined the official target and original image IDs to the 321 independently audited COCOA polygons, the frozen unseen inventory and the Shape training identities. Verified the polygon and unseen inventory hashes against their earlier audits. Of the 321 polygon targets, 28 come from Shape training image IDs. The frozen unseen split contains 211 polygon targets from 202 images, with zero training-image ID matches. Its five occlusion groups contain 126, 38, 27, 17 and 3 targets. On these 211 targets, historical Quadro PMC saved 110 completions with mean amodal IoU 0.423114, and Surface saved 109 with mean IoU 0.420102. Failed completions have IoU zero. Replaying the frozen 5,000-draw image-cluster bootstrap gave a Surface minus PMC difference of -0.003012, nominal 95% interval [-0.017500, +0.010171].

The full report builder now renders Vietnamese mean IoU for each of the four methods on these 211 unseen targets, plus the paired combined-minus-Shape interval. It reads the two verified full paired audits and checks both arms' denominators, means and historical controls. Deployed scripts/build_shape_report.py by SCP with a digest-checked rename on the cluster. Installed SHA-256 da7c9ef8049a6f914b89fec97a3a6a2205671d5458a4e11c8fd28be859d4865c. The normal report entry point, replayed locally with sacct elapsed seconds for jobs 44913, 44914 and 44917, reproduced the current report SHA-256 593923b27afe9ad7336f0b0af915b62d660657c8921f69a579d4623146d19d3e. WSL scripts/check.sh passed 302 Python tests, Bash regressions and CPU smoke. At 01:03:30 UTC Slurm showed Shape 44946 running with 1,258 native records. Combined 44948 was pending with 204. The full Shape and combined means require their actual score tables and audits.

2026-11-22 01:10 UTC. Recovered the two immutable L40S cohorts' selector evidence into progressive-l40s/shape-cohort and combined-cohort. The read-only SSH transfer ran tar -czf - --hard-dereference --no-recursion --null -T - from the remote study directory against explicit member names. The archive contained 6,723,008 bytes with SHA-256 627d7674d8b5e2fc01a6ab99d1db4aadc89f7d2f9f7e81ec4df49c7fb56fdb55. Verified both 204-target inventories against the frozen cohort, each result against its execution and each decision against its target trace. The native prior and eight candidate mask PNGs for every decision matched the hashes in that decision. Used scripts.collect_surface_job_evidence.recover_archive to install the 3,080 files. Re-enumerating and hashing installed files reproduced canonical sorted-JSON manifest SHA-256 ff29379ae1b2d9e95de78f8be814073f957e0d0c7b448b312db1c2d7721eac58.

Local replay from these masks reproduced all 2,464 prior/candidate IoU scores, including empty masks, all eight derived seeds per pass, and all selections and fallback flags for 158 Shape passes and 150 combined passes. Combined Surface losses recomputed from saved observable features and the verified frozen ranker agreed within 1e-10. The independent joint ordering assigned each valid candidate the average of its tied positions in each ranking and broke ties in their sum by seed order. Two Shape and one combined pass used fallback. On the first pass, the two methods sampled the same eight candidates on 87 targets, and neither sampled on 117. Among the 87, Shape and the frozen Surface-only ranking of the combined arm's saved candidates agreed on 14 slots and the joint rule selected that slot on all 14. On the other 73 targets, joint ranking selected Shape's slot on 24, Surface's on 27 and a third slot on 22. No first-pass fallback occurred. These counts describe candidate choices on the L40S first pass. Historical Surface trajectories and scored natural completions remain separate measurements. No GPU allocation was requested. Shape 44946 was running at 1,267 records and combined 44948 was queued.

2026-11-22 01:34 UTC. Shape worker 1 reached COCOA. Checked the first available unseen, polygon-annotated records outside the 204-target cohort as soon as one had a saved completion. The early inventory included the table in training image 013720 and cat in training image 018072. Both image identities are in the frozen unseen inventory. The table had 82.66% annotated occlusion, ended at target detection, and received geometric IoU zero. The cat had 4.06% annotated occlusion. A read-only SSH tar stream transferred the two result records and the cat's saved mask and RGBA completion to a temporary local directory. All four members matched independently fetched remote hashes, with canonical inventory SHA-256 37e91ee41d5e6669b0838c87807553d9546efbcd633570bb3e9ca43399a608bf. Checked the frozen inventory and polygon manifest hashes, both official input digests, individual polygon PNGs, the completion digest, binary alpha and recorded original dimensions.

The cat's 619-by-619 completion extends outside its 500-by-487 original frame at row -157 and column -120. Pasting its alpha into a blank original-sized image with Pillow and counting pixel intersection and union independently reproduced the evaluator's NumPy projection exactly (60,258 / 66,912). The two records check the slide 16 geometry and failure handling on newly processed targets. At 01:34:22 UTC job 44946 had 1,303 of 2,565 results, including five fresh COCOA cases and its retained cohort. It had 86 cached success scores. Combined 44948 was pending with 204 records and 87 caches. The two full audits and report 44950 were pending. The check used no GPU allocation.

2026-11-22 01:40 UTC. Checked sacct -X for jobs 44915, 44919 and 44921. The repeat Surface cohort, full PMC and full Surface allocations were canceled at zero elapsed seconds with no allocated GPUs. Replayed the two historical full registrations against the verified executions and summary, protocol, inputs, references, metric values and denominators. Both matched freshrefs-experiments.csv (SHA-256 97ccf0c59f13b4b33c8d718297f636204fecd6815432a0e825a0ff0793a482ed). A regression registered a synthetic full execution through the normal registry writer and rejected stale execution IDs, mismatched hashes, changed scores and denominators. WSL scripts/check.sh passed the Python suite, shell regressions and CPU smoke. Uploaded scripts/compare_shape_progressive.py via SCP and installed by a digest-checked rename, SHA-256 e2ec63ba11969293b79d496a2a0f5c910b9dc2d4dac603b95d9f5cb5d876659f. The pending CPU audits 44947 and 44949 will check and record each new registration. The 01:38 UTC collector observed 1,309 Shape records and 204 retained combined records. Slurm showed 44946 running and 44948 pending.

2026-11-22 01:46 UTC. Added the direct L40S Shape versus Shape + Surface appearance metrics used in the historical Surface report. The builder computes the paired metrics from the audited rows and checks the 204-target results against the independent image-cluster bootstrap replay. A separate read of both execution-hashed cohort score CSVs reproduced the four means, denominators and point gains to report precision. All 43 local report links resolved. WSL scripts/check.sh passed all 304 Python tests, Bash regressions and CPU smoke. Uploaded scripts/build_shape_report.py by SCP and verified the staged and installed digest 073024622e576e94ca7c7a430e576016b390a6bc287342bc3f944cf877db2b90. From the cluster project, sbatch --parsable -p cpu -c 2 --mem=3G --time=00:10:00 -J shape-appearance -o output/shape-20261121/progressive-l40s/shape-appearance-%j.log --wrap '.amodal-cache/surface-metrics-env/bin/python scripts/build_shape_report.py --accounting' submitted job 45299. It completed in one second. The cluster report, SCP recovery and independent WSL render with actual sacct allocations matched SHA-256 390266121367308e27bd00b4c9751ad89bea941de691bc0bbd657669f34f6d48. At 01:45:55 UTC the live Shape run 44946 had 1,316 native records, including 79 COCOA targets. Combined 44948 remained dependency-pending with 204 records. The full audit and report dependencies were intact.

2026-11-22 01:56 UTC. Traced released metric alignment in evaluate_open_world.py to the cropped RGBA image. Discovered that the existing cross-device reference filter compared raw mask, full RGBA and cropped RGBA hashes together. Slurm job 45300 used srun -p cpu -c 2 --mem=3G --time=00:10:00 -J shape-scoring-crops .amodal-cache/surface-metrics-env/bin/python -u - from the remote project. It verified 1,020 native scoring crops across the five cohort snapshots against execution-hashed target records, the loaded alignment and the reference audit. All 433 success-cache signatures agreed. Of 204 cases, 166 matched on all three files, eight more matched on the scoring crop and 30 differed on the scoring crop. The Quadro and L40S crops for dog 108113 both hashed to b25e960c79581a03ff0ab8fbbdd8209de17158ca159c30d72fdddd93fea36451. Locally reconstructed the 174-target statistics from execution-hashed score CSVs with 5,000 source-stratified bootstrap draws of image multiplicities. All three means, relative differences and interval endpoints matched the cohort table to report precision. CPU job 45300 completed in nine seconds. Cluster, SCP and WSL report hashes match 774779eca7630c45676bb3527189f6895d7c3d1f37d1341d24f7641c7998ee74. All 43 local links resolve. WSL scripts/check.sh passed 305 Python tests, Bash regressions and CPU smoke. Uploaded and digest-checked scripts/compare_shape_progressive.py (ca194f5a22e7095fb5f116959644168d035e4ef0712921185d31d16e2dfbe80c) and scripts/build_shape_report.py (cee621475b69b3b81d387557cc541e1b1e3fdc99f4194c331c230367799a2b42). Pending audits 44947 and 44949 will retain the historical crop digest for each target and calculate the separate same-scoring-crop LPIPS result. At 01:56:19 UTC Shape 44946 was running with 1,339 records. Combined 44948 was dependency-pending with 204. No new GPU job was submitted.

2026-11-22 02:00 UTC. Began the local full Shape selector review using the first 1,024 targets in official annotation order. All 1,024 had native result records in active execution 20261121T145149.791453Z-01089952. Of these, 92 were frozen cohort records and their live result hashes matched the completed local cohort execution. CPU job 45301 ran srun -p cpu -c 1 --mem=1G --time=00:15:00 -J shape-replay-inputs python3 - in the progressive-l40s directory to inventory the other 932 results and match decisions to the hashes in their target traces. Job 45302 streamed the explicit member list from the Shape run using srun -p cpu -c 1 --mem=1G --time=00:20:00 -J shape-replay-copy and tar --hard-dereference --no-recursion --null -czf - -T -. The CPU jobs completed in five and 19 seconds. Both used zero GPUs.

Verified 11,782 regular files in the tar stream against the native result, trace and decision digests, then reread the installed archive. It contains 32,320,890 uncompressed bytes. The 22,052,121-byte archive at output/shape-20261121/progressive-l40s/shape-selector-0001-1024.tar.gz has SHA-256 c3835fe7a39fc10cd8e08d3aef910c285e42317d9bad3714e20752a47f0810fe. The canonical sorted compact JSON member-to-hash map has SHA-256 9115b8ee5d7bc096b896df1a8c1500ebad7f86dc74ec19668f32da466685309a. The ordered 1,024-ID list hashes to 95de7c8a08d2bf86e450b1bdd7ace9b66e81aedc4aff7a18e08d66a17417ad6b. The sorted 1,024-entry target-ID to result.json digest map, including the 92 cohort records, hashes to 9d48cd9c284e0532e5a460080ee674bea7c14f384576725a7ec1bb3e9592bb0b. Compare the same map to the final full execution before treating this as verified reuse. Digests use json.dumps(value,sort_keys=True,separators=(',',':')).encode().

Checked the frozen checkpoint and training receipt, all fresh target seeds, source hashes, labels and image identities. Independently counted prior-mask intersection and union for each nonempty candidate and replayed seed-order ties and fallback from the saved binary masks. All 8,680 candidate scores and 1,085 selected slots agreed across the 932 fresh records. There were 62 empty candidate masks and one all-empty fallback. For every pass the selected raw RGB digest in the decision equaled the winning candidate's recorded RGB digest. The archive retains masks and JSON; native RGB is verified on the cluster in the full audit. At 02:00:20 UTC Slurm still had Shape 44946 running with 1,349 native records, combined 44948 pending with its frozen 204 records and full report 44950 pending.

2026-11-22 02:11 UTC. Prepared full metadata recovery on the same basis as the completed Surface review. Uploaded scripts/export_shape_metadata.py with SCP and verified SHA-256 3ad0439eaff8cb70144f66d0d0cff00b19cb8def4491f6a6299ff88fb38b14dd. The CPU exporter requires complete execution provenance, checks each target's decision hashes and successful score-cache inputs, and archives the target records, scores, decisions, run configuration, score table and provenance files. Every archived byte stream has a separate native SHA-256. The exporter publishes each finished archive in one operation.

From the cluster project, submitted CPU-only jobs 45303 (afterok:44947, Shape) and 45304 (afterok:44949, combined). Exact commands are in output/shape-20261121/progressive-l40s/metadata-export-jobs.tsv, SHA-256 c05652f666ca955764028db46b6c2ef975ae42b5ef12ed3eeeaa5ac3329da1d2. The local collector now retrieves each completed export against its archive digest and waits for both exports and the full report. Switched the local watcher from PID 19700 to 33200, verified the new PID, command line and status.json with both export jobs. GPU jobs and their dependencies were untouched.

Exercised the exporter on both real cohort snapshots, on Windows and in cluster CPU job 45305 (srun -p cpu -c 1 --mem=1G --time=00:05:00 -J shape-export-check .amodal-cache/surface-metrics-env/bin/python -). Both machines reread all archive member bytes and reproduced the same manifests. Shape had 503 files, including 158 decisions and 86 score caches, manifest digest 223105375351ef4bb313c3741181838cc21f932e6bb1b55560bc951813157af6. Combined had 496 files, including 150 decisions and 87 score caches, digest f11482146429f67c9c1517f538bc074d5693a596b859b54863eeab3734b2fd20. Both restored local cohort execution provenance checks passed. Job 45305 completed in five seconds on one CPU. WSL scripts/check.sh passed 307 Python tests, shell regressions and CPU smoke. At 02:11:20 UTC, Shape 44946 had 1,360 native results, and all four worker logs had updated two seconds earlier. Combined 44948 was waiting with 204 records. Full exports, registrations and report verification await the two GPU evaluations. After retrieval, compare the first 1,024 Shape result hashes to the digest recorded at 02:00 and verify all final paired rows and reference transfers.

2026-11-22 02:19 UTC. Locally replayed the natural COCOA outline scores for the frozen cohort. CPU job 45306 streamed the exact missing native completion.png and mask.png files from the six successful L40S PMC and seven successful Shape and combined targets with polygon matches (40 files, 10,728,346 bytes). Ran from the cluster project with srun -p cpu -c 1 --mem=512M --time=00:05:00 -J shape-cohort-outlines and tar --null --verbatim-files-from --no-recursion --hard-dereference -cf - -T -. The seven historical Surface pairs were already local. The sorted restored path-to-SHA-256 map hashes to 94744814299e70e22f1b624f5a175ccd5f7e6a04a418b1b4b0e89436d940274f.

Verified every compared result against its completed execution, each successful RGBA image against its completion_sha256, the binary mask against alpha, and each encoded annotation against its manifest. Independently pasted alpha into a blank original-size image at (column,row), clipped at the frame and counted intersection and union. All 88 target/method IoUs matched their audited values, including zeros on failed targets. The four methods scored 0.23020838 (L40S PMC), 0.26312102 (Shape), 0.24816440 (combined) and 0.26760592 (historical Quadro Surface). For Shape and combined, the same seven targets succeeded. The net difference was -0.01495662, nominal 95% interval [-0.04588473,+0.00195945] from 5,000 paired draws of 20 image IDs. The dog contributed -0.01264799 of the mean difference, or 84.6% of the observed net decline. The per-target receipt output/shape-20261121/cohort-cocoa-native-replay.json hashes to 1c2c8e823cf7e53b23d5b28c90717ca009dd80f932076c944585a6754b254d75.

Added this result to the slide 19 discussion. CPU job 45307 ran .amodal-cache/surface-metrics-env/bin/python scripts/build_shape_report.py --accounting via srun -p cpu -c 2 --mem=3G --time=00:10:00 -J shape-polygon-report. The builder SHA-256 is fece5330cb8345b79dc5809bcb6081de53ffc174d41aa3b5aab5e3c47ba82ced. The cluster render, SCP recovery and independent WSL render with the three sacct rows all matched report digest 368a49952c715c79405332ef526c2a7644ad56f49bbe3e21cda49c6d5b09cfc6. The prior report bytes were reproduced after removing only the new paragraph. All local report links resolve and 28 Shape regressions passed. At 02:19:09 UTC Slurm showed Shape 44946 running with 1,374 target records and combined 44948 waiting with 204. Neither full result had been registered. For the final 321-target local geometry review, recover successful native completion alpha from the closed full runs after transfer of metadata and compare directly to the archived annotations. The GPU evaluations and their dependencies were unchanged.

2026-11-22 02:30 UTC. Added scripts/replay_shape_outlines.py to the local collection workflow for the remaining full COCOA audit. After each arm's paired table is available, the collector checks its digest, the completed execution digest, official target order and frozen polygon manifest. It recovers missing result.json files for the 321 joins and each successful native completion PNG via read-only SSH tar, checking the bytes against the closed execution and its target records. The independent Pillow calculation pastes binary alpha at the recorded column and row, clips to the original frame and counts intersection and union. Each failure retains zero IoU. The per-arm receipt is written to output/shape-20261121/{shape,combined}-full-cocoa-native-replay.json only after every score matches. The two full receipts are pending.

Ran python scripts/replay_shape_outlines.py shape --cohort and the corresponding combined command. Both arms matched the independent 22-target cohort receipt on every status and IoU, including seven successful completions per arm. A temporary local recovery of the native combined-cohort book record and completion from COCO test image 296835 matched SHA-256 9efeca88cf5743dc42168647d5ded6824343d4390ec6c547ad72e49e2142a7df and 02f2795769306ccd542f631297f52a1fab9cfe067a0661ae32dc9c72ba93e87c. WSL scripts/check.sh passed 309 Python tests, shell regressions and CPU smoke. No Slurm allocation was requested for this check.

Started python -u scripts/collect_shape_results.py --watch under Windows PID 7548, verified its first completed poll and then retired the previous idle collector PID 33200. Its logs are collector-outlines.log and collector-outlines.err in the local study directory. At 02:30:06 UTC Shape job 44946 was running with 1,385/2,565 native results and all four current worker logs updated four seconds earlier. Combined 44948 was waiting with 204 retained records. Full audits, metadata exports, registrations and the final report remain pending. The repeat PMC and Surface jobs 44915, 44919 and 44921 show zero allocation.

2026-11-22 02:35 UTC. Independently checked the six currently populated slide 15 rows in output/shape-20261121/REPORT.md. Read the native execution-hashed per_target.csv for historical full Quadro PMC and Surface, historical cohort Surface, and the three closed L40S cohorts. Verified official order, fixed LPIPS/SSIM penalties on every failed target, arithmetic means and saved counts. Queried sacct -X -nP -j 39486,39487,39488,39490,39491,44902,44903,44905,44913,44914,44915,44917,44919,44921 --format=JobIDRaw,JobName,State%32,ElapsedRaw,AllocTRES. The published full costs follow from 22,137 wall seconds for PMC and 104,765 for Surface, each on four GPUs. Surface includes 86,419 seconds in its timed-out allocation. The completed L40S Shape and combined cohorts used 6,593 and 6,599 seconds on four GPUs. Separate setup, failed-attempt and cancellation rows matched the saved accounting. All six rows matched the report's displayed precision. The receipt is output/shape-20261121/slide-15-cohort-and-controls.json (SHA-256 30746fcd4f2a2cc6883e93b9db415f1f3e5ff735f65e1cd449893af668545c20). This check read existing outputs and used no Slurm allocation.

At 02:35:27 UTC, Shape 44946 was running with 1,392 ordered-manifest target results counted on disk and 12:15:44 left in the original allocation. The four active worker logs had updated within ten seconds. Combined 44948 was waiting on its dependency with 204 cohort records retained. The local data collector remained live under PID 7548. Full Shape and combined score and cost rows still await their registered closed runs.

2026-11-22 02:41 UTC. Added a comparison against the independent first-1,024 Shape selector review to the pending full Shape audit. Re-read the archived 11,782 members of shape-selector-0001-1024.tar.gz and verified its SHA-256 and complete member inventory. The archive contains the 932 new target records. verify_provenance verified the remaining 92 records from the immutable Shape cohort snapshot. Together they exactly cover the first 1,024 official IDs. Their ordered-ID digest is 95de7c8a08d2bf86e450b1bdd7ace9b66e81aedc4aff7a18e08d66a17417ad6b and their ID-to-result-digest map hashes to 9d48cd9c284e0532e5a460080ee674bea7c14f384576725a7ec1bb3e9592bb0b.

Staged scripts/compare_shape_progressive.py via SCP to scripts/.compare_shape_progressive-5c5a293618ed83b4.py. From the remote project ran srun -p cpu -c 1 --mem=1G --time=00:05:00 -J shape-check-records .amodal-cache/surface-metrics-env/bin/python - as CPU job 45308. The 1,024 current native Shape files still matched that same digest. The 92 cohort rows matched both the completed execution and the live native files. The staged check correctly rejected the still-running full execution. Job 45308 completed in two seconds with one CPU and zero GPUs.

The full audit will verify the target order and record digests against the closed execution and publish early_selector_snapshot in shape-full-summary.json. Verified that both downstream audits were pending, then atomically installed the CPU audit at SHA-256 5c5a293618ed83b4217591a1ead339016f8e76c5d04a8f86956404874eb9abef. WSL scripts/check.sh passed 310 Python tests, Bash regressions and CPU smoke. At 02:40:58 UTC, Shape 44946 was running with 1,416 native records and 86 cached cohort scores. Combined 44948 remained pending with 204 cohort records and 87 scores. No new model evaluation was submitted. The full registrations, closed-run comparisons and final report are pending.

2026-11-22 02:45 UTC. Updated the planning estimate for the current Shape allocation. CPU job 45309 read completed result timestamps and Shape pass counts via srun -p cpu -c 1 --mem=1G --time=00:05:00 -J shape-inference-eta python3 - from the remote project. It completed in two seconds. Locally joined to the official four image partitions, excluded their 204 frozen cohort records and used prior Surface pass counts from the registered comparison to estimate the remaining work. All observed fresh records formed contiguous prefixes of their assigned partitions. At 09:42:58 Vietnam time, there were 1,212 fresh results (1,164 VG and 48 COCOA). Workers had completed 280/598, 328/582, 312/599 and 292/582 fresh targets. An ordinary least-squares fit to consecutive result timestamps, excluding the first result per worker, gave 5.32 s per target, 114.76 s for the first eight-candidate pass and 114.56 s per further pass. Anchoring on each worker's last completed target and using historical remaining pass counts projected the last inference record at 22:15:55 Vietnam time on November 22. The original allocation ends at 21:51:11. Scoring and verification are additional.

This is a planning estimate and uses another method's pass counts for unprocessed targets. Backtesting on 277 targets completed since the prior 00:03 UTC estimate, actual time divided by predicted time was 1.227, 0.963, 1.379 and 1.054 on the four workers. The benchmark has yet to supply fresh timing for free_images and laion. The estimate does not trigger a retry. Verified that snapshot 45134 follows 44946 via afterany and controller 45142 follows 45134 via afterok. Shape audit 44947 and combined 44948 are each eligible after either the original successful Shape run or its recovery chain. Corresponding combined snapshot 45135, controller 45143, audits, exports and report retain their dependencies. Cluster snapshot, resume and run scripts matched the local SHA-256 values. At 02:45:06 UTC Shape was running with 1,419 native results and 86 cached cohort scores. All four worker logs had updated within one second, and collector 7548 was live. Combined retained 204 targets while waiting. No GPU submission or dependency change was made.

2026-11-22 02:54 UTC. Added the local review of full Shape metadata exports to the existing collector. The verifier checks each archive's SHA-256 against the export log, extracts member-hashed files into a fresh temporary directory, verifies the closed execution and source snapshot and matches the official target order. It compares each successful cache's reference and completion hashes and all four metric values with the published CSV and paired audit. Failed rows retain the benchmark penalties. It replays each stored selector score vector, checks the progressive seeds, trace digest, selected slot and fallback, and verifies archive target membership. After each full export, the collector writes {shape,combined}-full-metadata-replay.json. The separate 321-target native outline review still runs for each arm.

Generated temporary exports of both real cohort snapshots and applied the same archive verifier to their audited paired tables. Shape matched 204 rows, 86 success caches and 158 decisions (two fallbacks) across 503 files, with manifest SHA-256 223105375351ef4bb313c3741181838cc21f932e6bb1b55560bc951813157af6. Combined matched 204 rows, 87 success caches and 150 decisions (one fallback) across 496 files, with manifest SHA-256 f11482146429f67c9c1517f538bc074d5693a596b859b54863eeab3734b2fd20. The Windows focused tests passed. WSL PYTHON_BIN=$HOME/.cache/surface-replay-py310/bin/python bash scripts/check.sh passed 311 Python tests, Bash regressions and CPU smoke.

The exporter now recognizes the released evaluator's CLIP-only successful cache signature, which uses the full reference. Staged scripts/export_shape_metadata.py with scp to scripts/.export_shape_metadata-98385a2504b8.py on the cluster. Checked both CPU export jobs 45303 and 45304 were pending and the installed and staged SHA-256 values matched before replacing the exporter. Installed SHA-256 98385a2504b8e2dea12b10d610f1dbe4ed58923cc3411a50ea1a49d9e8ee41dc. Started python -u scripts/collect_shape_results.py --watch under Windows PID 26012, confirmed its completed poll and export job mapping, then retired old watcher PID 7548. Logs are collector-archive-replay.log and collector-archive-replay.err under the study directory. No Slurm allocation or dependency change was needed.

At 02:54:11 UTC, Slurm showed Shape 44946 running with 11:57:00 left. It had 1,426 native records including the cohort and 86 retained score caches. All four worker logs had updated within three seconds. Combined 44948 retained 204 records and 87 caches and was waiting on its dependency. The report still hashes to 368a49952c715c79405332ef526c2a7644ad56f49bbe3e21cda49c6d5b09cfc6. Full comparisons remain open.

2026-11-22 02:59 UTC. The earlier report page references came from the 20-slide November 16 export. Fetched the current Canva design DAHbEx8zTOs read-only and matched the content by page to the November 21 PowerPoint in output/week8-problem-statement-20261121 (SHA-256 103a3f0941773473dabc6ac2dac46a9f0872e24a3a1c26feb39ad6047cfd05df). Canva returned revision 19, 21 pages. Its new problem statement occupies page 2. Page 9 describes the original 3,000-case reproduction gap, pages 11–12 and 15 describe Shape and joint selection, page 16 contains the full results table, page 17 the COCOA severity chart, page 18 the umbrella and bus, page 19 the controlled ablation, and pages 20–21 the limitations and conclusion. The current deck still shows the Shape placeholders. The actual slide deck was read only. The chart on PowerPoint page 17 matches the chart from the earlier page 16 pixel for pixel (2063 × 963 RGB, SHA-256 aafff7c471e81121d38116c0c592285e097144322c2a9c9e7b03103ac742fbef). All eight embedded images on page 18 match the previous qualitative page. All eight historical metrics, counts and costs for the two control rows remain on page 16. The frozen experiment contract and evidence filenames retain their original numbering.

Updated scripts/build_shape_report.py, SHA-256 6c300df7ddbaf47975109aae4866f14226a1a60a084ab6aa09271896b2ce1b18, to use the live page numbers in the published and pending sections. Deployed the builder via scp to scripts/.build_shape_report-6c300df7ddbaf479.py, checked the previous builder and report digests, then installed it. Submitted from the cluster project with sbatch --parsable -p cpu -c 2 --mem=3G --time=00:10:00 -J shape-slide-pages -o output/shape-20261121/progressive-l40s/slide-pages-%j.log --wrap '.amodal-cache/surface-metrics-env/bin/python scripts/build_shape_report.py --accounting'. Job 45310 completed in two seconds with two CPUs. The cluster report, SCP copy and independent WSL render with actual sacct rows for 44913/44914/44917 matched SHA-256 e8f96c474b80ef46d000699ce22c9c033e63e00665e012ff6b749026e113692a. Compared all 72 Markdown table lines and all figure references with the previous report, verified 44 local links and checked the pending Vietnamese conclusion's reference to page 16. All 29 tests in tests/test_shape.py passed in WSL.

At 02:59:19 UTC, Slurm showed 44946 running with four L40S GPUs. The Shape directory contained 1,435 native results and the 86 retained cohort score caches. All four active worker logs had updated within four seconds. Combined 44948 was pending with 204 retained records and 87 cached scores. Local collector 26012 was live. Full scores for slide 16, the new severity rows for slide 17 and the two full audits remain pending.

2026-11-22 03:10 UTC. Verified the cross-device reference inventory from native PNGs in CPU job 45311. From the remote project, the command was `srun -p cpu -c 1 --mem=2G --time=00:12:00 -J shape-reference-overlap .amodal-cache/surface-metrics-env/bin/python -`. Slurm recorded 57 seconds and one CPU. The check covered the first 1,024 official targets and the 204 retained targets. The groups shared 92 IDs. For each of the 1,136 unique IDs, it hashed the source image, the raw SAM mask, full RGBA image and scoring crop in each of the three runs. Both registered Quadro controls had identical reference records. All stored target records agreed with their input and reference hashes, and all three target seeds matched. Historical target record hashes matched their closed executions. The 204 retained Shape records matched the cohort execution. The first 1,024 Shape records matched the prior ordered result map.

In the first 1,024, 803 matched on all three reference files and 850 matched on the scoring crop. The ordered matching-ID hashes reproduced the prior independent local check. The 850 identical crops included 620 ready references and 230 empty masks. Across the 1,136 unique IDs, 895 matched on all files and 948 on the crop. Four targets differed in original-frame crop geometry. All four were outside the matched-crop subset. In the frozen cohort, 174 crops matched. Looking up those IDs in the five native score CSVs showed 100 ready references and 74 empty masks. On all five runs the latter had 47 ambiguous and 27 undetected targets and each received LPIPS 1.0. These are reference and denominator checks. Full score estimates still await the closed runs. The per-ID receipt is [shape-reference-overlap-1136.json](output/shape-20261121/progressive-l40s/shape-reference-overlap-1136.json), SHA-256 `404e693ff3d421fd44693e6cd0a0a8541aff589ef56b2d26f5e870ea64271c0b`. Reconcile both full paired tables with its per-ID flags after export.

At 03:10:36 UTC, Shape job 44946 had 1,451/2,565 native result records and was running on gus. Combined job 44948 was dependency-pending with its 204 retained records. Collector 26012 was live. Full registrations and slide 16–17 results are pending.

2026-11-22 03:20 UTC. Added a local numerical replay of the final slide 16 and 17 tables to the collector. It runs after the full report and metadata exports are complete. It reads the official target order and execution-hashed score CSVs, and takes the 321 target memberships and occlusion ratios from the earlier presentation's severity-joined.csv. Original-frame IoUs come from the historical polygon audit and the two independent native L40S outline replays. Slurm supplies the actual elapsed seconds and GPU allocations, including retained cohorts and timeout continuations. The replay checks the report's devices, metric columns, rounded cells and per-bin saved counts, and verifies the severity CSV at full precision. It will publish full-slide-table-replay.json after the comparisons pass. Visual inspection of the finished chart and verification of the full paired intervals remain pending.

Ran the historical part on both official 2,565-row score tables and queried sacct for jobs 39486, 39487, 39488, 39490 and 39491. All 642 published per-target COCOA LPIPS/status pairs, ten previously charted LPIPS means and ten native mean IoUs agreed. The five groups contained 107/34/23/17/2 saved PMC outputs and 109/33/23/16/2 saved Surface outputs. Costs were 24.596667 and 116.405556 GPU-hours. The final-table tests rejected changed costs, method headings, exported scores and saved counts. WSL scripts/check.sh passed the Python suite, Bash regressions and CPU smoke.

Started the updated local collector with `python -u scripts/collect_shape_results.py --watch`, confirmed its first poll and both export dependencies, and retired PID 26012. Watcher PID 15260 writes collector-slide-tables.log and collector-slide-tables.err under output/shape-20261121. The checker and collector are local. At 03:20:27 UTC, Shape 44946 had 1,471/2,565 native records, and all four current worker logs had updated within three seconds. Combined 44948 remained dependency-pending with its 204 retained records. No new Slurm allocation was submitted.

2026-11-22 03:34 UTC. Reconstructed all 183 archived training mask crops from their original Grounded SAM references, verified image, label and checkpoint hashes, and independently reproduced each pixel intersection and union for the 25 exact-ID COCOA matches after the 256 by 256 training resize. Receipt SHA-256 is 578c2592aec952ef1fdab7e44ffee910d7df3b6d8ecba6c504e500156552ccc6. Transferred training-mask-audit.json to output/shape-20261121/ and build_shape_report.py to scripts/ on the cluster. Ran `srun -p cpu -c 2 --mem=4G --time=00:15:00 -J shape-report-labels -v .amodal-cache/surface-metrics-env/bin/python scripts/build_shape_report.py --accounting` from the project. Slurm job 45316 completed in two seconds. The local and cluster report hashes matched b64359abfd605c69710a38a3b481061e0a2f4721bcbdde3b5d47123a60a501db. All 72 existing table lines retained SHA-256 2f813fad4ce64c1b6cb64bee0d6672523615593a721df4dbac4bd561a3e594f2 and all 45 local links resolved.

Replayed reference membership on the 204 frozen targets from the four historical PMC/Surface and L40S Shape/combined cohort executions. All 166 exact-reference and 174 scoring-crop IDs agreed with the native PNG audit. The separate bootstrap implementation reproduced the two published Surface-versus-method exact-reference estimates. The trial rejected a changed membership flag and a changed estimate. Uploaded scripts/replay_shape_reference_intervals.py and progressive-l40s/shape-reference-overlap-1136.json. Queued CPU job 45320 with `sbatch --parsable -p cpu -c 2 --mem=4G --time=00:30:00 --dependency=afterok:44947:44949 --kill-on-invalid-dep=yes -J shape-ref-intervals -o output/shape-20261121/progressive-l40s/shape-reference-intervals-%j.log --wrap '.amodal-cache/surface-metrics-env/bin/python scripts/replay_shape_reference_intervals.py'`. After both audits, retrieve output/shape-20261121/full-reference-interval-replay.json and its log and reconcile the receipt with both full paired-table hashes. At 03:34:24 UTC, Slurm showed Shape 44946 running and combined 44948 dependency-pending. Shape had 1,491 native records and the local collector PID 15260 was live.
WSL scripts/check.sh passed the Python tests, Bash regressions and CPU smoke after the reference-group regression was added.

2026-11-22 03:39 UTC. Verified the 12 image tiles for the Week 8 umbrella and bus against native RGB from the frozen candidate pool, observed images and pre-occlusion crops. The umbrella used seeds 11, 44, 33 and 33 across PMC, Shape, Surface and combined. The bus used 11, 11, 55 and 11. Added Vietnamese captions with the native IoU and object LPIPS values to the slide 18 report section and the equal-budget interpretation to slide 19. CPU job 45324 ran `srun -p cpu -c 2 --mem=4G --time=00:15:00 -J shape-slide-captions -v .amodal-cache/surface-metrics-env/bin/python scripts/build_shape_report.py --accounting`. CPU job 45325 ran `srun -p cpu -c 1 --mem=1G --time=00:03:00 -v -J shape-slide-panel .amodal-cache/surface-metrics-env/bin/python -` with the native tile comparison. Both completed. Local and remote REPORT.md SHA-256 matched a59b6525cf3a255f4c33b24a9420f7d888bfe16a27fb83cc28eb360480d3ef8d. All 47 local evidence links resolved and the 72 table lines matched their previous hash. Shape 44946 had 1,494 stored results at 03:39:35 UTC, combined 44948 remained dependency-pending and collector 15260 was live.

2026-11-22 03:44 UTC. Added the already submitted CPU job 45320 to progressive-l40s/submissions.tsv under the coordinator lock and confirmed its existing dependencies on audits 44947 and 44949. The local collector now waits for this job, retrieves full-reference-interval-replay.json and the Slurm log, matches the four execution hashes and both paired-table hashes, and checks all eight LPIPS estimates against the two full summaries. No allocation was submitted. An isolated test with the four closed 204-target runs reproduced the eight intervals on the 166 identical-reference and 174 identical-crop targets. The check rejected a modified confidence interval and a changed paired-table hash. WSL scripts/check.sh passed the Python suite, Bash regressions and CPU smoke.

Started python -u scripts/collect_shape_results.py --watch as local PID 7768, verified its first live poll and interval job ID, and then retired the previous local collector PID 15260. Its logs are output/shape-20261121/collector-intervals.log and collector-intervals.err. The collector counted 1,504 Shape results at 03:43:42 UTC. Combined had 204 retained records. The verified full scores, final chart inspection and final registration review remain open.

2026-11-22 03:50 UTC. All 1,294 official VG target records were present in running Shape job 44946. CPU job 45326 ran `srun -p cpu -c 1 --mem=2G --time=00:12:00 -J shape-vg-coverage -v .amodal-cache/surface-metrics-env/bin/python -` from the remote project. It verified each Shape result's identity, all 724 saved native RGBA images and hashes, the 109 retained cohort hashes, and each historical control record and status against its registered execution and score CSV. It checked all 1,294 Shape record hashes again after reading the images. The first 1,024 matched the previously frozen hash map. Slurm recorded one CPU for 21 seconds. Local checks confirmed the official order, 1,234 images, both original execution and score hashes, all retained VG record hashes and every paired coverage count. Receipt SHA-256 0f46cbd326eb996c97566161fd46c5dbd2fa5b4826782a63a99c85388ad844bc is at [shape-vg-inference-coverage.json](output/shape-20261121/progressive-l40s/shape-vg-inference-coverage.json). Compare its per-ID Shape digests against the closed execution when available. The VG saved counts are Shape 724, historical PMC 726 and historical Surface 729. No full Shape metric or registration was available. At 03:50:04 UTC Shape had 1,510 total target records and combined remained queued.

2026-11-22 03:59 UTC. Slurm CPU job 45327 read the saved Visual Genome outputs from running Shape job 44946. From the remote project, the command was `PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 srun -v -p cpu -c 1 --mem=4G --time=00:20:00 -J shape-vg-ssim .amodal-cache/conda-env/bin/python -`. Slurm recorded one CPU for 124 seconds. The script checked all 1,294 native result hashes against the frozen VG inventory, checked the reference crop and completion hashes on 724 successes, and reproduced the released evaluator's SSIM with the prediction data range, adaptive window and exact-copy penalty. All 109 retained scores matched the frozen cohort within 1e-12. The original historical CSVs supplied all control scores. SCP recovered progressive-l40s/shape-vg-ssim.json with SHA-256 20f6d153315a672e337807f595c3418898dec9b11b0264a499624804e78c1af4. Local checks reconciled official order, statuses, all denominators, means, both historical executions and both historical CSV hashes. The archived paired bootstrap on 1,234 image identities gave Shape-minus-PMC SSIM -0.003469 [-0.007877, +0.000722] and Shape-minus-Surface -0.011514 [-0.015839, -0.007664]. Compare the 1,294 per-target SSIM values and 724 saved crop/completion hashes with the closed Shape execution and full score table. Shape had 1,517 records and combined was dependency-pending. Both full registrations remain open.

2026-11-22 04:05 UTC. The scheduled Shape full audit 44947 now compares every official VG result hash, status and scored SSIM with the frozen CPU receipts (45326/45327). It also checks the crop and completion hashes of each saved VG prediction. The full comparison summary will record the replay in vg_ssim_replay. A real replay against the closed Shape cohort matched all 109 VG targets, including both image hashes on its 48 successes, with score CSV SHA-256 d43739dec2ee8ff63cea3e404eba7e892d4c1cfdee70f367d421aefafeeb781f. The focused regression covers changed inputs, score, order and native record. The 32 focused tests and the WSL Python suite, shell regressions and smoke check passed. SCP staged scripts/.compare_shape_progressive-5be48cd75472.py, then the cluster install checked the previous digest, verified both pending audits and installed SHA-256 5be48cd7547260b62b4b48d716a9c562d6fa46e31fb7e1a7eba1297070289795. Both L40S jobs kept their original IDs and dependencies.

SCP staged scripts/.build_shape_report-0e7447d59264.py and installed it after checking the old builder and report hash. CPU job 45331 ran `srun -v -p cpu -c 1 --mem=3G --time=00:10:00 -J shape-vg-report .amodal-cache/surface-metrics-env/bin/python scripts/build_shape_report.py --accounting` and finished in two seconds. The updated report labels the 1,294-target VG-only SSIM comparison in Vietnamese under the pending full table. Its local and remote SHA-256 is a2491c71d0f4a18770855dbe37b5e199eafaa48b94818ad222fa35920a7d81e7. Removing that single paragraph recovers the previous byte-for-byte report. All 48 local links resolved and all 72 table lines were preserved. At 04:05 UTC the four Shape worker logs were current with 1,523 result records. Combined was dependency-pending.

2026-11-22 04:08 UTC. CPU job 45332 ran `srun -p cpu -c 1 --mem=1G --time=00:05:00 -J shape-remaining python3 -` from the remote project. The 1,321 newly inferred Shape targets and 204 retained targets gave 1,525 records. Each worker's new targets formed a contiguous prefix of its official shard. The last completed target timestamps, a fit of 5.24 seconds per target plus 114.64 seconds for the first Shape pool and 114.40 seconds for each further pool, and historical Surface pass counts for the remaining targets yielded estimated worker finish times of 21:59, 19:36, 22:10 and 20:41 Vietnam time. The allocation ends at 21:51:11. A sensitivity calculation adding the observed within-source difference in Shape versus Surface pass costs gave a latest worker finish of 23:38. Only VG (1,185 fresh targets) and COCOA (136) had fresh observations, so that adjustment uses no data for the other two sources. These estimates cover inference only and depend on how many passes the remaining objects require. Slurm had not timed out. The existing snapshot and recovery dependencies remain in place. Use the final full scores and actual Slurm elapsed times for the report.

2026-11-22 04:18 UTC. Section 2.5 of the original chat requests amodal IoU and hidden RGB metrics at K=4. Joined the 32 independent polygon rows to the frozen Shape decisions and hidden-core DreamSim records by target ID and decision SHA-256. Recomputed uniform and selected scores from candidate arrays, then independently checked the four-candidate policy summaries and 5,000 paired image-bootstrap draws against them to 1e-12. scripts/build_shape_report.py now includes the two appearance rows in the Vietnamese controlled COCOA section. Script SHA-256 e621faa25dcedab1f9a84bb051066f53e7845d5135c8231a84c146af0625203f. Both staged and original builder digests were checked before installation. CPU job 45333 ran `srun -v -p cpu -c 1 --mem=3G --time=00:10:00 -J shape-cocoa-report .amodal-cache/surface-metrics-env/bin/python scripts/build_shape_report.py --accounting` in two seconds. Retrieved REPORT.md SHA-256 976a766690bc9701101595cd20f6d020eafd89b4de21a0065201b96ed9bfe228. All 50 local evidence links resolve. Removing the new section recovers the previous report digest. The focused Windows suite passed 33 tests. Full scores and registrations are pending.
WSL scripts/check.sh completed successfully after this report change, including the Python suite, shell regressions and CPU smoke.

2026-11-22 05:10 UTC. Checked the request to reuse the historical controls against `output/shape-20261121/progressive-l40s/setup-accounting.psv`. Slurm recorded zero elapsed seconds and no allocated resources for the canceled full Surface job 44921, full PMC job 44919 and Surface cohort job 44915. Recomputed the hardware comparison from both 204-row PMC `per_target.csv` files after verifying their SHA-256 against the respective `execution.json` manifests. L40S minus Quadro mean LPIPS was +0.00399675. Dog 108113 contributed 97.085% of the difference. Across the other 203 targets the mean difference was +0.00011709. These runs do not establish why segmentation failed for the dog.

Direct SSH to `slurm.csse.rose-hulman.edu` and alias `hpc` failed at DNS resolution. WSL `getent ahostsv4` also failed. Windows showed the PANGP adapter disabled. Collector PID 7768 was live and is configured to retry. Fresh Slurm states for Shape 44946 and combined 44948 were unavailable. No job was submitted or restarted. The last successful collector poll was 04:23:42 UTC.

2026-11-22 05:17 UTC. DNS and SSH to the Slurm login recovered. Job 44946 was running on gus with all four worker logs current. Job 44948 and the downstream audits, exports and recovery controllers retained their dependencies. The existing collector resumed at 05:13 with 1,642 Shape targets and at 05:16 counted 1,648. The earlier outage did not require a new inference submission.

Reconciled all 1,294 Visual Genome target IDs, statuses and SSIM measurements with both historical controls. Verified SHA-256 for the native coverage/SSIM receipts and both controls’ execution manifests and scored CSVs. The report builder now gives mean SSIM on the 722 outputs shared by Shape and Surface alongside the failure-inclusive means. SCP staged `scripts/.build_shape_report-e6dddfe120b4.py`; the install verified previous and staged hashes before replacing the builder. CPU-only job 45335 ran `srun -v -p cpu -c 1 --mem=3G --time=00:10:00 -J shape-vg-report .amodal-cache/surface-metrics-env/bin/python scripts/build_shape_report.py --accounting` and completed in two seconds. Local and remote REPORT.md have SHA-256 d50b59f72e2c467457b97f9582f516c566351c2f89d6d972c57984e9afe8a93f. All 51 local evidence links resolve and the 76 table lines match the previous version. Windows ran 34 focused tests. The first WSL invocation used a Python without NumPy; with `PYTHON_BIN=$HOME/.cache/surface-replay-py310/bin/python`, `scripts/check.sh` passed 317 Python tests, the shell regressions and CPU smoke.

2026-11-22 05:21 UTC. Read-only SSH to the project login counted 1,661 Shape records while 44946 was running. All four worker logs were updating. Combined 44948 remained dependency-pending. The local collector (PID 7768) was live. Three of the four COCOA objects at 80–100% annotated occlusion had Shape result.json files. The fourth, elephant 073268, had 72 earlier outstanding targets in its official worker shard at the time of inspection.

Retrieved the three completed records, the native Shape completion for plate 060172 and the two historical completions with scripts.replay_shape_outlines.ensure_files and verified SHA-256 on every transfer. The Shape plate result hash is 0aaae8bd3f654af0989cde9a72cdaf35dcf7f9cdeebd05e4c194d0310287a744, and the RGBA hash is fc0180d5ac2d0f5a604a31e629842b75f81b997cd64dcb570901c168e4a55574. Ground truth came from the COCOA manifest f0e0bc47f7f4fdf9fb0cb8f6388826e16c06b6a670fd40a95402e6d091e57a9c, checking the file digest. Compared each PNG in its original image frame against the independent outline via both PIL and NumPy. Both historical target-record digests matched their closed execution manifests and the IoUs reproduced the historical audit exactly. The per-case Shape IoU and four-target upper bound are in discoveries.md. The transfer reused the local full-run directory so the closed outline replay can verify its copies against the eventual execution. No inference or scoring cache was changed.

2026-11-22 05:30 UTC. Added the natural severe-occlusion plate to the Week 8 report builder. It measures the native completion alpha inside the annotated original image frame, reports amodal IoU and the fraction of amodal pixels outside the visible annotation that the mask covers, and handles each new method when its target record exists. Checked the geometry manifest, annotation PNG and completion hashes. The Shape row independently reproduces 0.2061 IoU and 16,673 of 222,997 hidden pixels. A synthetic regression covers negative crop offsets, a nonzero alpha of one, a visible label outside the amodal outline, a failed output and a changed completion file. All 35 focused Windows tests passed with the deployed version.

CPU report job 45336 failed on a local replay-module import that was absent from the cluster. Replaced that dependency with the original-frame alpha projection in the builder, verified the prior and replacement hashes, and resubmitted the one-core CPU rendering command. Job 45337 completed in one second. Command: `srun -v -p cpu -c 1 --mem=3G --time=00:10:00 -J shape-plate-report .amodal-cache/surface-metrics-env/bin/python scripts/build_shape_report.py --accounting`. Installed builder SHA-256 8b4ead4a8fab16fa8e56ebc16076995ff4ce460095ef9159cfcdbdaf01bca154. The local and remote REPORT.md has SHA-256 21d4a8e755cd17315c77e3248ba88600b432c56962a62a51fd00f59f19397f85. Removing the new Vietnamese paragraph restores the previous report SHA-256 exactly; all 53 local links resolve. At 05:30 Slurm showed job 44946 running with all four logs updating, and job 44948 waiting on its dependency. No GPU allocation was requested for this report update.

2026-11-22 05:45 UTC. Slurm CPU job 45338 completed in 8 min 23 s with 8 CPUs allocated. It ran scripts/replay_shape_vg_lpips.py on saved L40S predictions using the released scorer. Recovered output/shape-20261121/progressive-l40s/shape-vg-lpips.json with SCP. The remote and local SHA-256 matched 767e5dd8f2e0f54a1a2b856a761fe0f22fb42cbfd48029d51154a04b9cf6816e. Verified the scorer and metric-module hashes, scoring configuration, original VG order, all 1,294 target statuses, all 570 LPIPS failure penalties and both control score tables against the closed executions. Verified all 109 retained L40S rows. The maximum absolute LPIPS deviation on the 48 saved retained predictions was 1.78814e-7. Locally replayed both complete 5,000-draw estimates with scripts.audit_surface_lastpass_result.paired_estimate, matching every field within 1e-10. The common-output breakdown is recorded in discoveries.md.

Staged the builder with scp scripts/build_shape_report.py hpc:Vu_Hung/cv_nangcao/scripts/.build_shape_report-ea55955efbdc.py, then installed it after checking the prior builder, staged builder, prior report and new receipt SHA-256. Final WSL check exited 0 with PYTHON_BIN=$HOME/.cache/surface-replay-py310/bin/python bash scripts/check.sh. From the remote project, sbatch -p cpu -c 1 --mem=3G --time=00:10:00 -J shape-vg-lpips-report ran .amodal-cache/surface-metrics-env/bin/python scripts/build_shape_report.py --accounting. Job 45339 completed in two seconds. The installed builder digest is ea55955efbdc808e606bb5b6d8c2353295adb350fcb170bded4b9fedbc64ea37. The report and its retrieved copy matched SHA-256 35e5a14e3df1cf3d1fe3d4c122f1fdc685003c3e79f59114c491557e0c6ab46e. Verified the displayed values and intervals, 54 local links and all three embedded PNGs. Removing the new Vietnamese LPIPS subsection reproduces the previous report digest exactly. GPU job 44946 is running, and 44948 is pending on its prerequisite. No control allocation was submitted.

2026-11-22 05:46 UTC. Joined the VG LPIPS and SSIM rows in official order to examine byte-identical scoring crops among joint successes. For every paired success, verified the historical native target record against its execution manifest and compared its evaluation_reference_artifacts.crop with the Shape crop SHA-256 from the SSIM replay. All 584 baseline and 585 Surface shared cases overlapping the earlier native reference audit agreed on crop membership. The ordered ID SHA-256 values, computed from newline-joined identifiers without a trailing newline, were 01ff0aabd2759ab3998897c678a67cfba253bb59acd31377428743f324209c42 for PMC (566 targets, 551 images) and efb3a23a8c04ecd201e7707cc946f075a8c2330137f796d0e2824fb55b3c63ec for Surface (568 targets, 552 images). Scores appear in discoveries.md. The 05:46:36 UTC cluster probe counted 1,691 Shape records and 204 combined records. Slurm showed 44946 running on four L40S GPUs and 44948 pending. The severe-occlusion elephant was still outstanding.


## CS331 presentation rubric ? 22 November 2026

Built the 28-page Vietnamese PowerPoint and PDF from the 21-page November 21 PowerPoint. The original presentation package (subsequently retired) includes the editable slides, PDF, one-page rubric comparison and offline candidate-selection demo. Updated the Shape, combined, controlled and cohort results from the closed records. Read-only cluster checks used `ssh daoqm@slurm.csse.rose-hulman.edu 'squeue -h -j 44946,44948'` and compared the remote report hash. Job 44946 was running and 44948 dependency-pending. For each of VG-020-2beacd8b40 and VG-007-23ffa2bcf0, used `scp -r daoqm@slurm.csse.rose-hulman.edu:Vu_Hung/cv_nangcao/output/surface-confirmation128-20261110/full/candidates/<case_id> <temporary-pool-directory>`. Verified all 16 RGB and mask digests against their generation records. Chrome reproduced the eight policy winners and native mask IoUs. Inspected all 28 PDF pages, checked the page numbers, speaker notes, relative demo links, table values and ZIP contents. PowerPoint found zero text boxes exceeding their bounds.

2026-11-22 05:54 UTC. Reconciled the actual Week 8 presentation order in the November 21 PPTX against the sections for slides 11, 12 and 15. The disjoint 183-object training receipt, checkpoint, training code, Shape policy and implementation digests match the frozen 204-target runs for both new methods. Verified eight seed slots in each and native visible RGB restoration in combined.

Recovered the existing Shape plate's six decisions, 66 binary images, two first-pass RGB candidates and original SAM mask from the cluster. Verified their SHA-256 against the target trace and reference record. The local replay recovered all six selection indices directly from candidate-mask IoU with the prior, projected masks to original coordinates and compared them with the independent COCOA outline. It verified the crop offset against the original SAM mask on each pass. A separate NumPy integer pixel-centre projection reproduced all 180 intersection, union and hidden-overlap counts for the six sets of ten masks exactly. The original mask resize assigns the central 512-to-879 nearest-neighbour tie to the lower sample. The result explains one previously highlighted severe case.

Staged scripts/replay_shape_plate.py using scp to scripts/.replay_shape_plate-0fff877d.py and scripts/build_shape_report.py to scripts/.build_shape_report-0e62c691.py. Verified staged and previous code and report digests before installation. From the remote project submitted sbatch -p cpu -c 1 --mem=3G --time=00:10:00 -J shape-plate-selection, running scripts/replay_shape_plate.py followed by scripts/build_shape_report.py --accounting in the surface-metrics-env. Job 45340 completed in four seconds. Installed replay SHA-256 0fff877d198eb08511c51a193825042c2dbd0d6c66c083ffa15ed1eb5ea4a65f and builder 0e62c691d03e939badaaf25dae256aa0abf5f75d58b5820057263c23745bdbf9. Recovered the published figure, receipt and report. Independently generated local and remote measurements matched field for field. The Windows and Linux renderers produced different figure bytes. The inspected published 1705 by 1234 PNG is 350f3a2c6ae8407742038d663d0836ea11e3f19f1071cd2027bfa9d6e6c85683, the receipt is 108ec185829cc1852a2dbd0ec9ed382d44754e49c0a65a3ebd0d32eb7d7d5cf0 and REPORT.md is 5f728666d65dbc151f408060bf8566694094cd758f3e207a9532519703b109ae. All 56 local links resolve. Removing the figure and its introductory paragraph restores the prior report bytes. The first test run caught an incorrect hidden-pixel count in the synthetic test fixture. With its denominator corrected, all 36 focused tests passed.

The final WSL scripts/check.sh invocation passed 319 Python tests, the Bash regressions and the CPU smoke check using the surface-replay-py310 Python. At 05:55:34 UTC Slurm showed Shape allocation 44946 running, with 1,706 target records. Combined 44948 was dependency-pending with its 204 retained records. Neither full verification receipt existed.

2026-11-22 05:58 UTC. Started local recovery of the native masks needed for the closed 321-target COCOA audit. A read-only SSH probe hashed the finished target records against the fixed annotation inventory at 05:57:32 UTC, and the remote run configuration hashes matched their respective frozen cohort snapshots. Retrieved missing records and RGBA files using scripts.replay_shape_outlines.ensure_files, reusing byte-identical local cohort files. All 136 available Shape records and all 22 available combined records agreed with official target IDs, image SHA-256, and the target seeds in the registered PMC execution. For 73 saved Shape and 7 saved combined outputs, PIL original-frame mask intersection and union matched an independent clipped NumPy projection exactly. Both methods' 22 retained polygon scores reproduced the verified cohort. The final replay will recheck the same files against the registered executions and paired audit.

In the frozen 20-point occlusion bins, the Shape snapshot covered 85/203, 19/54, 20/39, 9/21 and 3/4. Confirmed the 211 unseen polygon targets come from 202 distinct images. The remaining highest-bin target was elephant 073268. The incomplete inventory reflects worker order. No partial-bin average was added to the report. Shape snapshot record map SHA-256 4eb19fa7dd2a383b3571d2a2429adf8bdd33f1b07878865ac1a4d9708e2a7f54, overlap map 749c0ae870e79fb020890da807b3214c77b1dcf9cbd48452e823b3f6321daa75. Combined record map d59d22c03b2e0421427ba39d3598e3b24ccf7f1f87a989fb36af710d14286ef7, overlap map 864cb89f806bc49714912743387e9024d707236b467226272333d255eb920118. Maps use compact sorted-key JSON, with each overlap row storing result digest, completion digest or null, intersection, union and IoU. Collected files remain under the respective local full-run target directories.


## Current Canva presentation, 22 November 2026

Read the saved Canva presentation DAHbEx8zTOs, revision 28, before editing. It has 22 pages including the students' cover and the surfboard example. Synchronized the [editable PowerPoint](output/bao-cao-cs331-20261122/Bao_cao_CS331.pptx) from the live page order, elements, text and page previews, reusing the vector graphics from an earlier Canva PowerPoint export. The local presentation includes the two draft changes. All 22 pages and every editable text element were checked against Canva, with visual review of the two changed pages. Page 2 uses the original PD-MC paper's introduction and Figure 6. Page 4 uses the three examples at the top right of Figure 1. The [two-page preview](output/bao-cao-cs331-20261122/Xem_truoc_trang_2_va_4.pdf) contains Canva's rendered drafts. The Canva transaction is 7803009297784358162. User preview approval is pending. The superseded 28-page PowerPoint, PDF, ZIP and grading checklist were sent to the Windows Recycle Bin. The standalone demo remains in the presentation directory.

2026-11-22 06:09 UTC. Retrieved five more completed Shape COCOA records for the final outline replay using the existing checksum-verified archive transfer. For the three newly saved completions, explicit clipped alpha coordinates and PIL projection gave identical original-frame intersection and union counts. The other two records carry the protocol's failure penalty. The local full-run directory now contains 141 of 321 polygon records and 76 native completions from the 06:08:47 UTC snapshot. The source annotation digest, frozen run configuration, image checksums and retained seeds matched. Shape had 1,729 full target records at that snapshot. The combined job was waiting on Shape.

2026-11-22 06:15 UTC. The report now explains in Vietnamese which controls, candidate budgets and scoring references the natural-image comparisons use. The conclusions will show signed LPIPS reductions over 2,565 targets and amodal IoU gains over 321 polygon targets for each audited L40S method against registered PMC and Surface. The table appears for Shape as soon as its full historical-control audit is available. It takes the two paired intervals from the corresponding audit summary. An in-memory rendering check with actual archived Surface-versus-PMC scores reproduced the historical LPIPS reduction +0.0073 [0.0046, 0.0101] and IoU gain +0.0017 [-0.0139, 0.0169]. That calibration used no new predictions and its row was never written to the study report.

Uploaded scripts/build_shape_report.py by scp to scripts/.build_shape_report-b75f521b.py, compared staged and installed hashes, and installed the staged file by atomic rename. From the remote project, submitted sbatch --parsable -p cpu -c 1 --mem=3G --time=00:10:00 -J shape-control-report -o output/shape-20261121/progressive-l40s/shape-control-report-%j.log --wrap '.amodal-cache/surface-metrics-env/bin/python scripts/build_shape_report.py --accounting'. Job 45341 completed in two seconds on one CPU. The builder digest is b75f521bc2508b58e2e96293e7266ea5e07095296f7842c8973213371776db69. The local and published REPORT.md hash is efc5580407a5733aa960e351f391f7133c9d98727c2a9294889918437d94b10d. All 56 local evidence links resolve. Replacing the new Vietnamese protocol paragraph with its previous text restores the prior report hash. All 37 focused tests and the full WSL scripts/check.sh suite passed. At 06:15:08 UTC, Shape 44946 had 1,736 target records and combined 44948 was pending with its 204 retained results.

2026-11-22 06:20 UTC. Read the latest saved Canva presentation DAHbEx8zTOs with read_design and compared its presentation order with the 22-slide Bao_cao_CS331.pptx (SHA-256 d7e163effaeaf9a5d6fa033fc9ed706056025e248478d34d1a3097b640460f4c). The report referred to the former 21-slide presentation. Confirmed the current content of slides 10–13 and 15–22. Slides 11–13 and 16 still describe Shape or the combined method as unimplemented. Slide 17 has empty full rows, 18 uses 321 COCOA objects, 19 has empty umbrella and bus panels, and 20 has empty ablation rows. Slide 22 lists implementing Shape as future work. The page 15 reference pointing to full results on 16 is stale in the saved deck. The correct page is 17. Read the deck without editing Canva, the local PPTX or the outstanding presentation draft.

Updated the report's page references, its link to the saved presentation and the Vietnamese guide to which measurements go into each slide. The local final table replay now locates the COCOA severity section by topic. The regression exercised both the old and new page numbers and rejected altered cells. All 37 focused tests passed. The full WSL scripts/check.sh suite passed. Used scp to stage scripts/.build_shape_report-f9f93c4b.py and verified the installed builder and report digests before renaming. Submitted sbatch --parsable -p cpu -c 1 --mem=3G --time=00:10:00 -J shape-slide-order -o output/shape-20261121/progressive-l40s/shape-slide-order-%j.log --wrap '.amodal-cache/surface-metrics-env/bin/python scripts/build_shape_report.py --accounting' from the remote project. CPU job 45342 completed in two seconds. Builder SHA-256 f9f93c4b335af82866fc0fc5463da6fb295b91c46a39ba6d003c62547d9b4fce, report SHA-256 8e659cf61bd7e902301afa987fc47aa75b054b006b492732a15a0dd835280321. All 76 existing table lines matched exactly and all 56 local links resolved. Started the local updated collector with python -u scripts/collect_shape_results.py --watch as PID 26540, confirmed its live poll, final report and interval dependencies, then retired previous PID 7768. Its logs are output/shape-20261121/collector-deck22.log and collector-deck22.err. At 06:19:55 UTC it counted 1,753 Shape records and 204 retained combined records.

2026-11-22 06:31 UTC. Rechecked the saved Shape proposal and measured its difficult range on all 32 independently annotated controlled objects. Computed the union of naturally hidden and artificially occluded amodal pixels on the model input grid with both PIL masks and NumPy. Verified the 32 annotation and input hashes, candidate selection indices and uniform four-candidate IoUs. Nine crops met the inclusive 50–80% range, with zero meeting the original visibility and framing criteria. Updated the Vietnamese report with the subgroup outcomes and with the original fully framed bowl and cat occlusion fractions. All 37 focused tests and the full WSL scripts/check.sh suite passed. Uploaded scripts/build_shape_report.py to scripts/.build_shape_report-eb9d6ce4.py using scp, verified its SHA-256 and the deployed builder and report hashes, then installed the file. From the remote project, sbatch --parsable -p cpu -c 1 --mem=3G --time=00:10:00 -J shape-difficult-crops -o output/shape-20261121/progressive-l40s/shape-difficult-crops-%j.log --wrap '.amodal-cache/surface-metrics-env/bin/python scripts/build_shape_report.py --accounting' returned job 45343. The CPU job completed in two seconds. Builder SHA-256 eb9d6ce4d20eb950e2184813f892f834f4b51b5a00bce3ab0cfda87d4c8da742 and report SHA-256 225a23994e7e5bda422504df5be18f59827375a235da2b92a30afc90a7e0422e. Recovered the report with scp, checked all 56 local links, 76 table lines, and restored the previous report hash in memory by removing the additions.

At 06:29:26 UTC the local Shape outline audit staged 155 of 321 target records and 85 native saved completions, including 14 new records and nine new masks. Direct NumPy pixel projection matched PIL intersection and union counts for each new mask. The severe elephant remains outstanding. At 06:31:35 UTC, Shape and combined had 1,774 and 204 inference records and 86 and 87 native metric caches respectively. Slurm showed 44946 running and 44948 waiting for Shape. Used scontrol write batch_script for both jobs and verified that the queued combined script will copy Shape's references and verify and register full outputs. The queued script hashes matched. The two metadata exports and final report and interval jobs still depend on the full audits.

2026-11-22 06:38 UTC. Reconciled the 22-slide PowerPoint presentation order with the final Shape paragraph in plan.md. Measured native hidden-region IoU for the 204-target cohort's 22 independently annotated COCOA objects. Verified all four closed executions, all cohort IDs, all 88 result digests and each successful RGBA output against its native record. The 15 positive-hidden-area objects cover 13 images. Clipped NumPy alpha coordinates and PIL original-frame placement gave identical overlap counts. Recomputed all five paired image-bootstrap intervals using separate per-image difference lists. Wrote cohort-cocoa-hidden-iou.json, SHA-256 e121e3d4596d4a2b3032de9a7adc0da0a904c46a2cdfbb63f76681af1b8d7ffd. The two published cohort inputs and the historical native evidence remain unchanged.

Uploaded the receipt and report builder with scp to staged project paths, checked the previous builder, report and native source digests, then installed them. CPU job 45344 ran in the remote project using sbatch --parsable -p cpu -c 1 --mem=3G --time=00:10:00 -J shape-hidden-cohort -o output/shape-20261121/progressive-l40s/shape-hidden-cohort-%j.log --wrap '.amodal-cache/surface-metrics-env/bin/python scripts/build_shape_report.py --accounting'. It completed in two seconds. Builder SHA-256 99f67b7760616e0d7c9802fcfc484e60397dd3c1da051bf42cb789fc612e0a52. Retrieved REPORT.md SHA-256 6a4b2e9286b3cc05eba8416020e9ee58edfebd4689ed426ce4d6bfc6c616e2b5 and verified its displayed values and all 57 local links. Removing the single added paragraph restored the preceding report digest. All 76 existing table lines were preserved. At 06:38:07 UTC, 44946 was running with 1,784 Shape records, and 44948 remained dependency-pending on its 204 retained records. The worker partition places the outstanding severe elephant at target 475/644 on worker 02. Its last printed completion at the probe was target 429/644.

2026-11-22 06:44 UTC. Counted positive hidden annotations independently from every original-frame COCOA raster. Ran srun -v -p cpu -c 1 --mem=3G --time=00:10:00 -J shape-hidden-controls .amodal-cache/surface-metrics-env/bin/python - from the remote project via SSH, passing the replay source through stdin. CPU job 45345 completed in 15 seconds. For all 321 objects and both registered Quadro controls the replay checked the official order, execution hash, native target digest, completion RGBA digest for successes and historical whole-object IoU. PIL placement of native alpha and clipped NumPy coordinates produced the same hidden intersections and unions. Saved output/shape-20261121/full-cocoa-hidden-historical.json, SHA-256 c5790a1e052d40b516c06f2e89b94a2c939ad57d40e061ec111a6ffe691e50b4. Reproduced both 5,000-draw image-bootstrap comparisons locally with the independent paired_estimate function.

Extended the existing full comparison to compute hidden-region IoU from the same verified native masks. Empty hidden annotations are excluded and failed targets with positive hidden area receive zero. The original full-amodal and official appearance metrics retain their definitions. Added a Vietnamese supplemental comparison table using the 230 eligible targets and paired image intervals. Replayed all 88 cohort rows for four methods, and independently compared the historical Surface counts for all 22 polygon cohort targets with the full historical receipt. Exercised crop clipping, visible annotation outside the amodal mask, failed outputs, empty hidden regions, table signs and image counts. The 37 focused tests and the full WSL scripts/check.sh suite passed.

Transferred scripts/compare_shape_progressive.py, scripts/build_shape_report.py and the hidden-region control receipt by sftp, using temporary names in their destination directories. Checked the installed and staged hashes and confirmed audits 44947 and 44949 were still pending, then installed the files. Final compare script SHA-256 21950571cdd860a8018711024169fd310ac6f4e797e6a24f56e42b24ab9d1320; builder SHA-256 f9df3c5ba2ab6511390b1934bf6977cd422781190bb121f11fc937a57b074bdd. CPU job 45346 used srun -v -p cpu -c 1 --mem=3G --time=00:10:00 -J shape-verify-hidden .amodal-cache/surface-metrics-env/bin/python -. It completed in 18 seconds and verified 642 whole-object and 642 hidden-region results with the installed NumPy calculation, including the exact eligible and excluded cases. The accounting-enabled builder reproduced the current report digest 6a4b2e9286b3cc05eba8416020e9ee58edfebd4689ed426ce4d6bfc6c616e2b5 in memory. Full Shape and combined rows await closure. At 06:44:12 UTC, job 44946 was running, with 1,792 inference records and 86 native metric caches. Job 44948 was dependency-pending.

2026-11-22 06:45 UTC. Slurm job 44946 was running on gus with 1,796 Shape inference records. Job 44948 was waiting on its dependency with 204 retained records. Recovered nine newly completed COCOA records and five native RGBA completions by checksum-verified SSH archive transfer. The 06:45:56 UTC local Shape snapshot covers 164 of the 321 polygons, with 90 saved outputs and 120 targets with nonempty hidden annotations. The local combined snapshot includes the 22 cohort polygons, seven saved outputs and 15 nonempty hidden annotations. For every staged success, PIL original-frame placement and NumPy clipped coordinates gave identical whole-object and hidden intersection and union counts. Compared each arm's 22 retained records to its closed cohort execution and all 22 hidden-region rows to cohort-cocoa-hidden-iou.json, including the seven ineligible objects. Every record, whole-object IoU, hidden intersection, union and eligible IoU matched. Compact sorted JSON maps of target ID to result digest, whole-object overlap counts and hidden score record hash to e6c51c7ca004fded255cd4a16319b14147e20e369eeb205abfe4a763d851af6d (Shape) and 50268220465efb4c43678aa8379e0d40bdd54ae1a7dbc77cb24a5472f6a997d2 (combined). The severe elephant was still outstanding.


2026-11-22. Prepared the user-supplied canva_slide.webp as an editable slide for insertion after the cover in Canva design DAHbEx8zTOs. The reviewed source is DAHbnomXScc, transaction 4668503968576357910, with 20 text boxes, two replaceable portrait fills and 106 other native shapes. All 128 elements are unlocked. The preview is output/bao-cao-cs331-20261122/Xem_truoc_Boi_canh_trang_02.png. Destination transaction 6198601941846355926 contains the 22 updated footers and the full-results cross-reference to the future page 18. All 23 text changes were checked in the draft. Exact operation pending preview approval: save both drafts and insert page 1 of DAHbnomXScc after page 1 of DAHbEx8zTOs. The surfboard page becomes slide 3. The saved deck still has 22 pages. The earlier transaction 7803009297784358162 now returns not_found.

2026-11-22. The user approved and saved the editable context slide in the existing CS331 Canva presentation DAHbEx8zTOs. Both reviewed transactions committed. Canva insertion job 821aa983-de24-4ab7-a18c-53c494c5202a succeeded. Saved page PBvwWy0stW1W9sFM is slide 2, and the former second page PBYBhRL5bbKvX8bg is slide 3. Read the saved design and verified all 23 sequential footers, the results cross-reference to page 18, and the inserted slide's 128 unlocked elements. All 20 text boxes, both replaceable portraits and native shape positions matched the approved source. Edit link https://www.canva.com/d/ff2z8HQgh8ONUWt .

2026-11-22. Prepared slide 2 red-board replacement in Canva design DAHbEx8zTOs, transaction 1816599509754809146. User clarified that diving board means the existing red board before/after example on slide 3. Reused MAHbjoMOhkU and MAHbjtc1iDs in 210 x 210 square frames, reset crops to show the full images, and removed 44 old portrait annotation shapes/question mark. All other remaining slide 2 elements match the initial state exactly; 83 elements remain. Draft visually reviewed and shown inline. Canva edit-design requires explicit preview approval before commit; approval question is pending. Preview: output/bao-cao-cs331-20261122/Xem_truoc_trang_02_van_do.png. Receipt and complete draft: output/bao-cao-cs331-20261122/canva-slide2-board-draft.json. Saved presentation is still unchanged by this task.

2026-11-22. User approved saving the slide 2 red-board replacement. Committed Canva transaction 1816599509754809146 to DAHbEx8zTOs. Saved revision 32 has 23 pages. Reopened the saved design and verified its complete slide 2 element data matches the approved draft exactly: 83 editable elements, both original red-board images in square frames, old outline/question mark removed. The public thumbnail was stale; a fresh transaction render visually confirmed the saved replacement. Closed verification-only transaction 1681491522446072436 without edits. Saved preview: output/bao-cao-cs331-20261122/Trang_02_van_do_da_luu.png. Commit and verification receipts: canva-slide2-board-commit.json and canva-slide2-board-saved-verification.json in the same directory. Edit link: https://www.canva.com/d/WMkh6b6PvXbhpV3 . This completes the pending slide 2 image replacement.


2026-11-22. User explicitly requested deletion of slide 3. Deleted page PBYBhRL5bbKvX8bg from Canva DAHbEx8zTOs via merge job 15d964e7-253e-438d-92d2-b2ba667b9ad0 (success). Saved design now has 22 pages. The red board pair remains on slide 2. Draft transaction 4861032860694176687 updates 21 editable footers to the new /22 page order and the page 15 full-results cross-reference from 18 to 17. Exactly 22 text changes were verified; no other elements changed. These numbering edits await required Canva preview approval. Receipt and previews: output/bao-cao-cs331-20261122/slide3-removal/.


The cover footer was stored inside a shape and is now included in the same pending transaction 4861032860694176687. All 22 sequential footers 01 / 22 through 22 / 22 and the page 15 reference to results on page 17 are verified. Draft contains 23 text replacements in total. Final complete draft and operations: output/bao-cao-cs331-20261122/slide3-removal/final-renumber-draft.json. All 22 draft page previews inspected. Slide 3 deletion itself is already committed; only the numbering and reference corrections need preview approval.

2026-11-22. User approved the numbering corrections after deleting slide 3. Committed transaction 4861032860694176687 to Canva DAHbEx8zTOs. Reopened the saved design and verified all 22 footers (01 / 22 through 22 / 22), the removed page's absence, and slide 15's full-results reference to slide 17. Every saved page exactly matches the approved complete draft. Fresh saved renders of slides 1, 3, 15 and 22 passed visual review. Closed verification-only transaction 6375368235462951893 without changes. Evidence: output/bao-cao-cs331-20261122/slide3-removal/saved-verification.json and saved-slide-*.png. The deletion and all numbering corrections are complete; no presentation approval is pending. Edit link: https://www.canva.com/d/jGn0nMmVpGwLTXQ .

2026-11-22. Prepared six Vietnamese PD-MC paper/method slides for replacement of slides 3-8 in Canva DAHbEx8zTOs. Read all 22 slides and the source paper sections 3.2-3.6 and appendix A; removed training-dataset framing, kept dataset discussion for the later section, and added speaker notes. New editable source DAHboUU91IA (https://www.canva.com/d/r3cyhLQl6tmBVkz) has 100 text boxes and all six speaker notes. Source transaction 6299932939254693997 contains the final slide 4 clarification and removal of arrows; not committed. All six Canva previews visually reviewed; local PowerPoint reports zero text overflow. Original deck remains unchanged; title-only draft 338292932637170347 cancelled. Await explicit approval to commit source draft, delete destination pages 3-8, and insert all six source pages after destination page 2. Exact operations and final draft: output/bao-cao-cs331-20261122/pdmc-paper/status.json and final-source-draft.json. Dataset slides 9-10 and later sections will keep their existing numbers. After approval verify imported notes, content, dimensions and all footers in the destination.

2026-11-22. User requested original paper diagrams in the mechanism slides. Revised the pending six-slide PD-MC section: slide 5 uses Figure 2, slide 6 uses complete Figure 4, slide 7 enlarges Figure 4's intermediate mask/composite/denoising region, and slide 8 uses Figure 5. Original labels and image contents retained; Vietnamese explanations and citations surround them. Crop coordinates and hashes are recorded in output/bao-cao-cs331-20261122/pdmc-paper/paper-figures/figure-provenance.json. Imported revised source DAHbon-g-tQ, edit URL https://www.canva.com/d/-2G6nexcUdm_IkO. All 66 editable text boxes and all six speaker notes match the source PPTX; all six Canva renders visually reviewed; no PowerPoint text overflow. Closed verification-only transaction 7086936980517556279 and cancelled superseded draft 6299932939254693997. Current deck DAHbEx8zTOs is unchanged. Pending exact approval: delete destination slides 3-8 and insert all six pages of DAHbon-g-tQ after page 2. The current source needs no commit. Root pdmc-paper/status.json and paper-figures/status.json now point to this revision.


2026-11-22. Revised the pending PD-MC section introduction at the user's request. Slide 3 now uses the full original first page of the official CVPR PDF, authors and affiliations, CVPR 2024 Highlight, and 58 citations attributed to Semantic Scholar with date 22/11/2026. Count verified through a successful direct API lookup for ARXIV:2312.15540, linked to DOI 10.1109/CVPR52733.2024.00869; response saved in pdmc-paper/semantic-scholar-response-20261122.json. Highlight confirmed on the authors' project page and CVPR accepted-paper listing. PPTX page and notes XML for slides 4-8 are unchanged. Imported source DAHbolIp5_4 (https://www.canva.com/d/4j1eb7l_wL85iBk), superseding DAHbon-g-tQ. All 70 editable text boxes and all six notes match; all imported previews inspected, first-page image verified, zero PowerPoint text overflow. Closed verification-only transaction 1210865366569846539 without edits. Main deck DAHbEx8zTOs remains unchanged by this presentation task. Pending exact approval: delete destination pages 3-8 and insert pages 1-6 from DAHbolIp5_4 after page 2, preserving 22 pages and dataset slides 9-10. Evidence and preview: output/bao-cao-cs331-20261122/pdmc-paper/paper-intro/. Root pdmc-paper/status.json points to this source.


2026-11-22. User requested Vietnamese explanations with standard technical terms retained in English throughout the presentation, then said to continue working. Reviewed all 22 effective slides. Prepared 23 text replacements in main design DAHbEx8zTOs, transaction 4668503977709517503, on slides 2, 9, 10, 13, 15, 16, 17, 19, 20, 21, 22. Prepared 17 replacements in source DAHbolIp5_4, transaction 5390205818087078048, for destination slides 3, 4, 5, 7, 8. Terms include amodal/visible/intermediate mask, fine-tuning, co-occurrence bias, inpainting, outpainting, segmentation, denoising, U-Net features, ground truth, test set, baseline, features, linear ranker, reranking, 95% CI, GPU-hours and ablation study. Vietnamese explanatory prose remains. All 40 replacements verified; original figures, other elements, numerical results, notes and page ordering preserved. Canva auto-sized some text boxes; all 22 effective slide previews visually checked and the compact slide 16 Visible mask label kept on one line. Neither transaction has been committed. The user's terminology request was a revision, not approval of the prior page replacement. Pending preview approval: commit both terminology drafts, delete main pages 3-8, and insert pages 1-6 of DAHbolIp5_4 after main page 2. Evidence and full slide-navigation preview: output/bao-cao-cs331-20261122/pdmc-paper/terminology/. Root status.json contains current transactions and exact next operations.


2026-11-22 11:05 UTC. Completed the approved PD-MC Canva presentation revision after the user said "Approve saving". Source terminology transaction 5390205818087078048 and main transaction 4668503977709517503 committed successfully. Replaced slides 3-8 in DAHbEx8zTOs using source DAHbolIp5_4: deletion job ecbbdd11-dca3-4da3-adbe-099db32b8e9d and insertion job b4ae90db-18a1-4f2a-b2b7-a1c8335cc984. Canva inserted the section in reverse order; five successful single-page moves restored the approved order, with final repair job 3f94704a-2d4a-414f-93b5-863b92c2a9b8. Do not repeat these operations.

The saved 22-page deck exactly matches the approved effective content, notes, media references, geometry and sequential footers. Dataset pages remain 9-10. Slide 3 shows the original first page of Amodal Completion via Progressive Mixed Context Diffusion, authors/affiliations, CVPR 2024 Highlight and 58 citations from Semantic Scholar, checked 22/11/2026. Slides 5-8 reuse original Figures 2, 4, enlarged 4, and 5. Vietnamese explanations retain standard English technical terms. Fresh transaction thumbnails of saved slides 3-8 were visually inspected: correct order, first-page image and figures present, no clipping observed. Earlier cached thumbnails were stale; authoritative fresh renders are in terminology/saved-fresh/ and were also copied over the six stale previews in terminology/saved/. All 22 approved draft previews had already been visually reviewed.

Closed verification-only transaction 8020308003786434906 with cancel, without edits. Evidence: output/bao-cao-cs331-20261122/pdmc-paper/terminology/saved-final-ordered.json, saved-verification.json, saved-fresh-renders.json and saved-verification-close.json. Reconciled root pdmc-paper/status.json and terminology/status.json to completed_saved_and_verified, clearing pending transactions and next operations. This completion supersedes earlier pending-approval entries for this PD-MC revision. No presentation approval or edits remain pending. Saved presentation: https://www.canva.com/d/nrt37n3w0JoLiVc .


## Open-World dataset slides prepared in Canva (22 November 2026)

User requested at least three dataset-description slides with examples, excluding the PDMC dataset. Four editable Vietnamese slides are saved in Canva source design `DAHbpIYx2P0`: https://www.canva.com/d/h8kki1t55Op0jS- . They cover verified Open-World counts (2,379 images / 2,565 targets / 553 labels), real examples from all four sources, a two-target annotation example, and saved SAM visible references with the hidden-RGB ground-truth limitation. All 78 imported text boxes and all four speaker notes match; local and Canva previews reviewed, zero PowerPoint overflow. Artifacts and evidence: `output/bao-cao-cs331-20261122/dataset/`.

Main deck `DAHbEx8zTOs` remains at 22 pages. Pending explicit approval requested via async question: replace old slides 9?10 with source pages 1?4 after page 8, producing 24 pages; update page numbering and the retained internal page-17 reference to page 19. Canva merge requires approval for the exact operations. `dataset/status.json` contains the prepared arguments. All inspection-only transactions were cancelled without edits. Source design is saved; do not create another import.


## Open-World dataset section saved in main Canva deck (22 November 2026)

User approved replacement and numbering. Design `DAHbEx8zTOs` now has 24 slides: https://www.canva.com/d/ownIH_xzivnkQRH. Four Open-World dataset slides occupy pages 9–12, replacing the unused PDMC synthetic dataset slides. Real examples, all 78 source text elements and all four source notes are preserved. Updated all page numbers and page 17's reference to page 19. Canva required separate merge requests and inserted pages in reverse order; corrected to approved source order. Cross-reference block was normalized to 25 px following the tool's rich-text flattening. Committed transaction 5519684317569611692. Saved verification checks all 24 pages, text, notes, artwork and numbering. All read-only transactions closed. Evidence: `output/bao-cao-cs331-20261122/dataset/main-saved-verification.json`; current status: `dataset/status.json`. This supersedes the earlier pending-approval entry.


## Human preference reviewer handoff (22 November 2026)

Prepared sites/amodal-preference and output/human-preference-20261122 for the user's personal full Surface-aware versus annotated PD-MC comparison. Shape and combined remain deferred, in separate future rounds. Frozen contract: configs/research/human-preference-20261122.json. Offline recovery and all 6,935 image checks passed; launch output/human-preference-20261122/Start review.cmd. Browser QA passed with synthetic TEST_ONLY responses in an isolated qa- study. No human results are claimed.

The official source helper verified commit 2ee883f9b4c6c9fdd5f409cbd12c15f6f2d7150b in the existing owner-private Site appgprj_6b02fe6d29848191b670dd42ed2ab89d. Publication state and exact version/deployment IDs are in output/human-preference-20261122/deployment.json; resume that deployment if nonterminal instead of saving a duplicate version. Native archive upload was rejected solely for its 512 MiB limit; source-based remote build was accepted. The archived local-source-validation.json matches all 6,944 source files. No downloader or batch uploader remains active; do not rebuild the frozen images.

The user should export JSON after each rating session. Import later with scripts/summarize_preference_study.py into a new response output directory. Preserve original JSON, study ID, independent coverage, single-rater disclosure, and the original defined metrics. No Slurm jobs or experiment changes were made by this task.


2026-11-22 15:54 UTC. Human preference reviewer publication completed: https://amodal-preference-dv20261122.rompingsalmon2.chatgpt.site
The existing owner-private Site now serves the compact folder-based reviewer from source b0c8bd34ba624002dbe951eb56f693b3ad29cb54 (saved version 2). Open study folder must select output/human-preference-20261122/dist once per hosted session; all images stay local and are checksum-verified. The offline launcher works without folder selection. All 6,935 images are locally validated, and the updated browser test passed the folder-required state, voting, disclosures, ties, skips, backup export/restore, wrong-round rejection and mobile layout with no console exceptions. This supersedes the earlier publication-in-progress entry. The failed full-image remote-build receipt and original archive/source remain preserved in site-tools; deployment.json is the successful final receipt. No publishing, download, or validation work remains pending. The next input is the user's ratings; Shape and combined remain separate future rounds. No human preference scores have been produced.


2026-11-22 16:07 UTC. User explicitly resumed the Shape experiment. Confirmed collector PID 26540 alive and queried Slurm. Job 44946 timed out after 24 hours with 2,516 completed records and 86 cached scores. Snapshot 45134 and controller 45142 completed, archived two partial targets and queued 45361 with controller 45362. The archive receipt hash is 9a5fbb8b89ea427ffa8ad99ef454b06f5d8e4989e8ec3132ff9ada2a2754fa82. The continuation initially waited because its 24-hour request crossed maintenance root_1, from 2026-11-23 13:00 UTC through 2026-11-25 13:00 UTC. Five L40S GPUs were available. Executed scontrol update JobId=45361 TimeLimit=06:00:00 and scontrol update JobId=44948 TimeMin=06:00:00. Shape started at 16:06:43 UTC on gus with four L40S GPUs. Combined can receive an allocation between six and 24 hours. These are scheduling adjustments to the existing jobs. The commands, target inventories, frozen configuration, references and registered historical controls remain the existing experiment. Recovery guards and audit dependencies were verified. Both full registrations are still pending.


2026-11-22 16:09 UTC. Read the latest saved Canva DAHbEx8zTOs presentation, now 24 slides. Dataset pages occupy 9–12. Methods are 13–15 and 18, full results 19, severity 20, qualitative 21, ablation 22, limitations and conclusions 23–24. The full results and Shape/combined placeholders remain open. Updated the report builder and existing heading assertions for these page references. All 37 focused tests passed. Installed builder SHA-256 9b1f6e810ddd7710ccf46e98c1f370d70bc5847fc765682277800242a838f77a after checking the prior and staged hashes. From the remote project submitted sbatch --parsable -p cpu -c 1 --mem=3G --time=00:10:00 -J shape-deck24-report -o output/shape-20261121/progressive-l40s/shape-deck24-report-%j.log --wrap ".amodal-cache/surface-metrics-env/bin/python scripts/build_shape_report.py --accounting". Job 45366 completed in two seconds. Recovered report SHA-256 b23446f05222472a0d1d6da1e6a502294d367b3b77dca1f101daa89a35932576, with all 76 table lines preserved and all 57 local links resolving.


2026-11-22 16:16 UTC. The full WSL scripts/check.sh suite completed successfully, including Python tests, Bash regressions and CPU smoke. Staged all 321 frozen Shape COCOA records and 181 saved completion/mask pairs for final replay. Two independent original-frame projections agreed, and the existing natural_geometry routine matched the projected alpha. Recorded the full and unseen geometry estimates in shape-cocoa-preverification.json, SHA-256 9da6abc07c70b2c8648b15d8f3969ddd9048c507e9e7514481a2c90759490ffe. Full registration remains pending. Reconcile this receipt with the final Shape paired audit.


2026-11-22 16:20 UTC. Set a recurring 30-minute follow-up bound to this thread to continue the authorized experiment through final verification. Scheduled task identifier is scheduled-task:command:mcp:b56f24ad-8db3-4819-bbff-33187cec6ccf:schedule-task:shape-resume-20261122. Its first run is 2026-11-22T16:50:20.321Z, or 23:50 Vietnam time on 22 November. The callback inspects real Slurm state, the live collector, both registrations, exports, native geometry, reference intervals, scoring and allocation costs, then verifies the final report and removes the schedule upon completion. It also reconciles shape-cocoa-preverification.json with the final Shape audit. During confirmed maintenance, defer the same scheduled task until the reservation ends. The schedule is enabled and bound to the existing thread with freshSession false. The user resumed the experiment even though the legacy goal store still reports its handoff pause; the scheduled callback carries the resumed work.

All 22 retained Shape cohort outline rows and all 15 eligible hidden-region scores agree with the new 321-target native replay. At 16:19 UTC the collector was live with 2,519 Shape records. Combined retained 204 records while awaiting Shape. Final full verification remains outstanding.


2026-11-22 16:51 UTC. Scheduled continuation confirmed Shape 45361 running on gus after 44 minutes, with 2,542 of 2,565 inference records. Workers 00, 01 and 03 had finished their inventories; worker 02 was at 621/644. Full scoring still had the 86 retained success caches. Independently hashed all 2,516 pre-timeout result records and all 86 frozen metric caches against shape-timeout-44946.json; every digest matched. Combined 44948 remained dependency-pending with its 204 retained records and 87 cached scores. Audits 44947/44949, exports 45303/45304 and interval replay 45320 retained the expected live dependencies. Collector PID 26540 was alive, polling successfully with an empty error log. Neither full verification receipt nor registry existed. Report and 321-target geometry preverification hashes still matched the reviewed versions. This is a verified wait on live inference; the recurring follow-up remains enabled.


2026-11-22 17:21 UTC. Scheduled follow-up confirmed Shape continuation 45361 running on gus. At 17:20:43 UTC it had 2,552 records, and the inventory probe at 17:21:07 found only 12 targets remaining, all from LAION. This implies 2,553 completed inference records. Worker 02 was actively updating its diffusion progress log at the observation time. The other three workers had finished. The current execution contains exactly 2,565 unique ordered target IDs and every completed directory belongs to that inventory. Full scoring has the 86 retained caches; Shape and combined full verification receipts and registrations remain absent. Combined 44948 and all audits/exports remain dependency-pending. Collector PID 26540 is live with successful polls and an empty error log. This is a verified wait on the existing running allocation. The recurring follow-up remains enabled.


2026-11-22 18:00 UTC. Continued the resumed Shape study. Slurm confirms Shape continuation 45361 COMPLETED (01:33:23), audit 44947 COMPLETED (00:11:54), export 45303 COMPLETED (00:00:21), and timeout controller 45362 COMPLETED. Combined 44948 is RUNNING on gus, four L40S GPUs, from 17:40:20 UTC until 23:40:20 UTC under its six-hour allocation. Snapshot 45135 waits afterany:44948; controller 45143 waits afterok:45135. Maintenance root_1 remains 23 November 13:00 UTC to 25 November 13:00 UTC. No inference job was submitted, canceled or changed during this check.

Collector PID 26540 remained live with an empty error log and finished both Shape local receipts. Verified metadata export 59d16364d3b740d50c93fb5021e746daf3d926b4fd277bb2650c8e4801e97c7e, all 7,074 member files and 2,961 candidate decisions (four fallbacks). Its full native replay covers 321 outlines and 181 saved completions. Added and ran output/shape-20261121/reconcile_shape_full.py. It verifies official ordered IDs, all three native score tables, native reference records, the 1,136-target earlier reference snapshot, and all four Shape paired reference-subset LPIPS bootstrap estimates. It reconciles the full/unseen geometry and score prechecks, the 230 hidden-region endpoint, all 22 retained outline rows and 15 hidden-region rows, 2,017 unseen targets / 1,900 images, 211 unseen polygon targets / 202 images, final VG scores, five severity rows and actual sacct allocation cost. It also checks the new report's Shape benchmark row, both severity tables and all 61 local links. Receipt: shape-full-local-reconciliation.json.

Separately verified all 204 retained result hashes and exact CSV rows, all 2,516 timeout result hashes and all 86 retained score-cache hashes. Combined's reference transfer receipt covers 10,261 files, including 9,444 new copies; full/crop/mask hashes match the final Shape pairs for all 2,565 targets. Receipt: shape-preservation-and-combined-reference-replay.json. The first local transfer-map lookup used an incorrect filename suffix; inspecting the actual crops/ and masks/ paths corrected that local audit assumption, and the completed receipt passed. Experiment files were never changed.

Read the saved Canva DAHbEx8zTOs research slides again: still 24 pages and still old Shape placeholders. Methods remain 13?15 and 18, full results 19, severity 20, examples 21, ablation 22, limitations/conclusions 23?24. No presentation edit was made. Current REPORT.md hash remains 986828aa23c21e345b7399a78daf7662bc29f37378f62f8d26da7ae5236d76b1. Full Shape is complete; combined registration, full audit/export, all-eight reference interval replay, final figures/tables and completion audit remain outstanding. Existing collector and 30-minute scheduled follow-up stay active.

2026-11-22 18:01 UTC. Updated the existing scheduled task prompt with the closed Shape result and local receipts, preserving its 30-minute interval, current-thread binding, freshSession=false and enabled state. Confirmed next run 2026-11-22T18:20:29.449Z (01:20 Vietnam time on 23 November). The full Shape LPIPS chart passed html_preview (318 px, no console errors) and was displayed with html_render. Combined and final completion audits remain pending.


2026-11-22 18:21 UTC. Scheduled check confirmed combined 44948 running on gus after 40 minutes, with all four current worker logs active. Read 269 completed records in the official 2,565-target inventory. Every completed input hash, seed and scoring-reference set agrees with registered Shape; all 204 frozen combined records match their cohort hashes. Current records comprise 119 saved, 57 no_completion, 59 ambiguous_target and 34 not_detected. This is incomplete inference, not a scored full result. Existing snapshot 45135, controller 45143, audit 44949, export 45304, reference replay 45320 and report 44950 retain their valid dependencies. Collector PID 26540 is alive with an empty error log. Combined verification and registry are absent. All closed Shape reconciliation input hashes and REPORT.md are unchanged. Read current Canva research text after its metadata timestamp changed: still 24 slides and the same Shape/combined placeholders; no page-mapping or research-content change was found. No job submission, restart, cancellation or time-limit change was needed. Receipt: output/shape-20261121/monitoring/20261122T182111Z.json. Live inference is the remaining prerequisite; the recurring follow-up stays enabled.


2026-11-22 18:51 UTC. Scheduled check confirmed combined 44948 RUNNING on four L40S GPUs, elapsed 01:10:28, with 294 completed inference records. Counts: 132 saved, 60 no_completion, 67 ambiguous_target and 35 not_detected. All four current worker logs were updated within 25 seconds and showed active diffusion with zero Python tracebacks. Their last completed shard positions were 26/651, 34/635, 27/644 and 16/635; these counters include previously retained targets and are not summed as fresh records. Snapshot/controller/audit/export/interval/report dependencies remain valid. Full verification and registration remain absent. Collector PID 26540 is alive with an empty error log. The completed Shape audit inputs and remote/local report hashes are unchanged. Current Canva research text still has the same 24-page mapping and Shape placeholders; saved content snapshot at output/shape-20261121/monitoring/canva-research-current.json. No job or source change was necessary. Live-state receipt: output/shape-20261121/monitoring/20261122T185112Z.json. The verified wait is active combined inference, and the recurring follow-up remains enabled.


2026-11-23 07:16 UTC. Scheduled continuation could not reach the cluster: SSH alias hpc failed DNS resolution for slurm.csse.rose-hulman.edu. The PANGP university VPN adapter is disabled; Wi-Fi is up. Both Google DNS-over-HTTPS and Cloudflare DNS returned no public A record, so no validated alternate host route was available. Collector PID 26540 is alive and retrying. Its last successful observation is 2026-11-22T18:51:18.840316+00:00, with 294 combined records and job 44948 then RUNNING. That observation is stale; current combined and controller states are unknown. The former six-hour end time has passed, but no timeout, successful recovery or new job ID is asserted without sacct/submissions.tsv. No job was changed or submitted. All closed Shape reconciliation input hashes remain valid; no combined registry or completed full audit has arrived locally. Latest Canva has 24 pages and the research content exactly matches the saved snapshot. The existing 30-minute follow-up remains enabled. Restore university VPN access before inspecting controllers or adjusting any pending continuation to fit before maintenance. Receipt: output/shape-20261121/monitoring/20261123T071644Z-connection-unavailable.json.


2026-11-23 07:20 UTC. User restored VPN; SSH hpc and collector PID 26540 recovered. Authoritative sacct confirms combined 44948 TIMEOUT after 06:00:09 at 22 November 23:40:29 UTC. Snapshot 45135 and controller 45143 each completed in three seconds, archived four partial targets, and submitted continuation 45377 plus controller 45378. Verified all 700 completed result hashes and 87 metric-cache hashes remotely. Collector recovered timeout receipt 95e61d427eb390025f4e0718c96e4e0b66110476c7f375f6acb77f665c34ed3f; local replay confirms all 204 cohort records and 87 cohort score-cache digests remain intact. The run configuration matches the timeout digest c4fa8d12316075af03a9cd9311960d238a7a90f93501a8026ab2aa51a65db890.

Continuation 45377 was pending with a 24-hour request that crossed maintenance. All six L40S GPUs on gus were allocated to other jobs; scheduled release of four is 08:44:27 UTC. From the remote project, after verifying pending state and expected allocation, executed scontrol update JobId=45377 TimeLimit=04:00:00. Slurm now reports Resources, estimated start 08:44:27 UTC and end 12:44:27 UTC (15:44-19:44 Vietnam time), before maintenance at 13:00 UTC. This is a prediction, not a confirmed start. Also executed scontrol update JobId=45378 TimeLimit=00:10:00 after checking its pending state and afterany:45377 dependency. This leaves enough requested CPU time in the expected 15-minute gap before maintenance; the prior snapshot/recovery each needed three seconds. Audit 44949 remains afterok:45377?afterok:45378. No new job, source change, retuning or duplicate control was submitted.

Updated plan and the existing scheduled task prompt with authoritative recovery IDs and current constraints. Kept the 30-minute interval and current-thread binding; next check is 07:43:59 UTC. Final combined scoring, registration, export and audits remain pending. Monitoring receipt: output/shape-20261121/monitoring/20261123T072057Z-combined-recovery.json.


2026-11-23 07:48 UTC. Confirmed combined continuation 45377 RUNNING on gus, four L40S GPUs, started 07:40:44 UTC with TimeLimit 04:00:00 and EndTime 11:40:44 UTC. Startup verified all 10,261 existing reference files and launched four workers under 20261123T074445.395461Z-3796770. The read-only continuation verifier recorded 707 records: 367 success, 150 no_completion, 125 ambiguous_target and 65 not_detected. All four worker logs had updated within five seconds with zero Python tracebacks. Execution 20261123T074432.962391Z-bc35fd90 has the same ordered 2,565-target inventory as registered Shape. All completed input hashes, seeds and scoring-reference digests match Shape. All 700 timeout records, 87 metric caches and 204 frozen cohort records match their preserved digests. Reference transfer 45377 has SHA-256 253a8c0bd98da26789e0f42235ac4005fd9df99197174d47576c4446f3a1dd45; its entire file map equals the earlier transfer, with zero new copies. Receipt: output/shape-20261121/monitoring/20261123T074828Z-combined-continuation.json; read-only verifier beside it.

Controller 45378 remains pending afterany:45377 with ten minutes requested. Audit 44949 retains afterok:45377?afterok:45378; export 45304, reference replay 45320 and report 44950 remain dependency-pending. Maintenance root_1 is unchanged at 23 November 13:00 UTC through 25 November 13:00 UTC. Final combined verification and registration do not yet exist. Collector PID 26540 is alive with an empty error log and successfully polled at 07:48:32 UTC. All closed Shape reconciliation input hashes and REPORT.md remain unchanged. Saved Canva research text also remains unchanged, with 24 pages and the established mapping. Captured its current snapshot without editing the presentation. No job submission, cancellation, retry, time-limit change or GPU experiment change was made. The live allocation is a verified wait; recurring follow-up remains enabled.

Updated the same scheduled task with the verified running state and continuation receipt, preserving its enabled 30-minute interval, current-thread binding and freshSession=false. Next check: 2026-11-23T08:14:03.522Z (15:14 Vietnam time). Full research completion remains pending.


2026-11-23 08:14 UTC. Scheduled check confirmed combined continuation 45377 RUNNING on gus after 34:02, retaining the configured 11:40:44 UTC end. Read-only verification counted 753 records: 393 success, 165 no_completion, 125 ambiguous_target, 70 not_detected. All four workers were active within 16 seconds, with zero tracebacks. The ordered 2,565-target inventory, all completed input/seed/reference values, all 700 timeout result hashes, 87 retained cache hashes and 204 cohort record hashes passed again. The continuation reference receipt remains unchanged across 10,261 files. Controller 45378 and audit 44949 have the expected dependencies; export 45304, reference replay 45320 and report 44950 remain queued. No later ledger ID exists, and full verification/registration remain absent.

Collector PID 26540 is alive with an empty error log. All seven closed Shape reconciliation input hashes and REPORT.md remain valid. Saved Canva metadata timestamp changed to 1795420048, so reread all research slides; their text and 24-page mapping exactly match the prior snapshot. No slide edit was made. Reviewed existing final-audit code: it checks transfer receipts for every preserved timeout plus the final job, and report allocation sums terminal cohort/full jobs including interruptions. No code or Slurm change was needed. Restored the supplemental hidden-region endpoint reminder in plan.md while updating the current count. Receipt: output/shape-20261121/monitoring/20261123T081446Z-combined-continuation.json. This is a verified wait on the existing live allocation; recurring follow-up stays enabled.


2026-11-23 08:44 UTC. Scheduled check confirmed combined continuation 45377 RUNNING on gus after 01:03:57, with the same 11:40:44 UTC end and four L40S GPUs. The read-only continuation verifier counted 802 records: 423 success, 173 no_completion, 130 ambiguous_target and 76 not_detected. All four workers updated within 14 seconds with zero tracebacks. Ordered inventory, completed input/seed/reference values, 700 timeout result hashes, 87 cached-score hashes and 204 frozen cohort records passed verification. The 10,261-file reference transfer is unchanged. Controller 45378 and all audit/export/reference replay/report dependencies remain intact; no later submission IDs appeared. Final combined verification and registration remain absent.

Collector PID 26540 is alive with an empty error log and a successful poll at 08:44:03 UTC. All seven closed Shape reconciliation input hashes and REPORT.md remain unchanged. Canva saved timestamp advanced to 1795421718; reread the research slides and confirmed identical text and the same 24-page mapping. No presentation change was made. Maintenance root_1 remains 23 November 13:00 UTC through 25 November 13:00 UTC. No Slurm or experiment-code changes were needed. Receipt: output/shape-20261121/monitoring/20261123T084440Z-combined-continuation.json. The enabled 30-minute follow-up remains bound to this thread with freshSession=false; next check 2026-11-23T09:14:09.141Z (16:14 Vietnam time).


2026-11-23 09:14 UTC. Scheduled check confirmed combined continuation 45377 RUNNING on gus after 01:34:07, retaining its 11:40:44 UTC end and four L40S GPUs. The read-only verifier counted 833 records: 448 success, 176 no_completion, 130 ambiguous_target and 79 not_detected. All four worker logs updated within 17 seconds with zero tracebacks. All ordered target IDs, completed input/seed/reference values, 700 timeout record hashes, 87 cache hashes and 204 frozen cohort record hashes passed. The 10,261-file continuation reference transfer remains unchanged. Controller 45378 and audit/export/reference replay/report dependencies remain intact; no later submission IDs appeared. Final combined verification and registration remain absent.

Collector PID 26540 is alive with an empty error log. All seven closed Shape input hashes and REPORT.md remain unchanged. Canva timestamp advanced to 1795423500, so reread the research slides; all wording and the 24-page mapping still match the prior snapshot. No presentation change was made. Maintenance remains 23 November 13:00 UTC through 25 November 13:00 UTC. No Slurm or experiment-code change was needed. Receipt: output/shape-20261121/monitoring/20261123T091450Z-combined-continuation.json. The enabled 30-minute follow-up remains bound to this thread, freshSession=false; next check 2026-11-23T09:44:12.277Z (16:44 Vietnam time).


2026-11-23 09:45 UTC. Original Shape study scheduled check confirmed combined continuation 45377 RUNNING on gus after 02:04:17, retaining its 11:40:44 UTC end and four L40S GPUs. The read-only verifier counted 879 records: 475 success, 183 no_completion, 137 ambiguous_target and 84 not_detected. All four logs updated within nine seconds with zero tracebacks. All ordered target IDs, completed input/seed/reference values, 700 timeout hashes, 87 cache hashes and 204 frozen cohort hashes passed. Reference transfers remain unchanged across 10,261 files. Existing controller and downstream dependencies remain intact; no later original-study submission IDs appeared. Final combined verification and registration remain absent.

Workspace notes now describe the separate Shape-v2 study and Slurm shows its training job 45632 running on gebru. Read the new instructions and preserved that work. Local provenance.py contains v2 support, so independently checked the remote project against combined's immutable execution source manifest. All 47 captured files match, with no drift; frozen digest 2d9ae393bdc7f14977147d3a364f3491540d7558a92556a079e1487aad42f302. Receipt: output/shape-20261121/monitoring/20261123T0946-frozen-runtime-source.json. No local or remote experiment source was changed by this scheduled check.

Collector PID 26540 is alive with an empty error log. All seven closed Shape input hashes and REPORT.md remain unchanged. Canva timestamp advanced to 1795425307, but research-slide wording and 24-page mapping remain identical. No presentation edit was made. Maintenance remains 23 November 13:00 UTC through 25 November 13:00 UTC. Receipt: output/shape-20261121/monitoring/20261123T094500Z-combined-continuation.json. Updated the same scheduled task prompt to this running state and added the independent-study separation/source-preservation finding, preserving enabled 30-minute checks, current-thread binding and freshSession=false. Next check 2026-11-23T10:14:15.121Z (17:14 Vietnam time).


2026-11-23 10:15 UTC. Original study follow-up confirmed combined continuation 45377 RUNNING on gus after 02:34:29 with the same four-L40S allocation and 11:40:44 UTC end. The verifier recorded 927 targets: 502 success, 195 no_completion, 140 ambiguous_target and 90 not_detected. All four worker logs updated within four seconds without tracebacks. All ordered IDs, completed input/seed/reference values, 700 timeout hashes, 87 cache hashes and 204 frozen cohort hashes passed. The 10,261-file reference transfer remains unchanged. Controller and downstream audit/export/interval/report dependencies remain valid; final combined verification and registration remain absent.

Extended only output/shape-20261121/monitoring/verify_combined_45377.py to independently recompute the frozen source-manifest digest, hash every captured source copy, and compare each corresponding live cluster file. The actual read-only probe passed for all 47 files, with zero changed files and frozen digest 2d9ae393bdc7f14977147d3a364f3491540d7558a92556a079e1487aad42f302. The check is now embedded in each continuation monitoring receipt. No production script, running experiment, Slurm job or independent Shape-v2 job was modified. The separate v2 chain remains outside this original-study follow-up.

Collector PID 26540 is alive with an empty error log. All seven closed Shape input hashes and REPORT.md remain unchanged. Canva timestamp advanced to 1795427119; reread research slides and confirmed identical wording and 24-page mapping. No presentation edit was made. Maintenance remains 23 November 13:00 UTC through 25 November 13:00 UTC. Receipt: output/shape-20261121/monitoring/20261123T101513Z-combined-continuation.json. The enabled 30-minute follow-up remains bound to this thread with freshSession=false; next check 2026-11-23T10:44:18.124Z (17:44 Vietnam time).


2026-11-23 10:44 UTC. Original study follow-up confirmed combined continuation 45377 RUNNING on gus after 03:04:10 with four L40S GPUs and the same 11:40:44 UTC end. The verifier counted 980 records: 533 success, 208 no_completion, 145 ambiguous_target and 94 not_detected. All four worker logs updated within eight seconds with zero tracebacks. All ordered IDs, completed input/seed/reference values, 700 timeout hashes, 87 cache hashes and 204 frozen cohort hashes passed. All 47 recorded source files and their immutable captured copies match; no remote source drift was found. The reference transfer is unchanged across 10,261 files. Controller 45378 and downstream dependencies remain intact; final verification and registration remain absent.

Collector PID 26540 is alive with an empty error log and successful poll at 10:44:12 UTC. All seven closed Shape input hashes and REPORT.md remain unchanged. Canva timestamp is 1795428931, but research-slide wording and 24-page mapping remain identical. No presentation, Slurm, production-code or independent Shape-v2 job change was made. Maintenance remains 23 November 13:00 UTC through 25 November 13:00 UTC. Receipt: output/shape-20261121/monitoring/20261123T104454Z-combined-continuation.json. The same 30-minute follow-up remains enabled and bound to this thread with freshSession=false; next check 2026-11-23T11:14:20.892Z (18:14 Vietnam time). Inspect the existing controller and ledger only after confirmed terminal state before any recovery action.


## Original Shape-Guided PMC recovery verified (23 November 2026, 12:21 UTC)

Confirmed combined 45377 TIMEOUT after 04:00:15, ending 11:40:59 UTC. Existing controller 45378 completed in five seconds and automatically submitted continuation 45653 and controller 45654. At 12:17 UTC the pending continuation was shortened to 30 minutes and its pending controller to five minutes, using scontrol update on those existing IDs. No new job was submitted by the assistant. Slurm confirmed 45653 RUNNING on gus from 12:17:21 UTC, four L40S GPUs, configured end 12:47:21 UTC (19:47 Vietnam time).

Read-only replay at 12:21:32 UTC verified execution 20261123T122049.520157Z-e221e354 and the ordered 2,565-target inventory. Its 1,085 records comprise 592 success, 233 no_completion, 156 ambiguous_target and 104 not_detected. These are the preserved prefix, not a newly scored full result. All four new worker logs were updated within seven seconds and had zero Python tracebacks. Every completed input, seed and scoring reference matches Shape. All 1,085 records and 87 caches from timeout 45377, all 700 earlier records and caches from 44948, and all 204 frozen cohort records remain byte-identical. The archived partial-target replay passed all eight directories across both timeouts: seven empty directories and the 22 files (3,771,295 bytes) for VG/2350254_bus. Original partial attempts remain archived while unfinished targets resume normally.

All 47 immutable captured source files and corresponding live cluster files match frozen digest 2d9ae393bdc7f14977147d3a364f3491540d7558a92556a079e1487aad42f302. The new transfer combined-refs-full-45653.json verifies 10,261 files across 2,565 targets with zero copies, identical to both previous full maps; SHA-256 03f6399389d700ae1a049b97fdf9634ed14e36e4f574112b87e40cbfc8caa5e0. Local and remote timeout-45377 receipt hashes agree at a4acdc71c2fc74836746e3b106206d3153190caa6478af4486c4b11fc4abc661. All seven closed Shape reconciliation inputs, including REPORT.md, remain valid.

Controller 45654 remains pending afterany:45653; audit 44949 waits afterok:45653?afterok:45654. Export 45304, reference replay 45320 and report 44950 remain dependency-pending. The terminal combined allocation subtotal is 47.358889 GPU-hours across cohort 44917 and interrupted jobs 44948/45377; this excludes ongoing 45653 and is not final cost. Maintenance root_1 remains 23 November 13:00 UTC through 25 November 13:00 UTC. Preserve the next pending continuation across maintenance and inspect the ledger/controllers before recovery. Running time-limit extensions remain disallowed by the earlier denial.

Collector PID 26540 is alive, with an empty error log and a successful 12:20:09 UTC poll; it was not replaced. Canva still has 24 pages and exactly the same research-slide text despite a newer saved timestamp. Its original-study mapping and placeholders are unchanged; no presentation edit was made. The independent Shape-v2 chain remains untouched. Added only a parameterized read-only monitoring verifier to avoid reusing hardcoded 45377 identities. This check passed against the real continuation; no production experiment source or scientific configuration was changed.

Evidence: [continuation and archive replay](output/shape-20261121/monitoring/20261123T122132Z-combined-continuation.json), [scheduling receipt](output/shape-20261121/monitoring/20261123T1219-combined-recovery.json), [read-only verifier](output/shape-20261121/monitoring/verify_combined_continuation.py), [saved slide text](output/shape-20261121/monitoring/20261123T122133Z-canva-research.json). The scheduling receipt's actual observed_at is 12:17:21 UTC despite its filename, and its immediate state was PENDING; subsequent sacct and scontrol confirm RUNNING. Combined full scoring, registration, exports, independent replays and final report closure are still pending; the follow-up remains enabled.

Updated the existing original-Shape follow-up with verified 45653/45654 state and all timeout/archive receipts, preserving its enabled 30-minute interval, current-thread binding and freshSession=false. Confirmed next check 2026-11-23T12:44:58.348Z (19:44:58 Vietnam time), before the configured 12:47:21 UTC timeout. The next check must inspect actual controller/ledger state before maintenance deferral. Schedule receipt: output/shape-20261121/monitoring/20261123T1221-followup-update.json.


## Original Shape study queued across maintenance (23 November 2026, 12:48 UTC)

Continuation 45653 reached TIMEOUT at 12:47:30 UTC after 00:30:09, adding 43 completed records since its 1,085-record starting snapshot. Existing controller 45654 COMPLETED in five seconds at 12:47:37 UTC and automatically submitted continuation 45666 and controller 45667. No new Slurm submission, cancellation or duration change was made by this scheduled check. Job 45666 retains its 24-hour request on four L40S GPUs with four workers; Slurm reports PENDING, ReqNodeNotAvail/Reserved_for_maintenance, with no predicted start. Controller 45667 waits afterany:45666; audit 44949 waits afterok:45666?afterok:45667. Export 45304, reference replay 45320 and report 44950 remain dependency-pending.

The final snapshot contains 1,128 result records: 619 success, 246 no_completion, 159 ambiguous_target and 104 not_detected. All 87 score caches, 204 frozen cohort records, earlier 700- and 1,085-record snapshots, and every completed input/seed/scoring-reference identity passed independent replay. Timeout receipt combined-timeout-45653.json has matching local/remote SHA-256 2e16eb5bad124d3ce0b67ce0e9f3be53a3d325feb2dccba97dea1f36fc3565b3. Its four archived partial targets also passed native file rehashing: plate 110 files, greenboat 88, airplane 22, bicycle empty. All 12 archived partial directories across the three timeouts are preserved. All three full reference-transfer maps agree across 10,261 files/2,565 targets, and all 47 frozen captured/live source files remain unchanged. The seven closed Shape reconciliation inputs, including REPORT.md, remain valid.

Original combined allocated GPU-hours so far are 49.368889: cohort 44917 plus interrupted 44948, 45377 and 45653. This includes 2.010000 GPU-hours for the final pre-maintenance slice and is not final study cost. Combined still needs 1,437 target records and official full scoring/registration. Only the retained 22/321 COCOA records are complete, so full native replay cannot yet proceed. The existing collector PID 26540 recovered the new timeout receipt and successfully polled at 12:48:02 UTC; its error log remains empty. It was not replaced.

The same read-only verifier now supports a queued recovery with explicit terminal predecessor and completed controller IDs. Verified using `python output/shape-20261121/monitoring/verify_combined_continuation.py 45666 45667 --after-timeout 45653 --after-controller 45654`. It checks terminal states through sacct, verifies the unchanged execution belongs to 45653, and confirms 45666 remains pending before emitting its receipt. This was monitoring-only code; no production experiment or independent Shape-v2 source/job was changed. Saved Canva still has 24 slides and identical research text at 12:45 UTC; no slide edit occurred.

Maintenance root_1 remains 23 November 13:00 UTC through 25 November 13:00 UTC, and Slurm explicitly prevents the remaining original run from starting. Preserve the queued 24-hour continuation and downstream jobs. Defer the same scheduled task to 25 November 13:05 UTC (20:05 Vietnam time), restoring its 30-minute recurrence at that check if work remains. Full original-study completion remains unachieved.

Evidence: [timeout, source and archive replay](output/shape-20261121/monitoring/20261123T124815Z-combined-continuation.json), [allocation and remaining-work receipt](output/shape-20261121/monitoring/20261123T1248-maintenance-readiness.json), [saved presentation text](output/shape-20261121/monitoring/20261123T124537Z-canva-research.json).

Confirmed schedule deferral: the same original-Shape task remains enabled and bound to this thread with freshSession=false, now a one-time check on 2026-11-25T13:05:00.000Z (20:05 Vietnam time). Its updated prompt requires restoring 30-minute recurrence if work remains. No experiment job was altered. Receipt: output/shape-20261121/monitoring/20261123-maintenance-followup.json. Full research goal remains unachieved pending combined inference and final audits.

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

